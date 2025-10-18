# -*- coding: utf-8 -*-
"""
feat_ppr.py — фичи на основе PPR кеша (exp107):
  - ppr_score : score из кеша (0.0 если (u,i) не в top-N)
  - ppr_rank  : ранг в кеше (1..N), если доступен (см. compute_rank)

Ожидаемый формат кеша:
  FeatureResources.ensure_ppr_cache() -> dict[int, dict[int, float]]
  где cache[user_id][item_id] = score (чем больше — тем релевантнее).
Если кеш отсутствует — возвращаем нули.

Настройки через ctx["features_cfg"]["feat_ppr"]:
  - compute_rank: bool = True     # считать ppr_rank по локальной сортировке кеша пользователя
  - max_items_for_rank: int = 5000  # если у пользователя в кеше слишком много айтемов — ранги не считаем
  - missing_rank_value: int = 10_000_000  # sentry-ранг, если айтема нет в кеше
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


def _get_cfg(ctx: dict) -> dict:
    d = (ctx.get("features_cfg", {}) or {}).get("feat_ppr", {}) or {}
    return {
        "compute_rank": bool(d.get("compute_rank", True)),
        "max_items_for_rank": int(d.get("max_items_for_rank", 5000)),
        "missing_rank_value": int(d.get("missing_rank_value", 10_000_000)),
    }


SPEC = FeatureSpec(
    name="ppr_cached",
    cols=["ppr_score", "ppr_rank"],
    doc="PPR-скор из кеша + опциональный ранг в кеше (чем меньше, тем лучше)",
)


@register_feature(SPEC)
def build_ppr_cached(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must contain user_id,item_id"

    cfg = _get_cfg(ctx)
    compute_rank = bool(cfg["compute_rank"])
    max_items_for_rank = int(cfg["max_items_for_rank"])
    MISS_RANK = np.int32(cfg["missing_rank_value"])

    cache = res.ensure_ppr_cache()  # ожидаем dict[user_id] -> dict[item_id] -> score
    out = cand_df[[COL_USER, COL_ITEM]].copy()

    if not cache:
        out["ppr_score"] = np.float32(0.0)
        out["ppr_rank"] = np.int32(MISS_RANK)
        return out

    # предварительно подготовим:
    # 1) список уникальных пользователей из cand_df
    users = pd.unique(out[COL_USER]).astype("int64")
    # 2) по желанию — карты рангов в кеше (по убыванию score)
    rank_maps: Dict[int, Dict[int, int]] = {}

    if compute_rank:
        for u in users:
            umap: Optional[dict] = cache.get(int(u))
            if not umap:
                continue
            if len(umap) > max_items_for_rank:
                # слишком много айтемов — ранги для этого пользователя пропускаем
                continue
            # сортировка по score (desc), ранги с 1
            # item с одинаковым score — dense rank
            items_scores = sorted(umap.items(), key=lambda kv: (-kv[1], kv[0]))
            rnk = {}
            rank = 0
            prev_score = None
            for idx, (it, sc) in enumerate(items_scores, start=1):
                if prev_score is None or sc != prev_score:
                    rank += 1
                    prev_score = sc
                rnk[int(it)] = int(rank)
            rank_maps[int(u)] = rnk

    # основная петля — читаем ppr_score (и ранг, если возможно)
    n = len(out)
    sc = np.zeros(n, dtype=np.float32)
    rk = np.full(n, MISS_RANK, dtype=np.int32)

    u_arr = out[COL_USER].values
    i_arr = out[COL_ITEM].values

    for idx, (u, i) in enumerate(zip(u_arr, i_arr)):
        umap = cache.get(int(u))
        if umap is None:
            continue
        # score
        s = umap.get(int(i))
        if s is not None:
            sc[idx] = float(s)
        # rank (если сформировали для пользователя)
        if compute_rank:
            rm = rank_maps.get(int(u))
            if rm is not None:
                rk[idx] = int(rm.get(int(i), MISS_RANK))

    out["ppr_score"] = sc
    out["ppr_rank"] = rk
    return out
