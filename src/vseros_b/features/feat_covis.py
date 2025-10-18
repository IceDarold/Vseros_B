# -*- coding: utf-8 -*-
"""
feat_covis.py — co-vis фичи: насколько кандидат «связан» с последними кликами пользователя.

Использует res.ensure_covis() → dict {item_j: {item_i: weight}} (по TRAIN).
Если словарь пуст — возвращает нули.

Выходные колонки:
- covis_hits        — сколько из последних K кликов имеют co-vis ребро к кандидату (вес > 0)
- covis_weight_sum  — сумма весов w(j->i) по последним K (без затухания)
- covis_weight_max  — максимум веса по последним K
- covis_weight_sum_decay — сумма весов с позиционным затуханием по истории

Настройка через ctx["features_cfg"]["feat_covis"] (опц.):
- lastK: int = 10
- pos_decay: float = 0.20   # экспоненциальное затухание по позиции (от конца): w *= exp(-pos_decay * rank_from_end)
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
    d = (ctx.get("features_cfg", {}) or {}).get("feat_covis", {}) or {}
    return {
        "lastK": int(d.get("lastK", 10)),
        "pos_decay": float(d.get("pos_decay", 0.20)),
    }


SPEC = FeatureSpec(
    name="covis_stats",
    cols=["covis_hits", "covis_weight_sum", "covis_weight_max", "covis_weight_sum_decay"],
    doc="Co-vis фичи кандидата относительно последних K кликов пользователя",
)


@register_feature(SPEC)
def build_covis_stats(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must have user_id,item_id"

    cfg = _get_cfg(ctx)
    lastK = int(cfg["lastK"])
    pos_decay = float(cfg["pos_decay"])

    covis = res.ensure_covis() or {}  # {prev_item: {next_item: weight}}
    uh = res.ensure_user_history(lastK=lastK)
    lastK_map: Dict[int, List[int]] = uh.get("lastK", {}) if uh else {}

    out = cand_df[[COL_USER, COL_ITEM]].copy()
    out["covis_hits"] = np.int16(0)
    out["covis_weight_sum"] = np.float32(0.0)
    out["covis_weight_max"] = np.float32(0.0)
    out["covis_weight_sum_decay"] = np.float32(0.0)

    if not covis or not lastK_map:
        return out

    # группируем по пользователю
    for u, g in out.groupby(COL_USER, sort=False):
        hist = lastK_map.get(int(u), [])
        if not hist:
            continue
        # индексы (1 — последний, 2 — предпоследний, …)
        L = len(hist)
        pos_from_end = list(range(L, 0, -1))  # [L, ..., 2, 1] для удобства
        decay_w = np.exp(-pos_decay * np.arange(L-1, -1, -1, dtype=np.float32))  # [e^-0, e^-d, …]

        idx = g.index.to_numpy()
        items = g[COL_ITEM].to_numpy()

        for i, it in zip(idx, items):
            hit = 0
            wsum = 0.0
            wmax = 0.0
            wsum_decay = 0.0
            # идём по истории
            for t, prev in enumerate(hist):
                nbrs = covis.get(int(prev))
                if not nbrs:
                    continue
                w = float(nbrs.get(int(it), 0.0))
                if w > 0:
                    hit += 1
                    wsum += w
                    if w > wmax:
                        wmax = w
                    wsum_decay += w * float(decay_w[t])
            out.at[i, "covis_hits"] = int(hit)
            out.at[i, "covis_weight_sum"] = float(wsum)
            out.at[i, "covis_weight_max"] = float(wmax)
            out.at[i, "covis_weight_sum_decay"] = float(wsum_decay)

    out["covis_hits"] = out["covis_hits"].astype("int16")
    for c in ("covis_weight_sum", "covis_weight_max", "covis_weight_sum_decay"):
        out[c] = out[c].astype("float32")

    return out
