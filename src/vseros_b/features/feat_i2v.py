# -*- coding: utf-8 -*-
"""
feat_i2v.py — похожесть кандидата к последним кликам пользователя по item2vec.

Выходные колонки (на пару user_id, item_id):
- i2v_cos_max       — максимум косинусного сходства к последним K айтемам пользователя
- i2v_cos_mean      — среднее косинусное сходство к последним K
- i2v_cos_last      — косинус к самому последнему айтему
- i2v_cos_top3_mean — среднее по трём наилучшим совпадениям (если меньше трёх — по имеющимся)

Настройка через ctx["features_cfg"]["feat_i2v"] (опционально):
- lastK: int = 10   — сколько последних кликов пользователя учитывать
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


def _get_cfg(ctx: dict) -> dict:
    d = (ctx.get("features_cfg", {}) or {}).get("feat_i2v", {}) or {}
    return {
        "lastK": int(d.get("lastK", 10)),
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _get_item_vec(model, item_id: int, cache: dict) -> Optional[np.ndarray]:
    """
    Пытаемся вытащить вектор айтема из разных форматов модели:
      - объект с полями item_vecs (np.ndarray [I,D]) и item_map (DataFrame {item_id, i_idx})
      - dict с теми же ключами
      - объект с методом get_vec(item_id) или get(item_id)
      - gensim KeyedVectors: wv.get_vector(str(item_id)) / model[str(item_id)]
    Возвращаем НОРМИРОВАННЫЙ вектор (cosine ready) или None.
    """
    if item_id in cache:
        return cache[item_id]

    vec = None

    # 1) классическая обёртка с item_vecs и item_map
    if hasattr(model, "item_vecs") and hasattr(model, "item_map"):
        try:
            mp = model.item_map
            if not isinstance(mp, pd.DataFrame):
                # вдруг словарь
                i_idx = mp.get(int(item_id), -1)
            else:
                i_idx = mp.set_index("item_id").get("i_idx", pd.Series(dtype="int64")).get(item_id, -1)
            if i_idx is not None and i_idx >= 0:
                vec = model.item_vecs[int(i_idx)]
        except Exception:
            pass

    # 2) dict-подобное хранилище
    if vec is None and isinstance(model, dict):
        if "item_vecs" in model and "item_map" in model:
            mp = model["item_map"]
            if isinstance(mp, pd.DataFrame):
                i_idx = mp.set_index("item_id")["i_idx"].get(item_id, -1)
            else:
                i_idx = mp.get(int(item_id), -1)
            if i_idx is not None and i_idx >= 0:
                vec = model["item_vecs"][int(i_idx)]
        elif item_id in model:
            vec = model[item_id]
        elif str(item_id) in model:
            vec = model[str(item_id)]

    # 3) метод get_vec / get
    if vec is None:
        for meth in ("get_vec", "get"):
            if hasattr(model, meth):
                try:
                    cand = getattr(model, meth)(int(item_id))
                    if cand is not None:
                        vec = cand
                        break
                except Exception:
                    pass

    # 4) gensim-like
    if vec is None:
        for obj in (getattr(model, "wv", None), model):
            if obj is None:
                continue
            for meth in ("get_vector", "__getitem__"):
                f = getattr(obj, meth, None)
                if f is None:
                    continue
                try:
                    cand = f(str(item_id))
                    if cand is not None:
                        vec = cand
                        break
                except Exception:
                    pass
            if vec is not None:
                break

    if vec is None:
        cache[item_id] = None
        return None

    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    out = _normalize(vec)
    cache[item_id] = out
    return out


SPEC = FeatureSpec(
    name="i2v_sim",
    cols=["i2v_cos_max", "i2v_cos_mean", "i2v_cos_last", "i2v_cos_top3_mean"],
    doc="Косинусные похожести кандидата к последним кликам пользователя по item2vec",
)


@register_feature(SPEC)
def build_i2v_sim(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must have user_id,item_id"

    cfg = _get_cfg(ctx)
    lastK = int(cfg["lastK"])

    model = res.ensure_i2v()  # может быть None — тогда вернём нули
    uh = res.ensure_user_history(lastK=lastK)  # {"lastK": {u: [items...]}}
    lastK_map: Dict[int, List[int]] = uh.get("lastK", {}) if uh else {}

    out = cand_df[[COL_USER, COL_ITEM]].copy()
    out["i2v_cos_max"] = np.float32(0.0)
    out["i2v_cos_mean"] = np.float32(0.0)
    out["i2v_cos_last"] = np.float32(0.0)
    out["i2v_cos_top3_mean"] = np.float32(0.0)

    if model is None or not lastK_map:
        return out

    vec_cache: Dict[int, Optional[np.ndarray]] = {}

    # группируем по пользователю, чтобы не гонять одинаковые lastK заново
    for u, g in out.groupby(COL_USER, sort=False):
        hist = lastK_map.get(int(u), [])
        if not hist:
            continue

        # векторы истории (нормированные)
        H = []
        for h in hist:
            v = _get_item_vec(model, int(h), vec_cache)
            if v is not None:
                H.append(v)
        if not H:
            continue
        H = np.stack(H, axis=0)  # [K, D]
        # предрасчёт для last
        v_last = _get_item_vec(model, int(hist[-1]), vec_cache)
        if v_last is None:
            v_last = np.zeros_like(H[0])

        # для каждого кандидата — косинусы к истории
        idx = g.index.to_numpy()
        items = g[COL_ITEM].to_numpy()
        for i, it in zip(idx, items):
            v = _get_item_vec(model, int(it), vec_cache)
            if v is None:
                continue
            # косинусы как скалярные произведения нормированных векторов
            sims = H @ v  # [K]
            out.at[i, "i2v_cos_max"] = float(np.max(sims))
            out.at[i, "i2v_cos_mean"] = float(np.mean(sims))
            out.at[i, "i2v_cos_last"] = float(np.dot(v_last, v))
            # топ-3 среднее
            k = min(3, sims.shape[0])
            if k > 0:
                topk = np.partition(sims, -k)[-k:]
                out.at[i, "i2v_cos_top3_mean"] = float(np.mean(topk))

    # типы
    for c in SPEC.cols:
        out[c] = out[c].astype("float32")

    return out
