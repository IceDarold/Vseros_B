# -*- coding: utf-8 -*-
"""
feat_markov.py — марковские переходы: вероятность перейти к кандидату из последних кликов.

Использует res.ensure_markov() → dict {prev_item: {next_item: prob}} (по TRAIN).
Если словарь пуст — возвращает нули.

Выходные колонки:
- markov_max  — max_j P(i | H_K[j])
- markov_mean — среднее P(i | H_K[j]) по последним K
- markov_last — P(i | last_item)

Настройка через ctx["features_cfg"]["feat_markov"] (опц.):
- lastK: int = 10
- pos_decay: float = 0.0   # если >0 — делаем взвешенное среднее по позиции (эксп. затухание)
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


def _get_cfg(ctx: dict) -> dict:
    d = (ctx.get("features_cfg", {}) or {}).get("feat_markov", {}) or {}
    return {
        "lastK": int(d.get("lastK", 10)),
        "pos_decay": float(d.get("pos_decay", 0.0)),
    }


SPEC = FeatureSpec(
    name="markov_basic",
    cols=["markov_max", "markov_mean", "markov_last"],
    doc="Марковские фичи: max/mean/last вероятности переходов к кандидату",
)


@register_feature(SPEC)
def build_markov_basic(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must have user_id,item_id"

    cfg = _get_cfg(ctx)
    lastK = int(cfg["lastK"])
    pos_decay = float(cfg["pos_decay"])

    markov = res.ensure_markov() or {}  # {prev_item: {next_item: prob}}
    uh = res.ensure_user_history(lastK=lastK)
    lastK_map: Dict[int, List[int]] = uh.get("lastK", {}) if uh else {}

    out = cand_df[[COL_USER, COL_ITEM]].copy()
    out["markov_max"] = np.float32(0.0)
    out["markov_mean"] = np.float32(0.0)
    out["markov_last"] = np.float32(0.0)

    if not markov or not lastK_map:
        return out

    for u, g in out.groupby(COL_USER, sort=False):
        hist = lastK_map.get(int(u), [])
        if not hist:
            continue

        L = len(hist)
        if pos_decay > 0.0:
            weights = np.exp(-pos_decay * np.arange(L-1, -1, -1, dtype=np.float32))
            weights = weights / float(weights.sum())
        else:
            weights = None

        last_item = int(hist[-1]) if L > 0 else None

        idx = g.index.to_numpy()
        items = g[COL_ITEM].to_numpy()

        for i, it in zip(idx, items):
            it = int(it)
            probs = []
            wsum = 0.0
            w_mean = 0.0
            # пробегаем историю
            for t, prev in enumerate(hist):
                d = markov.get(int(prev))
                p = float(d.get(it, 0.0)) if d else 0.0
                probs.append(p)
                if weights is not None:
                    w_mean += p * float(weights[t])
            # метрики
            if probs:
                out.at[i, "markov_max"] = float(np.max(probs))
                out.at[i, "markov_mean"] = float(np.mean(probs) if weights is None else w_mean)
            if last_item is not None:
                dlast = markov.get(last_item)
                out.at[i, "markov_last"] = float(dlast.get(it, 0.0)) if dlast else 0.0

    for c in ("markov_max", "markov_mean", "markov_last"):
        out[c] = out[c].astype("float32")

    return out
