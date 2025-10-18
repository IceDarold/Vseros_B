# -*- coding: utf-8 -*-
"""
feat_interact.py — компактный набор «сильных» интеракций поверх базовых фич.
Ожидаемые входные признаки (если их нет — используем нули/фолбэки):
  - rr_sum, votes_graph
  - lgcn_dot
  - i2v_cos_max
  - markov_max
  - overlap_lastK или pos_lastK (если есть только pos_lastK, считаем overlap = 1{pos_lastK>0})
  - trend_14_decay
  - days_since_last (если нет — считаем из user_history)
  - novelty
  - seen_in_train_user, u_i_cnt (если нет — берём из user_history)

Выходные колонки:
  - lgcn_dot__rr_sum
  - i2v_cos_max__rr_sum
  - markov_max__overlap_lastK
  - trend_14_decay__recent
  - novelty__votes_graph
  - seen__inv_freq

Параметры (ctx["features_cfg"]["feat_interact"]):
  - T_recent: int (по умолчанию 3) — порог «недавно активен»
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


def _cfg(ctx: dict) -> dict:
    d = (ctx.get("features_cfg", {}) or {}).get("feat_interact", {}) or {}
    return {
        "T_recent": int(d.get("T_recent", 3)),
    }


SPEC = FeatureSpec(
    name="interact_strong",
    cols=[
        "lgcn_dot__rr_sum",
        "i2v_cos_max__rr_sum",
        "markov_max__overlap_lastK",
        "trend_14_decay__recent",
        "novelty__votes_graph",
        "seen__inv_freq",
    ],
    doc="Сильные интеракции: граф/похожесть×источники, тренд×реценси, новизна×графовые голоса, повторы×частота",
)


@register_feature(SPEC)
def build_interact_strong(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must contain user_id,item_id"
    n = len(cand_df)
    cfg = _cfg(ctx)
    T_recent = int(cfg["T_recent"])

    def grab(name: str, dtype: str, default_val=0):
        if name in cand_df.columns:
            s = cand_df[name]
            # для совместимости типов
            if dtype.startswith("float"):
                return s.astype("float32")
            if dtype.startswith("int"):
                return s.fillna(0).astype(dtype)
            return s
        # fallback — всё нули
        if dtype.startswith("float"):
            return pd.Series(np.zeros(n, dtype=np.float32), index=cand_df.index)
        return pd.Series(np.zeros(n, dtype=np.int64), index=cand_df.index).astype(dtype)

    # базовые сигналы (если нет — будут нули)
    rr_sum        = grab("rr_sum", "float32")
    votes_graph   = grab("votes_graph", "int16")
    lgcn_dot      = grab("lgcn_dot", "float32")
    i2v_cos_max   = grab("i2v_cos_max", "float32")
    markov_max    = grab("markov_max", "float32")
    overlap_lastK = grab("overlap_lastK", "int8")
    # если overlap нет, но есть pos_lastK — сконструируем
    if "overlap_lastK" not in cand_df.columns and "pos_lastK" in cand_df.columns:
        pos = cand_df["pos_lastK"].fillna(0).astype("int16")
        overlap_lastK = (pos > 0).astype("int8")

    trend_14_decay = grab("trend_14_decay", "float32")
    novelty        = grab("novelty", "float32")

    # days_since_last — если нет в cand_df, считаем из user_history
    if "days_since_last" in cand_df.columns:
        days_since_last = cand_df["days_since_last"].astype("int32")
    else:
        uh = res.ensure_user_history(lastK=10)
        u_last = uh.get("u_last_day", {})
        end_day = int(ctx["split"].train_end)
        days_since_last = cand_df[COL_USER].map(lambda u: int(end_day - u_last.get(int(u), end_day))).astype("int32")

    # seen_in_train_user / u_i_cnt — из cand_df или из user_history
    if "seen_in_train_user" in cand_df.columns:
        seen = cand_df["seen_in_train_user"].astype("int8")
    else:
        uh = res.ensure_user_history(lastK=10)
        u_i_cnt_map: Dict[Tuple[int, int], int] = uh.get("u_i_cnt", {})
        seen = pd.Series(
            [1 if (int(u), int(i)) in u_i_cnt_map else 0 for u, i in zip(cand_df[COL_USER].values, cand_df[COL_ITEM].values)],
            index=cand_df.index, dtype="int8"
        )

    if "u_i_cnt" in cand_df.columns:
        u_i_cnt_series = cand_df["u_i_cnt"].astype("int32")
    else:
        uh = res.ensure_user_history(lastK=10)
        u_i_cnt_map = uh.get("u_i_cnt", {})
        u_i_cnt_series = pd.Series(
            [int(u_i_cnt_map.get((int(u), int(i)), 0)) for u, i in zip(cand_df[COL_USER].values, cand_df[COL_ITEM].values)],
            index=cand_df.index, dtype="int32"
        )

    # --- интеракции ---
    out = cand_df[[COL_USER, COL_ITEM]].copy()

    out["lgcn_dot__rr_sum"] = (lgcn_dot.values * rr_sum.values).astype("float32")
    out["i2v_cos_max__rr_sum"] = (i2v_cos_max.values * rr_sum.values).astype("float32")
    out["markov_max__overlap_lastK"] = (markov_max.values * overlap_lastK.values.astype("float32")).astype("float32")

    recent_mask = (days_since_last.values <= T_recent).astype("float32")
    out["trend_14_decay__recent"] = (trend_14_decay.values * recent_mask).astype("float32")

    out["novelty__votes_graph"] = (novelty.values * votes_graph.values.astype("float32")).astype("float32")

    out["seen__inv_freq"] = (seen.values.astype("float32") * (1.0 / np.log1p(u_i_cnt_series.values.astype("float32")))).astype("float32")

    return out
