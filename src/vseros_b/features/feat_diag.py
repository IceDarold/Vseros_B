# -*- coding: utf-8 -*-
"""
feat_diag.py — диагностические/служебные признаки, полезные для отладки и иногда для ранкера:
  - diag_num_sources  : число источников, из которых пришёл кандидат (если in_* присутствуют)
  - diag_any_graph    : 1, если присутствует любой графовый источник (lgcn/ppr/covis1/covis2)
  - diag_any_simple   : 1, если присутствует хотя бы один «простой» источник (pop/trend/i2v)
  - diag_rank_min_any : минимальный ранг среди источников (если есть rank_*), иначе большой sentry
  - diag_seen_in_user_hist : 1, если (u,i) встречались в TRAIN по пользователю (из cand_df или user_history)

Фича не требует строгого порядка — если базовых колонок нет, заполняет нулями/дефолтами.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


GRAPH_IN_COLS = ["in_lgcn", "in_ppr", "in_covis1", "in_covis2"]
SIMPLE_IN_COLS = ["in_pop", "in_trend", "in_i2v"]
RANK_COLS = ["rank_pop", "rank_trend", "rank_covis1", "rank_covis2", "rank_i2v", "rank_lgcn", "rank_ppr"]
IN_COLS = ["in_pop", "in_trend", "in_covis1", "in_covis2", "in_i2v", "in_lgcn", "in_ppr"]


SPEC = FeatureSpec(
    name="diag_basic",
    cols=[
        "diag_num_sources",
        "diag_any_graph",
        "diag_any_simple",
        "diag_rank_min_any",
        "diag_seen_in_user_hist",
    ],
    doc="Диагностические фичи: количество источников, граф/не-граф флаги, min-ранг, признак повторов",
)


@register_feature(SPEC)
def build_diag_basic(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must contain user_id,item_id"
    n = len(cand_df)
    out = cand_df[[COL_USER, COL_ITEM]].copy()

    # --- num_sources ---
    if set(IN_COLS).issubset(set(cand_df.columns)):
        out["diag_num_sources"] = cand_df[IN_COLS].sum(axis=1).astype("int16")
    else:
        # суммируем только существующие in_*
        present = [c for c in IN_COLS if c in cand_df.columns]
        if present:
            out["diag_num_sources"] = cand_df[present].sum(axis=1).astype("int16")
        else:
            out["diag_num_sources"] = np.zeros(n, dtype=np.int16)

    # --- any_graph / any_simple ---
    g_present = [c for c in GRAPH_IN_COLS if c in cand_df.columns]
    s_present = [c for c in SIMPLE_IN_COLS if c in cand_df.columns]
    if g_present:
        out["diag_any_graph"] = (cand_df[g_present].sum(axis=1) > 0).astype("int8")
    else:
        out["diag_any_graph"] = np.zeros(n, dtype=np.int8)
    if s_present:
        out["diag_any_simple"] = (cand_df[s_present].sum(axis=1) > 0).astype("int8")
    else:
        out["diag_any_simple"] = np.zeros(n, dtype=np.int8)

    # --- min rank among sources ---
    r_present = [c for c in RANK_COLS if c in cand_df.columns]
    SENT = np.int32(10_000_000)
    if r_present:
        out["diag_rank_min_any"] = cand_df[r_present].replace(0, np.nan).min(axis=1, skipna=True).fillna(SENT).astype("int32")
    else:
        out["diag_rank_min_any"] = np.full(n, fill_value=SENT, dtype=np.int32)

    # --- seen in user history ---
    if "seen_in_train_user" in cand_df.columns:
        out["diag_seen_in_user_hist"] = cand_df["seen_in_train_user"].astype("int8")
    else:
        uh = res.ensure_user_history(lastK=10)
        u_i_cnt: Dict[Tuple[int, int], int] = uh.get("u_i_cnt", {})
        out["diag_seen_in_user_hist"] = np.array(
            [1 if (int(u), int(i)) in u_i_cnt else 0 for u, i in zip(cand_df[COL_USER].values, cand_df[COL_ITEM].values)],
            dtype=np.int8
        )

    return out
