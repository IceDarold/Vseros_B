# -*- coding: utf-8 -*-
"""
feat_regularizers.py — «регуляризаторы»/противовесы для ранкера:
  - anti_pop               : rr_sum / log1p(item_cnt_28)
  - anti_trend             : rr_trend / log1p(item_cnt_all)
  - penalize_old_repeat    : 1.0 если (seen_in_train_user == 1) и days_since_last_ui > T_old, иначе 0.0
  - fresh_boost            : trend_14_decay * 1{days_since_last <= T_recent}

Фича использует:
  - только TRAIN-данные (подсчёты по окнам и истории);
  - при наличии колонок из других фич (rr_sum, rr_trend, item_cnt_*) — возьмёт их,
    иначе рассчитает минимально необходимые величины сама (фолбек).

Параметры через ctx["features_cfg"]["feat_regularizers"]:
  - window_pop: 28
  - window_trend: 14
  - decay_lambda: 0.03
  - T_recent: 3
  - T_old: 14
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM, COL_DATE
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


def _cfg(ctx: dict) -> dict:
    d = (ctx.get("features_cfg", {}) or {}).get("feat_regularizers", {}) or {}
    return {
        "window_pop": int(d.get("window_pop", 28)),
        "window_trend": int(d.get("window_trend", 14)),
        "decay_lambda": float(d.get("decay_lambda", 0.03)),
        "T_recent": int(d.get("T_recent", 3)),
        "T_old": int(d.get("T_old", 14)),
    }


def _item_day_counts(train_df: pd.DataFrame) -> pd.DataFrame:
    """[day,item] → cnt."""
    if train_df.empty:
        return pd.DataFrame(columns=[COL_DATE, COL_ITEM, "cnt"])
    return (
        train_df.groupby([COL_DATE, COL_ITEM], as_index=False)
                .size()
                .rename(columns={"size": "cnt"})
                .astype({COL_DATE: "int32", COL_ITEM: "int64", "cnt": "int32"})
    )


SPEC = FeatureSpec(
    name="regularizers_basic",
    cols=["anti_pop", "anti_trend", "penalize_old_repeat", "fresh_boost"],
    doc="Регуляризаторы/противовесы для ранкера на основе поп-трендов и истории пользователя",
)


@register_feature(SPEC)
def build_regularizers_basic(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must contain user_id,item_id"

    cfg = _cfg(ctx)
    train_df: pd.DataFrame = ctx["train_df"]
    end_day: int = int(ctx["split"].train_end)
    start_day: int = int(train_df[COL_DATE].min()) if not train_df.empty else 0

    # ---- базовые подготовленные словари из ресурсов ----
    uh = res.ensure_user_history(lastK=10)  # содержит u_last_day, u_i_cnt
    u_last_day: Dict[int, int] = uh.get("u_last_day", {})
    u_i_cnt: Dict[Tuple[int, int], int] = uh.get("u_i_cnt", {})

    # ---- per-item частоты и тренд (если нет готовых колонок) ----
    di = _item_day_counts(train_df)

    # item_cnt_all
    item_cnt_all = di.groupby(COL_ITEM)["cnt"].sum().astype("int32") if not di.empty else pd.Series(dtype="int32")

    # item_cnt_window (window_pop)
    if not di.empty:
        left_pop = max(start_day, end_day - cfg["window_pop"] + 1)
        di_pop = di[(di[COL_DATE] >= left_pop) & (di[COL_DATE] <= end_day)]
        item_cnt_pop = di_pop.groupby(COL_ITEM)["cnt"].sum().astype("int32") if not di_pop.empty else pd.Series(dtype="int32")
    else:
        item_cnt_pop = pd.Series(dtype="int32")

    # trend_14_decay (window_trend with decay)
    if not di.empty:
        left_tr = max(start_day, end_day - cfg["window_trend"] + 1)
        di_tr = di[(di[COL_DATE] >= left_tr) & (di[COL_DATE] <= end_day)]
        if di_tr.empty:
            trend_decay = pd.Series(dtype="float32")
        else:
            age = (end_day - di_tr[COL_DATE]).astype("float32").clip(lower=0.0)
            w = np.exp(-float(cfg["decay_lambda"]) * age)
            trend_decay = (di_tr.assign(w=w * di_tr["cnt"])
                                .groupby(COL_ITEM)["w"].sum()
                                .astype("float32"))
    else:
        trend_decay = pd.Series(dtype="float32")

    # ---- подготовим пользовательские части ----
    # days_since_last(u)
    def dsl(u: int) -> int:
        last = u_last_day.get(int(u), end_day)
        return int(end_day - last)

    # seen_in_train_user(u,i) и days_since_last_ui
    if train_df.empty:
        u_i_last_day = {}
    else:
        u_i_last_day = (
            train_df.groupby([COL_USER, COL_ITEM])[COL_DATE].max().astype(int).to_dict()
        )

    # ---- подготовим каркас вывода ----
    out = cand_df[[COL_USER, COL_ITEM]].copy()

    # ---- rr сигнал из trending (если есть в cand_df после мёрджа feat_sources) ----
    rr_sum = out.get("rr_sum", None)
    if rr_sum is None:
        rr_sum = pd.Series(0.0, index=out.index, dtype="float32")
    else:
        rr_sum = out["rr_sum"].astype("float32")

    rr_trend = out.get("rr_trend", None)
    if rr_trend is None:
        # если нет rr_trend (от feat_sources/trending), считаем 0 — анти-тренд отключится
        rr_trend = pd.Series(0.0, index=out.index, dtype="float32")
    else:
        rr_trend = out["rr_trend"].astype("float32")

    # ---- джойним item-частоты/тренды ----
    # item_cnt_28
    if item_cnt_pop is not None and not item_cnt_pop.empty:
        out = out.merge(item_cnt_pop.rename("item_cnt_28").reset_index(), on=COL_ITEM, how="left")
    else:
        out["item_cnt_28"] = 0
    out["item_cnt_28"] = out["item_cnt_28"].fillna(0).astype("int32")

    # item_cnt_all
    if item_cnt_all is not None and not item_cnt_all.empty:
        out = out.merge(item_cnt_all.rename("item_cnt_all").reset_index(), on=COL_ITEM, how="left")
    else:
        out["item_cnt_all"] = 0
    out["item_cnt_all"] = out["item_cnt_all"].fillna(0).astype("int32")

    # trend_14_decay
    if trend_decay is not None and not trend_decay.empty:
        out = out.merge(trend_decay.rename("trend_14_decay").reset_index(), on=COL_ITEM, how="left")
    else:
        out["trend_14_decay"] = 0.0
    out["trend_14_decay"] = out["trend_14_decay"].fillna(0.0).astype("float32")

    # ---- anti_pop / anti_trend ----
    out["anti_pop"] = (rr_sum.values / np.log1p(out["item_cnt_28"].astype("float32").values)).astype("float32")
    out["anti_pop"] = out["anti_pop"].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    out["anti_trend"] = (rr_trend.values / np.log1p(out["item_cnt_all"].astype("float32").values)).astype("float32")
    out["anti_trend"] = out["anti_trend"].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    # ---- penalize_old_repeat ----
    # seen_in_train_user: по u_i_cnt (из ресурсов) или по наличию u_i_last_day
    def seen(u: int, i: int) -> bool:
        if (int(u), int(i)) in u_i_cnt:
            return True
        return (int(u), int(i)) in u_i_last_day

    # days_since_last_ui
    def dsl_ui(u: int, i: int) -> int:
        ld = u_i_last_day.get((int(u), int(i)))
        if ld is None:
            return 0
        return int(end_day - ld)

    T_old = int(cfg["T_old"])
    penal = np.zeros(len(out), dtype=np.float32)
    for idx, (u, i) in enumerate(zip(out[COL_USER].values, out[COL_ITEM].values)):
        if seen(int(u), int(i)):
            if dsl_ui(int(u), int(i)) > T_old:
                penal[idx] = 1.0
    out["penalize_old_repeat"] = penal.astype("float32")

    # ---- fresh_boost ----
    T_recent = int(cfg["T_recent"])
    dsl_u = np.array([dsl(int(u)) for u in out[COL_USER].values], dtype=np.int32)
    fresh_mask = (dsl_u <= T_recent).astype("float32")
    out["fresh_boost"] = (out["trend_14_decay"].astype("float32").values * fresh_mask).astype("float32")

    # Финальные колонки
    keep = [COL_USER, COL_ITEM] + SPEC.cols
    out = out[keep]

    return out
