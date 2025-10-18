# -*- coding: utf-8 -*-
"""
feat_calendar.py — простые календарные и «возрастные» фичи, независимые от item:
  - dow_sin, dow_cos           : «день недели» от train_end (псевдо-неделя, period=7)
  - age_user                    : train_end - первый день появления пользователя в train
  - days_since_last             : train_end - последний день активности пользователя в train
  - recency_bias                : 1 / (1 + days_since_last)
  - user_active_7/14/28         : число событий пользователя в последних окнах

Все вычисления — строго по train (без утечек).
Параметры можно переопределить через ctx["features_cfg"]["feat_calendar"] (необязательны).
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM, COL_DATE
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


SPEC = FeatureSpec(
    name="calendar_basic",
    cols=[
        "dow_sin", "dow_cos",
        "age_user", "days_since_last", "recency_bias",
        "user_active_7", "user_active_14", "user_active_28",
    ],
    doc="Календарные и «возрастные» фичи пользователя, зависящие от train_end",
)


@register_feature(SPEC)
def build_calendar_basic(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must contain user_id,item_id"

    train_df: pd.DataFrame = ctx["train_df"]
    end_day: int = int(ctx["split"].train_end)

    # Псевдо-день недели (0..6) + синус/косинус
    dow = int(end_day % 7)
    dow_sin = float(np.sin(2.0 * np.pi * dow / 7.0))
    dow_cos = float(np.cos(2.0 * np.pi * dow / 7.0))

    # История пользователя и активности из ресурсов
    uh = res.ensure_user_history(lastK=10)
    u_last_day: Dict[int, int] = uh.get("u_last_day", {})
    u_clicks_7: Dict[int, int] = uh.get("u_clicks_7", {})
    u_clicks_14: Dict[int, int] = uh.get("u_clicks_14", {})
    u_clicks_28: Dict[int, int] = uh.get("u_clicks_28", {})

    # Первый день пользователя (age_user)
    if train_df.empty:
        u_first_day = {}
    else:
        u_first_day = train_df.groupby(COL_USER)[COL_DATE].min().astype(int).to_dict()

    users = pd.unique(cand_df[COL_USER]).astype("int64")
    dfu = pd.DataFrame({COL_USER: users})

    dfu["last_day_user"] = dfu[COL_USER].map(lambda u: int(u_last_day.get(int(u), end_day)))
    dfu["first_day_user"] = dfu[COL_USER].map(lambda u: int(u_first_day.get(int(u), end_day)))
    dfu["age_user"] = (end_day - dfu["first_day_user"]).astype("int32")
    dfu["days_since_last"] = (end_day - dfu["last_day_user"]).astype("int32")
    dfu["recency_bias"] = (1.0 / (1.0 + dfu["days_since_last"].astype("float32"))).astype("float32")

    dfu["user_active_7"] = dfu[COL_USER].map(lambda u: int(u_clicks_7.get(int(u), 0))).astype("int32")
    dfu["user_active_14"] = dfu[COL_USER].map(lambda u: int(u_clicks_14.get(int(u), 0))).astype("int32")
    dfu["user_active_28"] = dfu[COL_USER].map(lambda u: int(u_clicks_28.get(int(u), 0))).astype("int32")

    # Константы на весь датасет
    dfu["dow_sin"] = np.float32(dow_sin)
    dfu["dow_cos"] = np.float32(dow_cos)

    # Мёрджим к кандидатам по user_id — item_id тут не влияет
    out = cand_df[[COL_USER, COL_ITEM]].merge(dfu, on=COL_USER, how="left")
    # Оставляем только заявленные фичи + ключи
    keep = [COL_USER, COL_ITEM] + SPEC.cols
    out = out[keep]

    # Типы
    out["age_user"] = out["age_user"].fillna(0).astype("int32")
    out["days_since_last"] = out["days_since_last"].fillna(0).astype("int32")
    for c in ["recency_bias", "dow_sin", "dow_cos"]:
        out[c] = out[c].fillna(0.0).astype("float32")
    for c in ["user_active_7", "user_active_14", "user_active_28"]:
        out[c] = out[c].fillna(0).astype("int32")

    return out
