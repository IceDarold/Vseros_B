# -*- coding: utf-8 -*-
"""
feat_user.py — фичи истории пользователя (по TRAIN), без утечек.

Выдаёт на пару (user_id, item_id):
  - seen_in_train_user : {0,1}  — видел ли юзер этот айтем в train
  - u_i_cnt            : int32  — сколько раз видел (u,i) в train
  - days_since_last    : int32  — (train_end - последний день активности пользователя)
  - days_since_last_ui : int32  — (train_end - последний день пары u,i) (0, если не видел)
  - overlap_lastK      : {0,1}  — item встречается в последних K кликах
  - pos_lastK          : int16  — позиция от конца (1 — самый последний), 0 если нет в lastK
  - lastK_len          : int16  — фактическая длина хвоста пользователя (<=K)

А также user-level (подмешиваем по user_id):
  - u_clicks_all, u_clicks_28, u_clicks_14, u_clicks_7 : int32
  - u_days_active : int16
  - u_uniqs_all   : int32
  - repeat_ratio  : float32 = 1 - u_uniqs_all / u_clicks_all

Настройки: ctx["features_cfg"]["feat_user"] (необязательно)
  - lastK: int (по умолчанию 10)
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM, COL_DATE
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


def _cfg(ctx: dict) -> dict:
    d = (ctx.get("features_cfg", {}) or {}).get("feat_user", {}) or {}
    return {"lastK": int(d.get("lastK", 10))}


SPEC = FeatureSpec(
    name="user_basic",
    cols=[
        # pair-level
        "seen_in_train_user", "u_i_cnt", "days_since_last", "days_since_last_ui",
        "overlap_lastK", "pos_lastK", "lastK_len",
        # user-level
        "u_clicks_all", "u_clicks_28", "u_clicks_14", "u_clicks_7",
        "u_days_active", "u_uniqs_all", "repeat_ratio",
    ],
    doc="История пользователя: повторы, lastK, активность и репетитивность",
)


@register_feature(SPEC)
def build_user_basic(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must have user_id,item_id"

    cfg = _cfg(ctx)
    lastK = int(cfg["lastK"])

    train_df: pd.DataFrame = ctx["train_df"]
    end_day: int = int(ctx["split"].train_end)

    # ---- ресурсы истории ----
    uh = res.ensure_user_history(lastK=lastK)
    lastK_map: Dict[int, List[int]] = uh.get("lastK", {})
    u_last_day: Dict[int, int] = uh.get("u_last_day", {})
    u_clicks_7: Dict[int, int] = uh.get("u_clicks_7", {})
    u_clicks_14: Dict[int, int] = uh.get("u_clicks_14", {})
    u_clicks_28: Dict[int, int] = uh.get("u_clicks_28", {})
    u_i_cnt_map: Dict[Tuple[int, int], int] = uh.get("u_i_cnt", {})

    # ---- user-level агрегаты из train ----
    if train_df.empty:
        u_clicks_all = {}
        u_days_active = {}
        u_uniqs_all = {}
        u_i_last_day = {}
    else:
        u_clicks_all = train_df.groupby(COL_USER).size().astype(int).to_dict()
        u_days_active = train_df.groupby(COL_USER)[COL_DATE].nunique().astype(int).to_dict()
        u_uniqs_all = train_df.groupby(COL_USER)[COL_ITEM].nunique().astype(int).to_dict()
        u_i_last_day = train_df.groupby([COL_USER, COL_ITEM])[COL_DATE].max().astype(int).to_dict()

    # ---- подготовим пер-пользовательские словари для быстрого доступа ----
    # pos_lastK: для каждого u — карта item -> позиция (1 = последний), иначе 0
    u_pos_map: Dict[int, Dict[int, int]] = {}
    u_lastk_len: Dict[int, int] = {}
    for u, seq in lastK_map.items():
        if not seq:
            u_pos_map[int(u)] = {}
            u_lastk_len[int(u)] = 0
            continue
        pos_d: Dict[int, int] = {}
        L = len(seq)
        for idx, it in enumerate(seq[::-1], start=1):  # от конца
            # если айтем повторяется в хвосте — позиция минимальная (ближе к концу)
            if int(it) not in pos_d:
                pos_d[int(it)] = idx
        u_pos_map[int(u)] = pos_d
        u_lastk_len[int(u)] = L

    # ---- начнём собирать вывод ----
    out = cand_df[[COL_USER, COL_ITEM]].copy()

    # pair-level — векторизуем через map’ы
    u_vals = out[COL_USER].values
    i_vals = out[COL_ITEM].values
    n = len(out)

    # seen & u_i_cnt
    seen = np.fromiter(((int(u), int(i)) in u_i_cnt_map for u, i in zip(u_vals, i_vals)), count=n, dtype=np.int8)
    u_i_cnt = np.fromiter((int(u_i_cnt_map.get((int(u), int(i)), 0)) for u, i in zip(u_vals, i_vals)), count=n, dtype=np.int32)

    # days_since_last (user-level)
    dsl_user = np.fromiter((int(end_day - u_last_day.get(int(u), end_day)) for u in u_vals), count=n, dtype=np.int32)

    # days_since_last_ui (pair-level)
    dsl_ui = np.fromiter((int(end_day - u_i_last_day[(int(u), int(i))]) if (int(u), int(i)) in u_i_last_day else 0
                          for u, i in zip(u_vals, i_vals)), count=n, dtype=np.int32)

    # overlap_lastK / pos_lastK / lastK_len
    pos_last = np.fromiter((int(u_pos_map.get(int(u), {}).get(int(i), 0)) for u, i in zip(u_vals, i_vals)),
                           count=n, dtype=np.int16)
    overlap = (pos_last > 0).astype(np.int8)
    lastk_len = np.fromiter((int(u_lastk_len.get(int(u), 0)) for u in u_vals), count=n, dtype=np.int16)

    out["seen_in_train_user"] = seen.astype("int8")
    out["u_i_cnt"] = u_i_cnt.astype("int32")
    out["days_since_last"] = dsl_user.astype("int32")
    out["days_since_last_ui"] = dsl_ui.astype("int32")
    out["overlap_lastK"] = overlap.astype("int8")
    out["pos_lastK"] = pos_last.astype("int16")
    out["lastK_len"] = lastk_len.astype("int16")

    # user-level аггрегаты (join по user)
    users = pd.unique(out[COL_USER]).astype("int64")
    dfu = pd.DataFrame({COL_USER: users})
    dfu["u_clicks_all"] = dfu[COL_USER].map(lambda u: int(u_clicks_all.get(int(u), 0))).astype("int32")
    dfu["u_clicks_28"] = dfu[COL_USER].map(lambda u: int(u_clicks_28.get(int(u), 0))).astype("int32")
    dfu["u_clicks_14"] = dfu[COL_USER].map(lambda u: int(u_clicks_14.get(int(u), 0))).astype("int32")
    dfu["u_clicks_7"]  = dfu[COL_USER].map(lambda u: int(u_clicks_7.get(int(u), 0))).astype("int32")
    dfu["u_days_active"] = dfu[COL_USER].map(lambda u: int(u_days_active.get(int(u), 0))).astype("int16")
    dfu["u_uniqs_all"] = dfu[COL_USER].map(lambda u: int(u_uniqs_all.get(int(u), 0))).astype("int32")
    # repeat_ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = 1.0 - (dfu["u_uniqs_all"].astype("float32") / np.maximum(dfu["u_clicks_all"].astype("float32"), 1.0))
    dfu["repeat_ratio"] = rr.astype("float32")

    out = out.merge(dfu, on=COL_USER, how="left")

    # финальные типы и NaN-safe
    for c in ["u_clicks_all", "u_clicks_28", "u_clicks_14", "u_clicks_7", "u_uniqs_all"]:
        out[c] = out[c].fillna(0).astype("int32")
    out["u_days_active"] = out["u_days_active"].fillna(0).astype("int16")
    out["repeat_ratio"] = out["repeat_ratio"].fillna(0.0).astype("float32")

    return out
