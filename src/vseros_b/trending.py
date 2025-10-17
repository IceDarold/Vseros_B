# -*- coding: utf-8 -*-
"""
trending.py — дневные тренды без утечек (per-day toplists) + утилиты.

Идея:
- Считаем популярность айтемов по дням на TRAIN-окне.
- Для каждого валидационного дня d берём скользящее окно [d - W, d - 1] и строим топ-лист,
  опционально с экспоненциальным затуханием по "возрасту" дня.
- Эти топ-листы можно напрямую использовать для рекомендаций на день d.

Основные функции:
- build_val_day_toplists(train_df, split, window_days=14, topk_per_day=1000, min_item_freq=1, decay_lambda=0.0)
- user_first_val_day(val_df) → dict[user_id -> day]
- candidates_from_day_toplists(val_df, toplists, M=1000, backfill=None)

Зависимости:
- pandas, numpy
- .config: COL_USER, COL_ITEM, COL_DATE, CAND_TOP_M_PER_USER
- .pop_decay: compute_pop_static, build_global_top (для бэкфилла при необходимости)
"""

from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import COL_USER, COL_ITEM, COL_DATE, CAND_TOP_M_PER_USER
from .pop_decay import compute_pop_static, build_global_top


# ============================ базовые агрегаты ============================

def day_item_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подсчёт частот по (день, item_id).
    Возвращает DataFrame: [day, item_id, cnt]
    """
    out = (
        df.groupby([COL_DATE, COL_ITEM], as_index=False)
          .size()
          .rename(columns={"size": "cnt"})
    )
    return out[[COL_DATE, COL_ITEM, "cnt"]]


# ============================ per-day toplists (без утечек) ============================

def build_val_day_toplists(
    train_df: pd.DataFrame,
    split,
    window_days: int = 14,
    topk_per_day: int = 1000,
    min_item_freq: int = 1,
    decay_lambda: float = 0.0,
) -> Dict[int, List[int]]:
    """
    Для каждого дня валидации d строит топ items на основе популярности в скользящем окне
    [d - window_days, d - 1], ИСКЛЮЧИТЕЛЬНО по TRAIN-окну (без утечек).

    Параметры:
      - train_df: DataFrame с колонками [user_id, item_id, day] и только TRAIN-данными
      - split: объект со свойствами train_end, val_start, val_end
      - window_days: ширина скользящего окна в днях
      - topk_per_day: длина итогового топ-листа на день
      - min_item_freq: фильтр минимальной частоты/веса в окне
      - decay_lambda: если >0, применяет экспоненциальный decay по "возрасту" дня:
            w = exp(-λ * (right - day)), где right = d-1

    Возвращает:
      словарь { day(int) -> [item_id, ...] } длиной до topk_per_day
    """
    assert {COL_ITEM, COL_DATE}.issubset(train_df.columns), "train_df must contain [item_id, day]"

    # дефолтно считаем, что train_df уже ограничен TRAIN-днями
    day_item = day_item_counts(train_df)  # [day, item_id, cnt]

    val_days = list(range(int(split.val_start), int(split.val_end) + 1))
    dmin_train = int(train_df[COL_DATE].min())
    dmax_train = int(split.train_end)

    out: Dict[int, List[int]] = {}
    for d in val_days:
        left = max(d - int(window_days), dmin_train)
        right = min(d - 1, dmax_train)
        if right < left:
            out[d] = []
            continue

        win = day_item[(day_item[COL_DATE] >= left) & (day_item[COL_DATE] <= right)]
        if win.empty:
            out[d] = []
            continue

        if decay_lambda and decay_lambda > 0.0:
            age = (right - win[COL_DATE]).astype(float).clip(lower=0)
            w = np.exp(-float(decay_lambda) * age)
            tmp = (
                win.assign(w=w * win["cnt"])
                   .groupby(COL_ITEM, as_index=False)["w"].sum()
                   .rename(columns={"w": "score"})
            )
        else:
            tmp = (
                win.groupby(COL_ITEM, as_index=False)["cnt"].sum()
                   .rename(columns={"cnt": "score"})
            )

        # фильтр редких/слабых
        tmp = tmp[tmp["score"] >= float(min_item_freq)]
        if tmp.empty:
            out[d] = []
            continue

        tmp = tmp.sort_values("score", ascending=False)
        items = tmp[COL_ITEM].astype("int64").head(int(topk_per_day)).tolist()
        out[d] = items

    return out


# ============================ утилиты для user→day и кандидатов ============================

def user_first_val_day(val_df: pd.DataFrame) -> Dict[int, int]:
    """
    Возвращает карту {user_id -> первый день появления в валидации}.
    Если у пользователя несколько вал-дней, берём минимальный (ранний).
    """
    x = (
        val_df.groupby(COL_USER, as_index=False)[COL_DATE]
              .min()
              .rename(columns={COL_DATE: "first_val_day"})
    )
    return {int(r[COL_USER]): int(r["first_val_day"]) for _, r in x.iterrows()}


def candidates_from_day_toplists(
    val_df: pd.DataFrame,
    toplists: Mapping[int, Sequence[int]],
    M: int = CAND_TOP_M_PER_USER,
    backfill: Optional[Sequence[int]] = None,
) -> Dict[int, List[int]]:
    """
    Собирает кандидатов per-user на основе пер-дневных топ-листов.
    Для каждого пользователя выбирается его первый вал-день, и выдаются top-M айтемов
    из соответствующего дневного топа. Если для дня список пуст — подставляется backfill.

    Параметры:
      - val_df: DataFrame валидации (используем для user_first_val_day)
      - toplists: {day -> [item_id, ...]}
      - M: длина списка кандидатов per-user
      - backfill: fallback-топ по популярности (item_id), если пер-дневный список пуст
    Возвращает:
      {user_id -> [item_id1, ...]}
    """
    assert COL_USER in val_df.columns and COL_DATE in val_df.columns, "val_df must have [user_id, day]"

    u2day = user_first_val_day(val_df)
    out: Dict[int, List[int]] = {}
    M = int(M)

    for u, d in u2day.items():
        items = toplists.get(int(d), [])
        if items:
            out[int(u)] = list(map(int, items[:M]))
        else:
            out[int(u)] = list(map(int, (backfill or [])[:M]))

    return out


# ============================ «всё-в-одном» (на случай быстрой сборки) ============================

def build_trending_candidates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    split,
    window_days: int = 14,
    M: int = CAND_TOP_M_PER_USER,
    min_item_freq: int = 1,
    decay_lambda: float = 0.0,
    backfill_with_global_top: bool = True,
    backfill_K: int = 3000,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Быстрый конвейер:
      1) строит per-day toplists для вал-окна (по TRAIN),
      2) собирает per-user кандидатов из топ-листа соответствующего вал-дня,
      3) (опц.) бэкфиллит пустых пользователей глобальным топом.

    Возврат:
      (candidates_map, toplists_map)
    """
    toplists = build_val_day_toplists(
        train_df=train_df,
        split=split,
        window_days=window_days,
        topk_per_day=max(M, 1000),
        min_item_freq=min_item_freq,
        decay_lambda=decay_lambda,
    )

    backfill = None
    if backfill_with_global_top:
        pop = compute_pop_static(train_df)
        backfill = build_global_top(pop, K=max(backfill_K, M))

    cand = candidates_from_day_toplists(
        val_df=val_df,
        toplists=toplists,
        M=M,
        backfill=backfill,
    )
    return cand, toplists


__all__ = [
    "day_item_counts",
    "build_val_day_toplists",
    "user_first_val_day",
    "candidates_from_day_toplists",
    "build_trending_candidates",
]
