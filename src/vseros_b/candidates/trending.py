# -*- coding: utf-8 -*-
"""
trending.py — дневные тренды без утечек + утилиты для кандидатов и оценки.

Что здесь:
- build_val_day_toplists: пер-дневные топы на вал-диапазон (считаются ТОЛЬКО по train-окну)
- candidates_from_day_toplists: преобразование пер-дневных топов в per-user кандидатов
- evaluate_trending_candidates: оценка recall@M для таких кандидатов
- coverage_from_frozen_trending: покрытие пользователей и длины листов (frozen-тренды)
- trending_global_from_last_train_window: глобальный тренд по последнему train-окну
- user_first_val_day, day_item_counts — служебные

Зависимости проекта:
- .config: COL_USER, COL_ITEM, COL_DATE, CAND_TOP_M_PER_USER
- .metrics: recall_at_m
- (опционально) .pop_decay: compute_pop_static, build_global_top — если нужен бэкфилл
"""

from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import COL_USER, COL_ITEM, COL_DATE, CAND_TOP_M_PER_USER
from ..metrics import recall_at_m


# ============================ базовые агрегаты ============================

def day_item_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подсчёт частот по (день, item_id).
    Возвращает DataFrame: [day, item_id, cnt]
    """
    if not {COL_DATE, COL_ITEM}.issubset(df.columns):
        raise ValueError(f"df must have [{COL_DATE}, {COL_ITEM}]")
    out = (
        df.groupby([COL_DATE, COL_ITEM], as_index=False)
          .size()
          .rename(columns={"size": "cnt"})
    )
    return out[[COL_DATE, COL_ITEM, "cnt"]]


# ============================ пер-дневные топы (без утечек) ============================

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
      - train_df: DataFrame с колонками [user_id, item_id, day] и ТОЛЬКО train-днями
      - split: объект со свойствами train_end, val_start, val_end
      - window_days: ширина скользящего окна в днях
      - topk_per_day: длина итогового топ-листа на день
      - min_item_freq: фильтр минимальной частоты/веса
      - decay_lambda: если >0, применяет экспоненциальный decay по "возрасту" дня:
            w = exp(-λ * (right - day)), где right = d-1

    Возвращает:
      { day(int) -> [item_id, ...] } длиной до topk_per_day
    """
    if train_df.empty:
        return {}

    di = day_item_counts(train_df)  # [day, item_id, cnt]
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

        win = di[(di[COL_DATE] >= left) & (di[COL_DATE] <= right)]
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
    if not {COL_USER, COL_DATE}.issubset(val_df.columns):
        raise ValueError(f"val_df must have [{COL_USER}, {COL_DATE}]")
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

    Возвращает:
      {user_id -> [item_id1, ...]}
    """
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


# ============================ оценка и покрытия ============================

def evaluate_trending_candidates(
    val_truth: Mapping[int, set],
    cand_map: Mapping[int, Sequence[int]],
    M_list: Sequence[int] = (50, 100, 200, 500, 1000),
    averaging: str = "micro",
) -> pd.DataFrame:
    """
    Оценивает recall@M для готовых trending-кандидатов.
    Ожидается, что и val_truth, и cand_map — в item_id (сырых).
    """
    rows = []
    for m in M_list:
        r = recall_at_m(val_truth, cand_map, m=int(m), averaging=averaging)
        rows.append({"M": int(m), "recall": float(r)})
    return pd.DataFrame(rows)


def coverage_from_frozen_trending(
    val_df: pd.DataFrame,
    toplists: Mapping[int, Sequence[int]],
    cand_map: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[dict, pd.DataFrame]:
    """
    Считает покрытие по пользователям и длины списков для frozen-трендов.
    - Общая сводка: % пользователей с ненулевым листом, средняя/медианная длины списков.
    - Пер-дневная сводка: сколько пользователей "падает" на день d, и какой длины там топ-лист.

    Параметры:
      - val_df: валидация
      - toplists: {day -> [item_id, ...]} — пер-дневные топы
      - cand_map: {user_id -> [item_id, ...]} — уже собранные кандидаты (необязательно)
                  если не задан — будет собран по toplists с M = длина списка дня (без бэкфилла)

    Возвращает:
      (summary_dict, per_day_df)
    """
    if not {COL_USER, COL_DATE}.issubset(val_df.columns):
        raise ValueError(f"val_df must have [{COL_USER}, {COL_DATE}]")

    # кому какой день
    u2day = user_first_val_day(val_df)
    per_day = (
        pd.DataFrame({COL_USER: list(u2day.keys()), COL_DATE: list(u2day.values())})
          .groupby(COL_DATE, as_index=False)
          .size()
          .rename(columns={"size": "users_on_day"})
    )
    # длина топ-листа в каждый день
    day_len = pd.DataFrame({
        COL_DATE: list(toplists.keys()),
        "toplist_len": [len(v) for v in toplists.values()]
    })
    per_day = per_day.merge(day_len, on=COL_DATE, how="left").fillna({"toplist_len": 0})

    # если cand_map не дан — соберём "как есть" (без бэкфилла)
    if cand_map is None:
        tmp_cand: Dict[int, List[int]] = {}
        for u, d in u2day.items():
            items = toplists.get(int(d), [])
            tmp_cand[int(u)] = list(map(int, items))
        cand_map = tmp_cand

    # покрытие и длины
    lens = [len(cand_map.get(int(u), [])) for u in u2day.keys()]
    users_total = len(lens)
    covered = sum(1 for L in lens if L > 0)
    summary = {
        "users_total": users_total,
        "users_with_list": covered,
        "coverage_pct": 100.0 * covered / max(1, users_total),
        "avg_list_len": float(np.mean(lens)) if lens else 0.0,
        "p50_list_len": float(np.percentile(lens, 50)) if lens else 0.0,
        "p90_list_len": float(np.percentile(lens, 90)) if lens else 0.0,
    }

    return summary, per_day.sort_values(COL_DATE).reset_index(drop=True)


# ============================ глобальный тренд по последнему train-окну ============================

def trending_global_from_last_train_window(
    train_df: pd.DataFrame,
    split,
    window_days: int = 14,
    topk: int = 1000,
    min_item_freq: int = 1,
    decay_lambda: float = 0.0,
) -> List[int]:
    """
    Глобальный тренд без утечек: считаем популярность по последним `window_days`
    TRAIN-дням (интервал [train_end - window_days + 1, train_end]) и отдаём топ-список.

    Можно потом использовать как backfill для дневных трендов/кандидатов.
    """
    if train_df.empty:
        return []

    end_day = int(split.train_end)
    start_day = max(int(train_df[COL_DATE].min()), end_day - int(window_days) + 1)

    x = train_df[(train_df[COL_DATE] >= start_day) & (train_df[COL_DATE] <= end_day)]
    if x.empty:
        return []

    di = day_item_counts(x)

    if decay_lambda and decay_lambda > 0.0:
        # чем свежее день, тем больше вес
        age = (end_day - di[COL_DATE]).astype(float).clip(lower=0)
        w = np.exp(-float(decay_lambda) * age)
        tmp = (
            di.assign(w=w * di["cnt"])
              .groupby(COL_ITEM, as_index=False)["w"].sum()
              .rename(columns={"w": "score"})
        )
    else:
        tmp = (
            di.groupby(COL_ITEM, as_index=False)["cnt"].sum()
              .rename(columns={"cnt": "score"})
        )

    tmp = tmp[tmp["score"] >= float(min_item_freq)]
    if tmp.empty:
        return []

    tmp = tmp.sort_values("score", ascending=False)
    return tmp[COL_ITEM].astype("int64").head(int(topk)).tolist()


# ============================ «всё-в-одном» (быстрый конвейер) ============================

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
        backfill = trending_global_from_last_train_window(
            train_df=train_df,
            split=split,
            window_days=window_days,
            topk=max(backfill_K, M),
            min_item_freq=min_item_freq,
            decay_lambda=decay_lambda,
        )

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
    "evaluate_trending_candidates",
    "coverage_from_frozen_trending",
    "trending_global_from_last_train_window",
    "build_trending_candidates",
]
