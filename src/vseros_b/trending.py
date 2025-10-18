# -*- coding: utf-8 -*-
"""
Trending helpers for day-based validation/test without leakage.

Колонки датасета:
  - user_id, item_id, day  (day — целое число, 0..N)

Что здесь есть:
  • build_val_day_toplists(train_df, val_start, val_end, ...)
      Для каждого дня d в [val_start, val_end] строит топ-список айтемов
      из скользящего окна прошлых дней [d-window_days, d-1] без утечки.

  • candidates_from_day_toplists(toplists_by_day, user_day_map, k_top)
      Превращает day→top_items в user→top_items, используя отображение user→day.

  • evaluate_trending_candidates(cand_map, truth, k_list)
      Считает HR@k / NDCG@k для мапы кандидатов (user→список item_id).

  • coverage_from_frozen_trending(train_df, truth, users, window_days, k_top)
      Быстрая проверка «замороженного» трендинга (один общий список для всех),
      возвращает метрики и coverage.

  • trending_global_from_last_train_window(train_df, last_window_days, topk)
      Глобальный топ айтемов из последних W дней train.

Примечания:
  - Никаких внешних зависимостей кроме pandas/numpy.
  - Безопасно к пустым окнам: если в окне нет событий, вернётся пустой список.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional, Tuple

import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM, COL_DATE
from vseros_b.metrics import recall_at_m, ndcg_at_k


# ---------------------------------------------------------------------------
# Вспомогалки
# ---------------------------------------------------------------------------

def _top_items_in_window(df: pd.DataFrame,
                         lo_day: int,
                         hi_day: int,
                         min_count: int = 1,
                         topk: Optional[int] = None) -> List[int]:
    """
    Берёт срез по дням [lo_day, hi_day] включительно и возвращает
    список item_id по убыванию частоты.
    """
    if lo_day > hi_day:
        return []
    mask = (df[COL_DATE] >= int(lo_day)) & (df[COL_DATE] <= int(hi_day))
    if not mask.any():
        return []
    cnt = (df.loc[mask, [COL_ITEM, COL_USER]]
             .groupby(COL_ITEM, sort=False)[COL_USER]
             .count()
             .astype("int64"))
    if len(cnt) == 0:
        return []
    if min_count > 1:
        cnt = cnt[cnt >= int(min_count)]
        if len(cnt) == 0:
            return []
    order = cnt.sort_values(ascending=False).index.values.tolist()
    if topk is not None and len(order) > int(topk):
        order = order[:int(topk)]
    return list(map(int, order))


def _ensure_int_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_USER] = out[COL_USER].astype("int64")
    out[COL_ITEM] = out[COL_ITEM].astype("int64")
    out[COL_DATE] = out[COL_DATE].astype("int32")
    return out


# ---------------------------------------------------------------------------
# 1) Day→TopList (валидационное построение без утечки)
# ---------------------------------------------------------------------------

from typing import Dict, List, Optional, Any
from types import SimpleNamespace
import numpy as np
import pandas as pd

def build_val_day_toplists(*args,
                           df_all: Optional[pd.DataFrame] = None,
                           split: Optional[Any] = None,
                           window_days: int = 3,
                           topk_per_day: int = 1000,
                           min_count: int = 1,
                           mode: str = "frozen_train",
                           date_col: str = "date",
                           **kwargs) -> Dict[int, List[int]]:
    """
    Построить топ-списки айтемов для каждого дня d ∈ [val_start, val_end],
    считая частоты по окну [d - window_days, d - 1] (день d не входит).

    Поддерживаются два способа вызова:

    (A) новый (рекомендуемый):
        build_val_day_toplists(
            df_all=DF, split=Split, window_days=3, topk_per_day=1000,
            min_count=1, mode='frozen_train', date_col='date'
        )
        где split — объект с атрибутами/ключами: train_end, val_start, val_end.

    (B) старый (backward-compat):
        build_val_day_toplists(train_df, val_start, val_end, window_days=3, topk_per_day=1000, min_count=1)

    Режимы:
      - mode='frozen_train': окно берётся ТОЛЬКО из train-отрезка (<= train_end).
      - mode='moving': окно берётся из df_all, но всегда <= (d-1), без утечки в день d.
    """
    # -------- helpers --------
    def _get_attr(s, key):
        if s is None: return None
        if isinstance(s, dict): return s.get(key, None)
        return getattr(s, key, None)

    def _ensure_date_alias(df: pd.DataFrame) -> pd.DataFrame:
        # Истинная колонка — date. Если пережиток 'day' есть — используем его как источник date.
        if date_col not in df.columns:
            if "day" in df.columns:
                df = df.copy()
                df[date_col] = df["day"].astype(df["day"].dtype)
            else:
                raise KeyError(f"Дата-колонка '{date_col}' не найдена (и нет 'day').")
        # типы
        if not np.issubdtype(df[date_col].dtype, np.integer):
            df = df.copy()
            df[date_col] = df[date_col].astype("int32")
        if "item_id" in df.columns and not np.issubdtype(df["item_id"].dtype, np.integer):
            df["item_id"] = df["item_id"].astype("int32")
        return df

    def _top_items_in_window_df(df: pd.DataFrame, lo: int, hi: int) -> List[int]:
        if lo > hi:
            return []
        sub = df.loc[(df[date_col] >= int(lo)) & (df[date_col] <= int(hi)), "item_id"]
        if sub.empty:
            return []
        vc = sub.value_counts()
        if min_count > 1:
            vc = vc[vc >= int(min_count)]
            if vc.empty:
                return []
        return vc.index.astype("int64").tolist()[: int(topk_per_day)]

    # -------- parse signatures --------
    if df_all is None and split is None and len(args) >= 3:
        # старый стиль: (train_df, val_start, val_end, ...)
        train_df = args[0]
        val_start = int(args[1])
        val_end   = int(args[2])
        df_all = None
        # работаем по train_df только (как раньше)
        df_train = _ensure_date_alias(train_df)
        toplists: Dict[int, List[int]] = {}
        for d in range(val_start, val_end + 1):
            lo, hi = d - int(window_days), d - 1
            toplists[int(d)] = _top_items_in_window_df(df_train, lo, hi)
        return toplists

    # новый стиль: ожидаем df_all и split (Namespace или dict)
    if df_all is None or split is None:
        raise TypeError("build_val_day_toplists: нужно передать df_all= и split= в новом стиле, "
                        "либо (train_df, val_start, val_end) в старом стиле.")

    df_all = _ensure_date_alias(df_all)

    val_start = _get_attr(split, "val_start")
    val_end   = _get_attr(split, "val_end")
    train_end = _get_attr(split, "train_end")
    if val_start is None or val_end is None:
        raise ValueError("split должен содержать val_start и val_end")
    if mode == "frozen_train" and train_end is None:
        raise ValueError("split.train_end обязателен для mode='frozen_train'")

    # подвыборка для подсчётов
    if mode == "frozen_train":
        # считаем частоты ТОЛЬКО по train-отрезку (<= train_end)
        df_base = df_all.loc[df_all[date_col] <= int(train_end), ["item_id", date_col]]
    elif mode == "moving":
        # будем резать по (<= d-1) внутри цикла
        df_base = df_all[["item_id", date_col]]
    else:
        raise ValueError("mode должен быть 'frozen_train' или 'moving'")

    toplists: Dict[int, List[int]] = {}
    for d in range(int(val_start), int(val_end) + 1):
        lo, hi = d - int(window_days), d - 1
        if mode == "frozen_train":
            toplists[int(d)] = _top_items_in_window_df(df_base, lo, hi)
        else:  # moving
            # ограничиваем верх границы окном d-1
            df_m = df_base.loc[df_base[date_col] <= hi]
            toplists[int(d)] = _top_items_in_window_df(df_m, lo, hi)

    return toplists


# ---------------------------------------------------------------------------
# 3) Оценка кандидатов трендинга
# ---------------------------------------------------------------------------

def evaluate_trending_candidates(cand_map: Dict[int, List[int]],
                                 truth: Dict[int, set],
                                 k_list: Sequence[int] = (20, 50, 100)) -> pd.DataFrame:
    """
    cand_map: user -> ranked list of items
    truth:    user -> set of true items (в валидационном периоде)
    """
    # фильтруем до тех u, у кого есть кандидаты и правда
    users = sorted(set(cand_map.keys()) & set(truth.keys()))
    sub = {u: cand_map[u] for u in users}
    gt  = {u: truth[u] for u in users}

    rows = []
    for k in k_list:
        rows.append({
            "k": int(k),
            "HR@k": float(recall_at_m(sub, gt, m=int(k))),
            "NDCG@k": float(ndcg_at_k(sub, gt, k=int(k))),
            "users_eval": int(len(users)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4) Frozen-trending coverage (один общий топ для всех)
# ---------------------------------------------------------------------------

def coverage_from_frozen_trending(train_df: pd.DataFrame,
                                  truth: Dict[int, set],
                                  users: Optional[Sequence[int]] = None,
                                  last_window_days: int = 3,
                                  k_top: int = 20,
                                  min_count: int = 1) -> Tuple[Dict[int, List[int]], pd.DataFrame]:
    """
    Строит один общий топ из последних W дней train и выдаёт его всем пользователям.
    Возвращает (cand_map, metrics_df).
    """
    df = _ensure_int_cols(train_df)
    max_day = int(df[COL_DATE].max())
    lo = max_day - int(last_window_days) + 1
    global_top = _top_items_in_window(df, lo, max_day, min_count=min_count, topk=int(k_top))

    if users is None:
        users = sorted(truth.keys())

    cand_map = {int(u): list(global_top) for u in users}
    metrics = evaluate_trending_candidates(cand_map, truth, k_list=(k_top,))  # логично посчитать хотя бы @k_top
    # добавим простые coverage-показатели
    item_coverage = len(set().union(*[set(v) for v in cand_map.values()])) if cand_map else 0
    user_coverage = sum(1 for u in users if len(cand_map.get(int(u), [])) > 0)
    metrics["item_coverage"] = int(item_coverage)
    metrics["user_coverage"] = int(user_coverage)
    metrics["global_top_size"] = int(len(global_top))
    return cand_map, metrics


# ---------------------------------------------------------------------------
# 5) Глобальный трендинг из последнего окна train
# ---------------------------------------------------------------------------

def trending_global_from_last_train_window(train_df: pd.DataFrame,
                                           last_window_days: int = 3,
                                           topk: int = 1000,
                                           min_count: int = 1) -> List[int]:
    """
    Возвращает глобальный список трендинга по последним W дням train.
    """
    df = _ensure_int_cols(train_df)
    max_day = int(df[COL_DATE].max())
    lo = max_day - int(last_window_days) + 1
    return _top_items_in_window(df, lo, max_day, min_count=min_count, topk=int(topk))


# ---------------------------------------------------------------------------
# (опционально) Вспомогалка: построить user→day map из вал.таблицы
# ---------------------------------------------------------------------------

def build_user_day_map(val_df: pd.DataFrame, policy: str = "min") -> Dict[int, int]:
    """
    Полезно для exp102: по валидационной выборке собрать отображение user→day.

    policy:
      - "min": берём минимальный day пользователя в вал.окне (первое появление);
      - "max": берём максимальный day (последнее появление).
    """
    df = _ensure_int_cols(val_df)
    if policy == "max":
        agg = df.groupby(COL_USER, sort=False)[COL_DATE].max().astype("int32")
    else:
        agg = df.groupby(COL_USER, sort=False)[COL_DATE].min().astype("int32")
    return {int(u): int(d) for u, d in agg.items()}
