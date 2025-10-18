# -*- coding: utf-8 -*-
"""
trending.py — утилиты для "трендинга"/глобальных топов без утечек.

Основные функции:
- build_val_day_toplists: построение топ-айтемов по дням валидации (без утечек)
- candidates_from_day_toplists: генерация кандидатов (user_id, item_id[, rank, score]) из day-топов
- evaluate_trending_candidates: простая оффлайн-оценка (HR@k, NDCG@k, Recall@k)
- coverage_from_frozen_trending: покрытие айтемов из day-топов
- trending_global_from_last_train_window: один глобальный список трендов по последнему train-окну

Колонка времени — `date`. Если в данных старая колонка `day`, будет создан алиас.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Any, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_date_alias(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Истинная колонка — `date`. Если присутствует только `day`, создадим алиас.
    Гарантируем целочисленные типы для date/item_id.
    """
    if date_col not in df.columns:
        if "day" in df.columns:
            df = df.copy()
            df[date_col] = df["day"].astype(df["day"].dtype)
        else:
            raise KeyError(f"Дата-колонка '{date_col}' не найдена (и нет 'day').")
    if "item_id" in df.columns and not np.issubdtype(df["item_id"].dtype, np.integer):
        df = df.copy()
        df["item_id"] = df["item_id"].astype("int64")
    if not np.issubdtype(df[date_col].dtype, np.integer):
        df = df.copy()
        df[date_col] = df[date_col].astype("int64")
    return df


def _top_items_in_window(df: pd.DataFrame,
                         lo: int,
                         hi: int,
                         date_col: str = "date",
                         topk: int = 1000,
                         min_count: int = 1) -> List[int]:
    """Топ айтемов по value_counts в окне [lo, hi]. Пустые/узкие окна → пустой список."""
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
    return vc.index.astype("int64").tolist()[: int(topk)]


def _get_attr(s: Any, key: str):
    """Поддерживаем split как dict или SimpleNamespace."""
    if s is None:
        return None
    if isinstance(s, dict):
        return s.get(key, None)
    return getattr(s, key, None)


# ---------------------------------------------------------------------
# 1) Day toplists for validation
# ---------------------------------------------------------------------

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
    считая частоты в окне [d - window_days, d - 1]. День d НЕ входит (без утечки).

    Поддерживаются два способа вызова:

    NEW (рекомендуемый):
        build_val_day_toplists(
            df_all=DF, split=SPLIT, window_days=3, topk_per_day=1000,
            min_count=1, mode='frozen_train'|'moving', date_col='date'
        )
        где split имеет train_end, val_start, val_end.

    LEGACY (backward-compat):
        build_val_day_toplists(
            train_df, val_start, val_end, window_days=3, topk_per_day=1000, min_count=1
        )
        работает только по train_df.

    Режимы:
      - mode='frozen_train': окно считаем ТОЛЬКО из train-части (<= split.train_end).
      - mode='moving': окно считаем из df_all, ограничивая верх границы (<= d-1).

    Возврат:
      dict: date(int) -> List[item_id]
    """
    # Legacy сигнатура: (train_df, val_start, val_end, ...)
    if df_all is None and split is None and len(args) >= 3:
        train_df = args[0]
        val_start = int(args[1])
        val_end = int(args[2])
        df_train = _ensure_date_alias(train_df, date_col=date_col)
        out: Dict[int, List[int]] = {}
        for d in range(val_start, val_end + 1):
            lo, hi = d - int(window_days), d - 1
            out[int(d)] = _top_items_in_window(df_train, lo, hi,
                                               date_col=date_col,
                                               topk=int(topk_per_day),
                                               min_count=int(min_count))
        return out

    # Новый стиль: df_all + split
    if df_all is None or split is None:
        raise TypeError("build_val_day_toplists: передайте df_all= и split= (или используйте legacy-вызов).")

    df_all = _ensure_date_alias(df_all, date_col=date_col)

    val_start = _get_attr(split, "val_start")
    val_end = _get_attr(split, "val_end")
    train_end = _get_attr(split, "train_end")
    if val_start is None or val_end is None:
        raise ValueError("split должен содержать val_start и val_end")
    if mode == "frozen_train" and train_end is None:
        raise ValueError("split.train_end обязателен для mode='frozen_train'")

    if mode == "frozen_train":
        df_base = df_all.loc[df_all[date_col] <= int(train_end), ["item_id", date_col]]
    elif mode == "moving":
        df_base = df_all[["item_id", date_col]]
    else:
        raise ValueError("mode должен быть 'frozen_train' или 'moving'")

    out: Dict[int, List[int]] = {}
    for d in range(int(val_start), int(val_end) + 1):
        lo, hi = d - int(window_days), d - 1
        if mode == "moving":
            df_cut = df_base.loc[df_base[date_col] <= hi]
            out[int(d)] = _top_items_in_window(df_cut, lo, hi,
                                               date_col=date_col,
                                               topk=int(topk_per_day),
                                               min_count=int(min_count))
        else:
            out[int(d)] = _top_items_in_window(df_base, lo, hi,
                                               date_col=date_col,
                                               topk=int(topk_per_day),
                                               min_count=int(min_count))
    return out


# ---------------------------------------------------------------------
# 2) Candidates from day toplists
# ---------------------------------------------------------------------

def candidates_from_day_toplists(target_pairs: pd.DataFrame,
                                 day_toplists: Dict[int, List[int]],
                                 k_top: int = 1000,
                                 date_col: str = "date") -> pd.DataFrame:
    """
    Строит кандидатов для пар (user_id, date) — подставляя соответствующий day-топ.
    Возвращает DataFrame с колонками: ['user_id','item_id','rank','score'].

    - Если для конкретного дня нет топа — вернёт пусто для этого дня.
    - score = 1.0 / (rank + 1), rank начинается с 0.
    """
    need_cols = {"user_id", date_col}
    miss = need_cols - set(target_pairs.columns)
    if miss:
        raise KeyError(f"candidates_from_day_toplists: нет колонок {sorted(miss)}")

    # аккуратно сортируем пары, чтобы стабильный порядок
    pairs = target_pairs[["user_id", date_col]].drop_duplicates().sort_values(["user_id", date_col])

    # собираем
    rows = []
    for d, grp in pairs.groupby(date_col):
        top = day_toplists.get(int(d), [])
        if not top:
            continue
        cut = top[: int(k_top)]
        # ранги/скор
        ranks = np.arange(len(cut), dtype=np.int32)
        scores = 1.0 / (ranks + 1.0)
        # развернём для всех пользователей этого дня
        users = grp["user_id"].astype("int64").values
        # кросс-продукт: каждый user получает одинаковый список cut
        for u in users:
            rows.append(pd.DataFrame({
                "user_id": u,
                "item_id": np.asarray(cut, dtype=np.int64),
                "rank": ranks,
                "score": scores.astype("float32"),
                date_col: int(d),
            }))

    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id", "rank", "score", date_col])

    out = pd.concat(rows, ignore_index=True)
    return out[["user_id", "item_id", "rank", "score", date_col]]


# ---------------------------------------------------------------------
# 3) Simple eval for trending candidates
# ---------------------------------------------------------------------

def _group_truth(val_df: pd.DataFrame) -> Dict[int, set]:
    """
    Словарь user_id -> set(item_id) по валидации (все позитивы пользователя).
    """
    v = val_df[["user_id", "item_id"]].dropna()
    v["user_id"] = v["user_id"].astype("int64")
    v["item_id"] = v["item_id"].astype("int64")
    return v.groupby("user_id")["item_id"].agg(lambda s: set(s.tolist())).to_dict()


def _group_preds(cand_df: pd.DataFrame, k: int) -> Dict[int, List[int]]:
    """
    Словарь user_id -> top-k item_id (по возрастанию rank, при равенстве — стабильный порядок).
    """
    if "rank" in cand_df.columns:
        c = cand_df.sort_values(["user_id", "rank"], kind="mergesort")
    else:
        c = cand_df.sort_values(["user_id"], kind="mergesort")
    c = c[["user_id", "item_id"]].drop_duplicates()
    topk = c.groupby("user_id")["item_id"].apply(lambda s: s.tolist()[: int(k)]).to_dict()
    return {int(u): [int(x) for x in xs] for u, xs in topk.items()}


def evaluate_trending_candidates(val_df: pd.DataFrame,
                                 cand_df: pd.DataFrame,
                                 k_list: Iterable[int] = (20, 50, 100)) -> pd.DataFrame:
    """
    Быстрая оценка: HR@k / NDCG@k / Recall@k по пользователям в `val_df`.

    HR@k: 1, если в топ-k есть хотя бы один таргет айтем пользователя.
    NDCG@k: 1/log2(rank+2) для первого найденного таргета (0, если нет).
    Recall@k: |hits ∩ positives| / |positives|.
    """
    truth = _group_truth(val_df)
    users = list(truth.keys())
    if not len(users):
        return pd.DataFrame(columns=["k", "users", "HR@k", "NDCG@k", "Recall@k"])

    rows = []
    for k in k_list:
        preds = _group_preds(cand_df, k=k)
        hr_sum, ndcg_sum, rec_sum, cnt = 0.0, 0.0, 0.0, 0
        for u in users:
            pos = truth[u]
            rec = preds.get(u, [])
            if not pos:
                continue
            cnt += 1
            # HR
            hit_positions = [i for i, it in enumerate(rec) if it in pos]
            hr = 1.0 if hit_positions else 0.0
            # NDCG — по первому хиту
            if hit_positions:
                r = hit_positions[0]
                ndcg = 1.0 / np.log2(r + 2.0)
            else:
                ndcg = 0.0
            # Recall
            inter = len(set(rec) & pos)
            recall = float(inter) / float(len(pos))
            hr_sum += hr
            ndcg_sum += ndcg
            rec_sum += recall
        if cnt == 0:
            rows.append({"k": int(k), "users": 0, "HR@k": 0.0, "NDCG@k": 0.0, "Recall@k": 0.0})
        else:
            rows.append({
                "k": int(k),
                "users": int(cnt),
                "HR@k": float(hr_sum / cnt),
                "NDCG@k": float(ndcg_sum / cnt),
                "Recall@k": float(rec_sum / cnt),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4) Coverage of frozen-train day-toplists
# ---------------------------------------------------------------------

def coverage_from_frozen_trending(day_toplists: Dict[int, List[int]],
                                  universe_items: Optional[Iterable[int]] = None) -> Tuple[int, Optional[float]]:
    """
    Покрытие уникальных айтемов, попавших в day-топы (в frozen-тренде).

    Возвращает: (unique_items, coverage_rate or None)
    - coverage_rate вычисляется, если передан universe_items (например, все айтемы в val).
    """
    all_items = set()
    for lst in day_toplists.values():
        all_items.update(int(x) for x in lst)
    uniq = len(all_items)
    if universe_items is not None:
        uni = set(int(x) for x in universe_items)
        rate = 0.0 if not uni else float(len(all_items & uni)) / float(len(uni))
        return uniq, rate
    return uniq, None


# ---------------------------------------------------------------------
# 5) Global trending from last train window
# ---------------------------------------------------------------------

def trending_global_from_last_train_window(df_all: pd.DataFrame,
                                           split: Any,
                                           window_days: int = 3,
                                           topk: int = 1000,
                                           min_count: int = 1,
                                           date_col: str = "date") -> List[int]:
    """
    Глобальный тренд-лист по последнему train-окну:
      окно = [train_end - window_days + 1, train_end]  (оба конца включительно)

    Возвращает list[item_id] длины ≤ topk.
    """
    df_all = _ensure_date_alias(df_all, date_col=date_col)
    train_end = _get_attr(split, "train_end")
    if train_end is None:
        raise ValueError("split.train_end обязателен")

    lo = int(train_end) - int(window_days) + 1
    hi = int(train_end)

    return _top_items_in_window(df_all, lo, hi,
                                date_col=date_col,
                                topk=int(topk),
                                min_count=int(min_count))
