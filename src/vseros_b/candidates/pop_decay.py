# -*- coding: utf-8 -*-
"""
pop_decay.py — утилиты для глобальной популярности:
- статическая популярность
- time-decay популярность: w = exp(-λ * max(0, ref - date))
- λ-свип
- coverage@K на валидации
- построение глобального топа и простые предсказания на пользователя
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import COL_USER, COL_ITEM, COL_DATE
from ..metrics import coverage_at_k, mapk, hitk, ndcgk


# ============================== расчёт скоров ==============================

def compute_pop_static(
    train_df: pd.DataFrame,
    ensure_unique_triplets: bool = False,
) -> pd.Series:
    """
    Статическая популярность: количество появлений айтема в TRAIN.
    Ожидается, что train_df уже дедуплен по (user,date,item). Если нет — передай ensure_unique_triplets=True.
    Returns: pd.Series(index=item_id, values=pop_static:int64)
    """
    df = train_df
    if ensure_unique_triplets:
        df = df.drop_duplicates([COL_USER, COL_DATE, COL_ITEM])
    s = (
        df.groupby(COL_ITEM)[COL_DATE]
          .count()
          .rename("pop_static")
          .astype("int64")
    )
    return s


def compute_pop_decay(
    train_df: pd.DataFrame,
    lambda_val: float,
    ref_day: Optional[int] = None,
    ensure_unique_triplets: bool = False,
) -> pd.Series:
    """
    Time-decay популярность: веса событий w = exp(-λ * age), где age = max(0, ref_day - date).
    Если ref_day=None → разумно использовать конец train (но обычно вычисляется снаружи).
    Returns: pd.Series(index=item_id, values=decayed_sum:float64)
    """
    df = train_df
    if ensure_unique_triplets:
        df = df.drop_duplicates([COL_USER, COL_DATE, COL_ITEM])

    if ref_day is None:
        ref_day = int(df[COL_DATE].max())

    age = (ref_day - df[COL_DATE]).clip(lower=0)
    w = np.exp(-float(lambda_val) * age.astype("float64"))

    s = (
        df.assign(w=w)
          .groupby(COL_ITEM)["w"]
          .sum()
          .rename(f"pop_decay_l{str(lambda_val).replace('.','')}")
          .astype("float64")
    )
    return s


def sweep_pop_decay(
    train_df: pd.DataFrame,
    lambdas: Sequence[float],
    ref_day: Optional[int] = None,
    ensure_unique_triplets: bool = False,
) -> Dict[float, pd.Series]:
    """
    Считает decayed-популярность для набора λ. Возвращает dict: λ -> Series(item→score).
    """
    out: Dict[float, pd.Series] = {}
    for lam in lambdas:
        out[lam] = compute_pop_decay(
            train_df=train_df,
            lambda_val=float(lam),
            ref_day=ref_day,
            ensure_unique_triplets=ensure_unique_triplets,
        )
    return out


# ============================== топ-листы и предикты ==============================

def build_global_top(score_s: pd.Series, K: int) -> List[int]:
    """
    Возвращает top-K item_id по убыванию score.
    """
    K = int(K)
    if K <= 0:
        return []
    order = score_s.sort_values(ascending=False).head(K)
    # гарантируем int64 id
    return list(order.index.astype("int64"))


def prepare_seen_map(train_df: pd.DataFrame) -> Dict[int, set]:
    """
    user -> множество item'ов, встречавшихся у него в TRAIN (для фильтрации).
    """
    g = train_df.groupby(COL_USER)[COL_ITEM].apply(lambda s: set(map(int, s.values)))
    return {int(k): v for k, v in g.to_dict().items()}


def predict_per_user_global(
    users: Sequence[int],
    global_top: Sequence[int],
    k: int = 20,
    seen_map: Optional[Mapping[int, set]] = None,
    extra_pool: int = 500,
) -> Dict[int, List[int]]:
    """
    Возвращает предсказания для каждого пользователя:
      - без фильтра: всем один и тот же список global_top[:k]
      - с фильтром: исключаем айтемы, которые этот пользователь уже видел в TRAIN (seen_map)
    extra_pool — запас длины пула, чтобы после фильтра точно набрать k.
    """
    users = list(map(int, users))
    global_top = list(map(int, global_top))
    k = int(k)

    if k <= 0:
        return {u: [] for u in users}

    if seen_map is None:
        base = global_top[:k]
        return {u: base for u in users}

    pool = list(global_top[: max(k + int(extra_pool), k)])
    preds: Dict[int, List[int]] = {}
    for u in users:
        seen = seen_map.get(u, set())
        out: List[int] = []
        for it in pool:
            if it not in seen:
                out.append(it)
                if len(out) == k:
                    break
        if len(out) < k:
            # если не хватило — добиваем из глобалки (крайне редко)
            need = k - len(out)
            out += global_top[:need]
        preds[u] = out
    return preds


# ============================== оценка на валидации ==============================

@dataclass
class GlobalEvalResult:
    variant: str
    k_eval: int
    map_at_k: float
    hit_at_k: float
    ndcg_at_k: float


def evaluate_global_predictions(
    val_truth: Mapping[int, set],
    preds_by_user: Mapping[int, Sequence[int]],
    k_eval: int = 20,
    variant_name: str = "global",
) -> GlobalEvalResult:
    """
    Считает mAP@K / Hit@K / NDCG@K для глобальных предсказаний.
    """
    m = mapk(val_truth, preds_by_user, k=k_eval)
    h = hitk(val_truth, preds_by_user, k=k_eval)
    n = ndcgk(val_truth, preds_by_user, k=k_eval)
    return GlobalEvalResult(variant=variant_name, k_eval=k_eval, map_at_k=m, hit_at_k=h, ndcg_at_k=n)


def coverage_curves(
    item_score: Union[pd.Series, pd.DataFrame],
    val_item_cnt: pd.Series,
    ks: Sequence[int],
) -> pd.DataFrame:
    """
    Обёртка над metrics.coverage_at_k с унифицированным выходом.
    Returns: DataFrame[K, coverage]
    """
    return coverage_at_k(item_score, val_item_cnt, ks=ks)


# ============================== удобные сценарии для оркестратора ==============================

def best_variant_by_map(
    metrics_df: pd.DataFrame,
    variant_col: str = "variant",
    map_col: str = "map@20",
) -> Optional[str]:
    """
    Возвращает имя лучшего варианта по метрике (например, "decay@0.05" или "static").
    Ожидает таблицу с колонками [variant, map@20].
    """
    if metrics_df is None or metrics_df.empty:
        return None
    row = metrics_df.sort_values(map_col, ascending=False).iloc[0]
    return str(row[variant_col])


def assemble_pop_table(
    pop_static: Optional[pd.Series],
    pop_decay_dict: Optional[Dict[float, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Склеивает одну таблицу со всеми скорингами популярности для удобного сохранения/логирования.
    Колонки: ['pop_static', 'pop_decay_l002', 'pop_decay_l005', ...]
    """
    frames: List[pd.Series] = []
    if pop_static is not None:
        frames.append(pop_static.rename("pop_static"))
    if pop_decay_dict:
        for lam, s in pop_decay_dict.items():
            frames.append(s.rename(f"pop_decay_l{str(lam).replace('.','')}"))
    if not frames:
        return pd.DataFrame()
    tbl = pd.concat(frames, axis=1).fillna(0.0)
    tbl.index.name = COL_ITEM
    return tbl
