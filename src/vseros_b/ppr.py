# -*- coding: utf-8 -*-
"""
ppr.py — Personalized PageRank на бипартитном графе user–item.

Идея:
- Строим разреженную матрицу A (U×I) с весами (unit / count / time-decay).
- Нормируем:
    R = row-normalized(A)        # p(item | user)
    C = column-normalized(A)     # p(user | item)
- Делаем 2-шаговую итерацию на item-пространстве:
    x_{t+1} = α * v + (1-α) * (R^T @ (C @ x_t))
  где v — персонализированный вектор по seed item’ам.
- Возвращаем top-M item’ов как кандидатов.

Утилиты:
- build_bipartite_matrices(...) → GraphPPR (R, C, маппинги)
- recent_items_by_user(...)     → seeds (user→последние K item_id)
- ppr_scores_items(...)         → скор по item’ам для одного набора seed’ов
- ppr_candidates_from_seeds(...)→ candidates per-user (item_id)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import (
    COL_USER, COL_ITEM, COL_DATE,
    K_RECENT_ITEMS_PER_USER, CAND_TOP_M_PER_USER,
)
# переиспользуем готовую функцию сбора последних K айтемов
from .covis import last_k_items_by_user


# ============================== модели данных ==============================

@dataclass
class GraphPPR:
    """Контейнер нормированных матриц и маппингов для PPR."""
    U: int
    I: int
    R: sp.csr_matrix  # [U×I], row-normalized: p(item | user)
    C: sp.csc_matrix  # [U×I], column-normalized: p(user | item)
    user_map: pd.DataFrame  # [user_id, u_idx]
    item_map: pd.DataFrame  # [item_id, i_idx]

    def shapes(self) -> Tuple[int, int]:
        return self.U, self.I


# ============================== построение матриц ==============================

def _aggregate_weights(
    df: pd.DataFrame,
    mode: Literal["unit", "count", "decay"] = "decay",
    decay_lambda: float = 0.05,
    ref_day: Optional[int] = None,
) -> pd.DataFrame:
    """
    Агрегирует веса для (user,item):
    - unit  → вес = 1, если пара встречалась хотя бы раз
    - count → вес = # встреч
    - decay → вес = sum(exp(-λ * max(0, ref - day))) по всем встречам
    Возвращает df: [user_id, item_id, w]
    """
    x = df[[COL_USER, COL_ITEM, COL_DATE]].copy()
    if mode == "unit":
        g = (x.drop_duplicates([COL_USER, COL_ITEM])
               .assign(w=1.0))
        return g[[COL_USER, COL_ITEM, "w"]]

    if mode == "count":
        g = (x.groupby([COL_USER, COL_ITEM], as_index=False)[COL_DATE]
               .count()
               .rename(columns={COL_DATE: "w"}))
        g["w"] = g["w"].astype("float64")
        return g[[COL_USER, COL_ITEM, "w"]]

    # decay
    if ref_day is None:
        ref_day = int(x[COL_DATE].max())
    age = (ref_day - x[COL_DATE]).clip(lower=0).astype("float64")
    w = np.exp(-float(decay_lambda) * age)
    g = (x.assign(_w=w)
           .groupby([COL_USER, COL_ITEM], as_index=False)["_w"].sum()
           .rename(columns={"_w": "w"}))
    g["w"] = g["w"].astype("float64")
    return g[[COL_USER, COL_ITEM, "w"]]


def build_bipartite_matrices(
    train_df: pd.DataFrame,
    weight_mode: Literal["unit", "count", "decay"] = "decay",
    decay_lambda: float = 0.05,
    ref_day: Optional[int] = None,
    users: Optional[Sequence[int]] = None,  # можно ограничить вселенную (quick)
    items: Optional[Sequence[int]] = None,
) -> GraphPPR:
    """
    Строит нормированные матрицы R и C из TRAIN.
    - R[u,*] = p(item|user) (row-normalized CSR)
    - C[*,j] = p(user|item) (column-normalized CSC)
    """
    df = train_df[[COL_USER, COL_ITEM, COL_DATE]].copy()

    if users is not None:
        df = df[df[COL_USER].isin(users)]
    if items is not None:
        df = df[df[COL_ITEM].isin(items)]
    if df.empty:
        raise ValueError("Empty train_df after filtering — нечего строить.")

    # агрегируем веса по (u,i)
    ui = _aggregate_weights(df, mode=weight_mode, decay_lambda=decay_lambda, ref_day=ref_day)

    # стабильные маппинги
    u_ids = pd.Index(ui[COL_USER].unique(), name=COL_USER)
    i_ids = pd.Index(ui[COL_ITEM].unique(), name=COL_ITEM)
    user_map = pd.DataFrame({COL_USER: u_ids, "u_idx": np.arange(len(u_ids), dtype=np.int64)})
    item_map = pd.DataFrame({COL_ITEM: i_ids, "i_idx": np.arange(len(i_ids), dtype=np.int64)})

    U = len(user_map)
    I = len(item_map)

    # индексы
    tmp = (ui.merge(user_map, on=COL_USER, how="left")
              .merge(item_map, on=COL_ITEM, how="left"))[["u_idx", "i_idx", "w"]]
    u = tmp["u_idx"].to_numpy(dtype=np.int64)
    i = tmp["i_idx"].to_numpy(dtype=np.int64)
    w = tmp["w"].to_numpy(dtype=np.float64)

    # A — U×I (CSR)
    A = sp.csr_matrix((w, (u, i)), shape=(U, I), dtype=np.float64)

    # R: row-normalize (user→item)
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    row_inv = np.divide(1.0, np.maximum(row_sum, 1e-12), where=(row_sum > 0))
    R = sp.diags(row_inv).dot(A)  # CSR, строки суммируются к 1

    # C: column-normalize (item→user)
    C = A.tocsc(copy=True)
    col_sum = np.asarray(C.sum(axis=0)).ravel()
    # масштабируем данные по колонкам
    indptr = C.indptr
    data = C.data
    for j in range(I):
        s = col_sum[j]
        if s > 0:
            data[indptr[j]:indptr[j+1]] /= s
        else:
            # колонка пустая — оставим нули
            pass

    C = sp.csc_matrix((data, C.indices, indptr), shape=(U, I), dtype=np.float64)

    return GraphPPR(U=U, I=I, R=R.tocsr(), C=C.tocsc(), user_map=user_map, item_map=item_map)


# ============================== seeds / вспомогательное ==============================

def recent_items_by_user_ids(
    train_df: pd.DataFrame,
    split,
    k: int = K_RECENT_ITEMS_PER_USER,
) -> Dict[int, List[int]]:
    """
    Обёртка над covis.last_k_items_by_user: собирает последние k item_id на пользователя,
    обрезая по train_end.
    """
    return last_k_items_by_user(train_df, k=int(k), upto_day=int(split.train_end))


def _items_to_iidx(items: Sequence[int], item_map: pd.DataFrame) -> np.ndarray:
    """Конвертирует список item_id → i_idx (без -1)."""
    df = pd.DataFrame({COL_ITEM: pd.Series(items, dtype="int64")})
    idx = (df.merge(item_map, on=COL_ITEM, how="left")["i_idx"]
             .dropna().astype("int64").to_numpy())
    return idx


# ============================== PPR на item-пространстве ==============================

def ppr_scores_items(
    G: GraphPPR,
    seed_item_idx: Sequence[int],
    alpha: float = 0.15,
    iters: int = 20,
    tol: Optional[float] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Считает Personalized PageRank по item-узлам.
    Возвращает вектор длины I (float64).
    """
    if len(seed_item_idx) == 0:
        return np.zeros(G.I, dtype=np.float64)

    # персонализированный вектор v
    v = np.zeros(G.I, dtype=np.float64)
    seed_idx = np.asarray(seed_item_idx, dtype=np.int64)
    seed_idx = seed_idx[(seed_idx >= 0) & (seed_idx < G.I)]
    if seed_idx.size == 0:
        return np.zeros(G.I, dtype=np.float64)
    v[seed_idx] = 1.0
    v /= v.sum()

    x = v.copy()
    Rt = G.R.T.tocsr()  # [I×U]
    for t in range(iters):
        # item→user→item
        y = G.C.dot(x)     # [U]
        z = Rt.dot(y)      # [I]
        x_new = alpha * v + (1.0 - alpha) * z
        if tol is not None:
            if np.linalg.norm(x_new - x, ord=1) <= tol:
                x = x_new
                if verbose:
                    print(f"[PPR] converged at iter={t+1}")
                break
        x = x_new
    return x


def topk_items_from_scores(
    scores: np.ndarray,
    M: int,
    exclude: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Берёт top-M индексов по scores, исключая список exclude (индексы i_idx).
    """
    M = int(M)
    if M <= 0 or scores.size == 0:
        return []
    if exclude:
        mask = np.ones_like(scores, dtype=bool)
        mask[np.asarray(list(exclude), dtype=np.int64)] = False
        idx = np.argpartition(scores[mask], kth=min(M, mask.sum())-1)[-M:]
        part = np.argsort(-scores[mask][idx])
        # восстановим настоящие индексы
        true_idx = np.flatnonzero(mask)[idx][part]
        return [int(i) for i in true_idx[:M]]
    else:
        k = min(M, scores.size)
        idx = np.argpartition(scores, kth=k-1)[-k:]
        part = np.argsort(-scores[idx])
        return [int(i) for i in idx[part][:M]]


# ============================== Кандидаты per-user ==============================

def ppr_candidates_from_seeds(
    G: GraphPPR,
    seeds_by_user_item_ids: Mapping[int, Sequence[int]],
    M: int = CAND_TOP_M_PER_USER,
    alpha: float = 0.15,
    iters: int = 20,
    exclude_seen_map: Optional[Mapping[int, Sequence[int]]] = None,  # u_idx→i_idx set/list (на ИНДЕКСАХ!)
    return_item_ids: bool = True,
) -> Dict[int, List[int]]:
    """
    Строит карту кандидатов для пользователей на основе их seed item’ов.
    seeds_by_user_item_ids: user_id → [item_id, ...] (сырые id)
    exclude_seen_map: ожидается в ИНДЕКСАХ (u_idx→set(i_idx)); можешь собрать через lightgcn_import.build_seen_map(...)
    return_item_ids=True — вернуть item_id; иначе вернуть i_idx.
    """
    # подготовим обратные словари для быстрого маппинга
    u2idx = G.user_map.set_index(COL_USER)["u_idx"].to_dict()
    i2idx = G.item_map.set_index(COL_ITEM)["i_idx"].to_dict()
    idx2item = G.item_map.set_index("i_idx")[COL_ITEM].to_dict()

    out: Dict[int, List[int]] = {}
    for u_id, seed_items in seeds_by_user_item_ids.items():
        u_idx = u2idx.get(int(u_id), None)
        if u_idx is None:
            out[int(u_id)] = []
            continue

        seed_iidx = [i2idx[i] for i in seed_items if i in i2idx]
        if not seed_iidx:
            out[int(u_id)] = []
            continue

        scores = ppr_scores_items(G, seed_iidx, alpha=alpha, iters=iters, tol=None, verbose=False)

        # исключить просмотренные (на индексах)
        exclude_idx = None
        if exclude_seen_map is not None:
            exclude_idx = exclude_seen_map.get(int(u_idx), None)

        top_iidx = topk_items_from_scores(scores, M=int(M), exclude=exclude_idx)

        if return_item_ids:
            out[int(u_id)] = [int(idx2item[i]) for i in top_iidx if i in idx2item]
        else:
            out[int(u_id)] = [int(i) for i in top_iidx]

    return out


# ============================== Быстрый end-to-end ==============================

def build_graph_and_candidates(
    train_df: pd.DataFrame,
    split,
    weight_mode: Literal["unit", "count", "decay"] = "decay",
    decay_lambda: float = 0.05,
    alpha: float = 0.15,
    iters: int = 20,
    k_recent: int = K_RECENT_ITEMS_PER_USER,
    M: int = CAND_TOP_M_PER_USER,
    users_subset: Optional[Sequence[int]] = None,
    exclude_seen_idx_map: Optional[Mapping[int, Sequence[int]]] = None,
) -> Tuple[GraphPPR, Dict[int, List[int]]]:
    """
    Удобная обёртка: строит граф и сразу отдаёт кандидатов (по последним K item’ам).
    """
    ref = int(train_df[COL_DATE].max())
    G = build_bipartite_matrices(
        train_df=train_df,
        weight_mode=weight_mode,
        decay_lambda=decay_lambda,
        ref_day=ref,
        users=users_subset,
        items=None,
    )
    seeds = recent_items_by_user_ids(train_df, split, k=int(k_recent))
    cand = ppr_candidates_from_seeds(
        G, seeds, M=int(M), alpha=alpha, iters=iters,
        exclude_seen_map=exclude_seen_idx_map, return_item_ids=True,
    )
    return G, cand
