# -*- coding: utf-8 -*-
"""
covis.py — item-item co-visitation:
- корзины = (user_id, date) с уникальными (u, d, i)
- пары (i, j) внутри корзины, опционально с time-decay весом w = exp(-λ * max(0, ref - date))
- меры сходства: cosine / jaccard / lift
- topN соседей для каждого item
- генерация кандидатов для пользователей из последних K семян

Независим от путей/рантайма; использует только pandas/numpy и локальные константы из config.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
from itertools import combinations

from .config import (
    COL_USER, COL_ITEM, COL_DATE,
    DECAY_LAMBDA_COVIS, TOPN_NEIGHBORS_PER_ITEM,
    K_RECENT_ITEMS_PER_USER, CAND_TOP_M_PER_USER,
    COVIS_MIN_PAIR_COUNT,
)


# ============================== базовые вычисления ==============================

@dataclass
class CoVisConfig:
    decay_lambda: float = float(DECAY_LAMBDA_COVIS)  # λ в w = exp(-λ * age)
    ref_day: Optional[int] = None                    # если None → возьмём max(date) входного df
    score: Literal["cosine", "jaccard", "lift"] = "cosine"
    topn_per_item: int = int(TOPN_NEIGHBORS_PER_ITEM)
    min_pair_count: int = int(COVIS_MIN_PAIR_COUNT)  # фильтр по количеству совместных появлений (без весов)
    # если True — для поддержки (support) используем те же веса, что и для пар
    weighted_support: bool = True


def _ensure_basket_unique(basket_items: pd.DataFrame) -> pd.DataFrame:
    """
    Гарантируем уникальность (user, date, item). Если уже дедуплено — дёшево вернётся как есть.
    """
    if not basket_items.duplicated([COL_USER, COL_DATE, COL_ITEM]).any():
        return basket_items
    return basket_items.drop_duplicates([COL_USER, COL_DATE, COL_ITEM]).reset_index(drop=True)


def compute_support(
    basket_items: pd.DataFrame,
    decay_lambda: float = 0.0,
    ref_day: Optional[int] = None,
    weighted: bool = True,
) -> pd.Series:
    """
    Поддержка айтема (# корзин, где он встретился).
    Если weighted=True → сумма весов w; иначе — счётчик корзин.
    Returns: pd.Series(index=item_id, values=support(float64|int64))
    """
    b = _ensure_basket_unique(basket_items)
    if weighted and decay_lambda > 0:
        if ref_day is None:
            ref_day = int(b[COL_DATE].max())
        age = (ref_day - b[COL_DATE]).clip(lower=0).astype("float64")
        w = np.exp(-float(decay_lambda) * age)
        sup = b.assign(w=w).groupby(COL_ITEM)["w"].sum().astype("float64")
        sup.name = "support_w"
    else:
        sup = b.groupby(COL_ITEM)[COL_DATE].count().astype("int64")
        sup.name = "support_cnt"
    return sup


def compute_pair_counts(
    basket_items: pd.DataFrame,
    decay_lambda: float = 0.0,
    ref_day: Optional[int] = None,
) -> pd.DataFrame:
    """
    Строит пары (i, j) внутри каждой корзины (user, date).
    Возвращает DataFrame с колонками:
        item_i, item_j, co_cnt, co_w
    где:
        - co_cnt — сумма счётчиков (каждая корзина даёт +1 на пару)
        - co_w   — сумма весов w = exp(-λ * age) (если λ=0 → co_w == co_cnt)
    Правило: i < j (упорядочиваем для уверенной агрегации без дублей).
    """
    b = _ensure_basket_unique(basket_items)
    if b.empty:
        return pd.DataFrame(columns=["item_i", "item_j", "co_cnt", "co_w"]).astype({
            "item_i": "int64", "item_j": "int64", "co_cnt": "int64", "co_w": "float64"
        })

    if ref_day is None:
        ref_day = int(b[COL_DATE].max())
    # вес корзины по её дате
    if decay_lambda > 0:
        age = (ref_day - b[COL_DATE]).clip(lower=0).astype("float64")
        b = b.assign(_w=np.exp(-float(decay_lambda) * age))
    else:
        b = b.assign(_w=1.0)

    # группируем по корзине и генерируем пары
    pairs: List[Tuple[int, int, int, float]] = []  # (i, j, cnt, w)
    for _, g in b.groupby([COL_USER, COL_DATE], sort=False):
        items = g[COL_ITEM].values
        if len(items) < 2:
            continue
        w = float(g["_w"].iloc[0])  # вес всей корзины
        # все неупорядоченные пары без повторов
        for a, c in combinations(np.unique(items), 2):
            i, j = (int(a), int(c)) if a < c else (int(c), int(a))
            pairs.append((i, j, 1, w))

    if not pairs:
        return pd.DataFrame(columns=["item_i", "item_j", "co_cnt", "co_w"]).astype({
            "item_i": "int64", "item_j": "int64", "co_cnt": "int64", "co_w": "float64"
        })

    df = pd.DataFrame(pairs, columns=["item_i", "item_j", "co_cnt", "co_w"])
    agg = (df.groupby(["item_i", "item_j"], as_index=False)
             .agg(co_cnt=("co_cnt", "sum"), co_w=("co_w", "sum")))
    agg["item_i"] = agg["item_i"].astype("int64")
    agg["item_j"] = agg["item_j"].astype("int64")
    agg["co_cnt"] = agg["co_cnt"].astype("int64")
    agg["co_w"]   = agg["co_w"].astype("float64")
    return agg


# ============================== скоринг пар и топ-N соседи ==============================

def score_pairs(
    pairs_df: pd.DataFrame,
    support: pd.Series,
    score: Literal["cosine", "jaccard", "lift"] = "cosine",
    use_weighted: bool = True,
    total_baskets: Optional[int] = None,
) -> pd.DataFrame:
    """
    Присваивает каждой паре (i, j) скор по выбранной мере.
    support: Series item→support (по возможности согласованный с use_weighted)
    use_weighted: если True — используем co_w, иначе co_cnt
    total_baskets: для lift (если None — оценим по сумме support_cnt)
    Возвращает pairs_df c колонкой 'score'.
    """
    if pairs_df.empty:
        return pairs_df.assign(score=np.zeros(0, dtype="float64"))

    x = "co_w" if use_weighted else "co_cnt"
    df = pairs_df.copy()

    # джойним поддержки для i и j
    sup = support.rename("sup")
    df = df.join(sup, on="item_i").rename(columns={"sup": "sup_i"})
    df = df.join(sup, on="item_j").rename(columns={"sup": "sup_j"})

    # защита от нулей
    df["sup_i"] = df["sup_i"].replace(0, np.nan)
    df["sup_j"] = df["sup_j"].replace(0, np.nan)

    if score == "cosine":
        df["score"] = df[x] / np.sqrt(df["sup_i"] * df["sup_j"])
    elif score == "jaccard":
        df["score"] = df[x] / (df["sup_i"] + df["sup_j"] - df[x])
    elif score == "lift":
        # lift = P(i,j) / (P(i) P(j)); приблизим через счётчики/веса
        # total_baskets: лучше брать сумму support (не уникальных корзин при weighted=False)
        if total_baskets is None:
            # оценим как сумму "безвесовых" поддержек (если weighted — это лишь приближение)
            total_baskets = int(support.sum()) if support is not None else 1
        df["score"] = (df[x] / total_baskets) / ((df["sup_i"] / total_baskets) * (df["sup_j"] / total_baskets))
    else:
        raise ValueError(f"Unknown score: {score}")

    df["score"] = df["score"].fillna(0.0).astype("float64")
    return df


def topn_neighbors_from_pairs(
    scored_pairs: pd.DataFrame,
    topn_per_item: int = TOPN_NEIGHBORS_PER_ITEM,
    min_pair_count: int = COVIS_MIN_PAIR_COUNT,
    use_weighted: bool = True,
) -> pd.DataFrame:
    """
    Делает «двунаправленных» соседей (для каждого i добавляет j, и наоборот),
    фильтрует по co_cnt (и/или co_w) и оставляет topN по score на item.
    Возвращает DataFrame: [item_id, neighbor_id, score]
    """
    if scored_pairs.empty:
        return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype({
            "item_id": "int64", "neighbor_id": "int64", "score": "float64"
        })

    # фильтр по количеству совместных появлений (устойчивость связи)
    keep = scored_pairs["co_cnt"] >= int(min_pair_count)
    df = scored_pairs.loc[keep, ["item_i", "item_j", "score"]].copy()

    # двунаправленно
    a = df.rename(columns={"item_i": "item_id", "item_j": "neighbor_id"})
    b = df.rename(columns={"item_j": "item_id", "item_i": "neighbor_id"})
    nei = pd.concat([a, b], ignore_index=True)

    # topN на айтем
    nei = nei.sort_values(["item_id", "score"], ascending=[True, False])
    nei["rank"] = nei.groupby("item_id").cumcount()
    nei = nei[nei["rank"] < int(topn_per_item)].drop(columns=["rank"])
    nei["item_id"] = nei["item_id"].astype("int64")
    nei["neighbor_id"] = nei["neighbor_id"].astype("int64")
    nei["score"] = nei["score"].astype("float64")
    return nei.reset_index(drop=True)


def build_neighbors(
    basket_items: pd.DataFrame,
    cfg: Optional[CoVisConfig] = None,
) -> pd.DataFrame:
    """
    Полный пайплайн: support + pairs + score + topN.
    basket_items: уникальные (user, date, item) из TRAIN
    Returns: DataFrame[item_id, neighbor_id, score]
    """
    cfg = cfg or CoVisConfig()

    # поддержка
    sup = compute_support(
        basket_items,
        decay_lambda=cfg.decay_lambda,
        ref_day=cfg.ref_day,
        weighted=cfg.weighted_support,
    )

    # пары
    pairs = compute_pair_counts(
        basket_items,
        decay_lambda=cfg.decay_lambda,
        ref_day=cfg.ref_day,
    )

    # скоринг
    scored = score_pairs(
        pairs_df=pairs,
        support=sup,
        score=cfg.score,
        use_weighted=(cfg.decay_lambda > 0 and cfg.weighted_support),
    )

    # соседи topN
    nei = topn_neighbors_from_pairs(
        scored_pairs=scored,
        topn_per_item=cfg.topn_per_item,
        min_pair_count=cfg.min_pair_count,
        use_weighted=(cfg.decay_lambda > 0),
    )
    return nei


# ============================== кандидаты для пользователей ==============================

def last_k_items_by_user(
    df: pd.DataFrame,
    k: int = K_RECENT_ITEMS_PER_USER,
    upto_day: Optional[int] = None,
) -> Dict[int, List[int]]:
    """
    Собирает последние k айтемов на пользователя, опционально ограничив по дню (<= upto_day).
    Ожидается, что df — TRAIN; если нет — поставь upto_day = split.train_end.
    """
    if upto_day is not None:
        df = df[df[COL_DATE] <= int(upto_day)]
    # упорядочим по дате так, чтобы последние были «свежее»
    df = df.sort_values([COL_USER, COL_DATE])
    # заберём последние k уникальных по item для каждого пользователя
    out: Dict[int, List[int]] = {}
    for u, g in df.groupby(COL_USER, sort=False):
        items = g.drop_duplicates([COL_ITEM], keep="last")[COL_ITEM].tail(int(k)).tolist()
        out[int(u)] = list(map(int, items))
    return out


def neighbors_to_map(nei_df: pd.DataFrame, keep_score: bool = True) -> Dict[int, List[Tuple[int, float]]]:
    """
    Преобразует таблицу соседей в dict: item → [(neighbor, score), ...] по убыванию score.
    """
    out: Dict[int, List[Tuple[int, float]]] = {}
    if nei_df.empty:
        return out
    for i, g in nei_df.sort_values(["item_id", "score"], ascending=[True, False]).groupby("item_id", sort=True):
        if keep_score:
            out[int(i)] = [(int(r["neighbor_id"]), float(r["score"])) for _, r in g.iterrows()]
        else:
            out[int(i)] = [int(x) for x in g["neighbor_id"].tolist()]
    return out


def candidates_from_covis(
    recent_items_by_user: Mapping[int, Sequence[int]],
    neighbors_df_or_map: Mapping[int, Sequence[Tuple[int, float]]] | pd.DataFrame,
    M: int = CAND_TOP_M_PER_USER,
    exclude_seed: bool = True,
) -> Dict[int, List[int]]:
    """
    Генерирует per-user кандидатов: суммируем скор соседей для последних K айтемов пользователя.
    - recent_items_by_user: user → [seed_i1, seed_i2, ...]
    - neighbors_df_or_map: либо DataFrame[item_id, neighbor_id, score], либо dict item→[(nbr,score)]
    - M: сколько кандидатов вернуть
    - exclude_seed: исключать ли сами seed-айтемы из кандидатов
    Returns: user → [item_id1, item_id2, ...] (top-M по суммарному скору)
    """
    # подготовим map item→[(nbr,score)]
    if isinstance(neighbors_df_or_map, pd.DataFrame):
        nbr_map = neighbors_to_map(neighbors_df_or_map, keep_score=True)
    else:
        nbr_map = neighbors_df_or_map  # уже map

    out: Dict[int, List[int]] = {}
    M = int(M)

    for u, seeds in recent_items_by_user.items():
        bucket: Dict[int, float] = {}
        seed_set = set(seeds) if exclude_seed else set()
        for s in seeds:
            for (nbr, sc) in nbr_map.get(int(s), []):
                if exclude_seed and nbr in seed_set:
                    continue
                bucket[nbr] = bucket.get(nbr, 0.0) + float(sc)
        if not bucket:
            out[int(u)] = []
            continue
        # сортировка по суммарному скору, затем по id для стабильности
        ordered = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
        out[int(u)] = [it for it, _ in ordered[:M]]
    return out
