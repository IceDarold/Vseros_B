# -*- coding: utf-8 -*-
"""
Метрики для Stage 1 (recall & базовые ранжировочные).
Никаких предположений о путях/рантайме — чистые функции.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
from math import log2


# ----------------------------- ранжировочные метрики -----------------------------

def apk(actual: Set[int], pred: Sequence[int], k: int = 20) -> float:
    """
    AP@K для одного пользователя.
    actual: множество релевантных item_id (на вал-окне)
    pred:   список предсказанных item_id по убыванию
    """
    if not actual or not pred:
        return 0.0
    score, hits = 0.0, 0
    for i, p in enumerate(pred[:k]):
        if p in actual:
            hits += 1
            score += hits / float(i + 1)
    return score / min(len(actual), k)


def mapk(actual_dict: Mapping[int, Set[int]],
         pred_dict: Mapping[int, Sequence[int]],
         k: int = 20) -> float:
    users = set(actual_dict.keys()) & set(pred_dict.keys())
    if not users:
        return 0.0
    return float(np.mean([apk(actual_dict[u], pred_dict[u], k) for u in users]))


def hitk(actual_dict: Mapping[int, Set[int]],
         pred_dict: Mapping[int, Sequence[int]],
         k: int = 20) -> float:
    users = set(actual_dict.keys()) & set(pred_dict.keys())
    if not users:
        return 0.0
    return float(np.mean([int(len(actual_dict[u].intersection(set(pred_dict[u][:k]))) > 0) for u in users]))


def ndcgk(actual_dict: Mapping[int, Set[int]],
          pred_dict: Mapping[int, Sequence[int]],
          k: int = 20) -> float:
    """
    NDCG@K: per-user DCG / IDCG, где IDCG считается как сумма 1/log2(i+2) для i=0..min(k,|actual|)-1
    """
    users = set(actual_dict.keys()) & set(pred_dict.keys())
    if not users:
        return 0.0

    def dcg(rel_positions: Iterable[int]) -> float:
        return sum(1.0 / log2(i + 2) for i in rel_positions)

    scores: List[float] = []
    for u in users:
        actual = actual_dict[u]
        pred = pred_dict[u][:k]
        pos = [i for i, p in enumerate(pred) if p in actual]
        idcg = sum(1.0 / log2(i + 2) for i in range(min(k, len(actual))))
        scores.append(dcg(pos) / (idcg if idcg > 0 else 1.0))

    return float(np.mean(scores))


# ----------------------------- recall для кандидатов -----------------------------

def recall_at_m(actual_dict: Mapping[int, Set[int]],
                cand_dict: Mapping[int, Sequence[int]],
                m: int,
                averaging: str = "micro") -> float:
    """
    Recall@M (для источников кандидатов).
    averaging = "micro" → (Σ хитов) / (Σ релевантных)
    averaging = "macro" → среднее по пользователям
    """
    users = set(actual_dict.keys()) & set(cand_dict.keys())
    if not users:
        return 0.0

    if averaging == "micro":
        hits = 0
        total = 0
        for u in users:
            act = actual_dict[u]
            c = set(cand_dict[u][:m]) if isinstance(cand_dict[u], Sequence) else set(cand_dict[u])
            hits += len(act & c)
            total += len(act)
        return (hits / total) if total > 0 else 0.0

    # macro
    vals: List[float] = []
    for u in users:
        act = actual_dict[u]
        c = set(cand_dict[u][:m]) if isinstance(cand_dict[u], Sequence) else set(cand_dict[u])
        denom = len(act)
        vals.append((len(act & c) / denom) if denom > 0 else 0.0)
    return float(np.mean(vals))


# ----------------------------- coverage для глобалок -----------------------------

def coverage_at_k(item_scores: Union[pd.Series, pd.DataFrame],
                  val_item_cnt: pd.Series,
                  ks: Sequence[int]) -> pd.DataFrame:
    """
    Coverage@K: доля вал-интеракций, покрытых топ-K айтемов по item_scores.
    item_scores: Series(index=item_id, values=score) или DataFrame с колонкой 'score' и индексом item_id
    val_item_cnt: Series item_id -> # вал-интеракций
    """
    if isinstance(item_scores, pd.DataFrame):
        if "score" not in item_scores.columns:
            raise ValueError("DataFrame item_scores должен содержать колонку 'score'.")
        s = item_scores["score"]
        s.index = item_scores.index
    else:
        s = item_scores

    order = s.sort_values(ascending=False)
    joined = order.to_frame("score").merge(val_item_cnt.to_frame("val_cnt"),
                                           left_index=True, right_index=True, how="left").fillna(0)
    joined["val_cnt"] = joined["val_cnt"].astype("int64")

    total_val = int(val_item_cnt.sum()) if len(val_item_cnt) else 1
    out: List[dict] = []
    cum = 0
    idx = 0
    for K in ks:
        while idx < min(K, len(joined)):
            cum += int(joined.iloc[idx]["val_cnt"])
            idx += 1
        out.append({"K": int(K), "coverage": (cum / total_val) if total_val > 0 else 0.0})
    return pd.DataFrame(out)


# ----------------------------- дрейф популярности -----------------------------

def spearman_rank_corr(train_cnt: pd.Series, val_cnt: pd.Series) -> float:
    """
    Приближённый Spearman по рангам (без учета тай-брека как в scipy).
    """
    tr = train_cnt.rank(method="first", ascending=False)
    va = val_cnt.rank(method="first", ascending=False)
    join = tr.to_frame("train_rank").join(va.to_frame("val_rank"), how="inner")
    if len(join) < 2:
        return float("nan")
    r = np.corrcoef(join["train_rank"].to_numpy(), join["val_rank"].to_numpy())[0, 1]
    return float(r)


def jaccard_topk(train_cnt: pd.Series, val_cnt: pd.Series, ks: Sequence[int]) -> pd.DataFrame:
    rows = []
    tr_sorted = train_cnt.sort_values(ascending=False)
    va_sorted = val_cnt.sort_values(ascending=False)
    for K in ks:
        S_tr = set(tr_sorted.head(int(K)).index)
        S_va = set(va_sorted.head(int(K)).index)
        inter = len(S_tr & S_va)
        union = len(S_tr | S_va) if (S_tr or S_va) else 1
        rows.append({"K": int(K), "jaccard": inter / union})
    return pd.DataFrame(rows)
