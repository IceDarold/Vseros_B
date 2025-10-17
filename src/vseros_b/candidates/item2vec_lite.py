# -*- coding: utf-8 -*-
"""
item2vec_lite.py — SGNS (Word2Vec) по «корзинам дня» (user, date).

Что умеет:
- build_sentences_from_baskets(basket_items): собирает «предложения» как наборы item_id внутри (user, date)
- train_item2vec(sentences, cfg): учит эмбеддинги (gensim.Word2Vec, skip-gram + negative sampling)
- extract_item_embeddings(model): извлекает item_id → вектор
- build_neighbors_from_embeddings(...): topN по косинусной близости (FAISS, если доступен; иначе — батчевый матмул)
- candidates_from_item2vec(...): per-user кандидаты из последних K семян (переиспользуем логику из covis)

Зависимости: numpy, pandas, gensim (для обучения), опционально faiss (для быстрых NN).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# попытка подключить FAISS (ускоряет NN)
try:
    import faiss  # type: ignore
    _FAISS = True
except Exception:
    faiss = None  # type: ignore
    _FAISS = False

# gensim — для SGNS
try:
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec
    _GENSIM = True
except Exception:
    Word2Vec = None  # type: ignore
    CallbackAny2Vec = object  # type: ignore
    _GENSIM = False

from ..config import (
    COL_USER,
    COL_ITEM,
    COL_DATE,
    I2V_DIM,
    I2V_NS,
    I2V_TOPN,
    K_RECENT_ITEMS_PER_USER,
    CAND_TOP_M_PER_USER,
    SEED,
)
from .covis import last_k_items_by_user, candidates_from_covis


# ----------------------------- Конфиг -----------------------------

@dataclass
class I2VConfig:
    vector_size: int = int(I2V_DIM)
    window: int = 5                   # симметричное окно внутри корзины (порядок условный)
    sg: int = 1                       # 1 = skip-gram
    negative: int = int(I2V_NS)       # число отрицательных примеров
    epochs: int = 5
    min_count: int = 1
    sample: float = 1e-5              # subsampling для частых токенов
    ns_exponent: float = 0.75         # степень для распределения негативов
    alpha: float = 0.025
    min_alpha: float = 0.0007
    workers: int = 4
    seed: int = int(SEED)

    # NN и кандидаты
    topn_neighbors_per_item: int = int(I2V_TOPN)
    use_faiss: bool = True
    batch_size_nn: int = 8192         # для fallback-режима без faiss

    # генерация кандидатов
    k_recent: int = int(K_RECENT_ITEMS_PER_USER)
    m_per_user: int = int(CAND_TOP_M_PER_USER)
    exclude_seed: bool = True


# ----------------------------- Подготовка предложений -----------------------------

def build_sentences_from_baskets(
    basket_items: pd.DataFrame,
    shuffle_within_basket: bool = True,
    as_str_tokens: bool = True,
    seed: int = SEED,
) -> List[List[str]]:
    """
    Из (user, date, item) строит «предложения»: по одной строке на каждую (user, date).
    Порядок внутри корзины не известен — можем перемешать для разнообразия контекста.

    as_str_tokens=True — gensim предпочитает строки (и экономит на кастах).
    """
    if not isinstance(basket_items, pd.DataFrame):
        raise TypeError("basket_items must be a pandas DataFrame.")

    # убеждаемся в уникальности триплетов
    b = basket_items.drop_duplicates([COL_USER, COL_DATE, COL_ITEM])
    rng = np.random.default_rng(seed)

    sentences: List[List[str]] = []
    for _, g in b.groupby([COL_USER, COL_DATE], sort=False):
        items = g[COL_ITEM].astype(np.int64).tolist()
        if shuffle_within_basket and len(items) > 1:
            rng.shuffle(items)
        if as_str_tokens:
            sentences.append([str(it) for it in items])
        else:
            sentences.append(items)  # type: ignore
    return sentences


# ----------------------------- Тренировка Word2Vec (SGNS) -----------------------------

class _LossLogger(CallbackAny2Vec):  # type: ignore
    def __init__(self):
        self.epoch = 0
        self.loss_prev = 0.0

    def on_epoch_end(self, model):  # type: ignore
        try:
            loss = model.get_latest_training_loss()
            delta = loss - self.loss_prev
            self.loss_prev = loss
            print(f"[i2v] epoch {self.epoch} loss={loss:.2f} (+{delta:.2f})")
        except Exception:
            pass
        self.epoch += 1


def train_item2vec(
    sentences: List[List[str]],
    cfg: Optional[I2VConfig] = None,
) -> "Word2Vec":
    """
    Обучает SGNS-модель на предложениях.
    Возвращает gensim.models.Word2Vec.
    """
    if not _GENSIM:
        raise ImportError("gensim не установлен. Установи 'gensim' перед обучением item2vec.")

    cfg = cfg or I2VConfig()

    model = Word2Vec(
        sentences=sentences,
        vector_size=int(cfg.vector_size),
        window=int(cfg.window),
        sg=int(cfg.sg),
        negative=int(cfg.negative),
        epochs=int(cfg.epochs),
        min_count=int(cfg.min_count),
        sample=float(cfg.sample),
        ns_exponent=float(cfg.ns_exponent),
        alpha=float(cfg.alpha),
        min_alpha=float(cfg.min_alpha),
        workers=int(cfg.workers),
        seed=int(cfg.seed),
        compute_loss=True,
    )
    return model


# ----------------------------- Извлечение эмбеддингов -----------------------------

def extract_item_embeddings(model: "Word2Vec") -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (item_ids:int64[N], emb:float32[N,D]) только для токенов, которые видел тренинг.
    """
    # gensim 4.x: ключи в model.wv.key_to_index
    keys = list(model.wv.key_to_index.keys())
    item_ids = np.array([int(k) for k in keys], dtype=np.int64)
    emb = model.wv.vectors.astype(np.float32)  # shape [N, D]
    return item_ids, emb


# ----------------------------- Top-N соседи по косинусу -----------------------------

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n


def _build_neighbors_faiss(
    item_ids: np.ndarray,
    emb: np.ndarray,
    topn: int,
    batch: int = 65536,
) -> pd.DataFrame:
    """
    Быстрый NN через FAISS (cosine как dot после L2-нормализации).
    Возвращает DataFrame[item_id, neighbor_id, score] без самих себя.
    """
    d = emb.shape[1]
    xb = _normalize_rows(emb.copy())
    index = faiss.IndexFlatIP(d)
    index.add(xb)

    # ищем (topn+1), чтобы убрать self
    K = int(topn) + 1
    out_rows = []
    for start in range(0, xb.shape[0], batch):
        end = min(start + batch, xb.shape[0])
        D, I = index.search(xb[start:end], K)  # cosine ≈ dot (normalized)
        for qi in range(end - start):
            src = int(item_ids[start + qi])
            for rank in range(K):
                j = I[qi, rank]
                if j < 0:
                    continue
                dst = int(item_ids[j])
                if dst == src:
                    continue
                out_rows.append((src, dst, float(D[qi, rank])))
    df = pd.DataFrame(out_rows, columns=["item_id", "neighbor_id", "score"])
    # оставим topN per item с сортировкой
    df = df.sort_values(["item_id", "score"], ascending=[True, False])
    df["rank"] = df.groupby("item_id").cumcount()
    df = df[df["rank"] < int(topn)].drop(columns=["rank"])
    df["item_id"] = df["item_id"].astype("int64")
    df["neighbor_id"] = df["neighbor_id"].astype("int64")
    df["score"] = df["score"].astype("float64")
    return df.reset_index(drop=True)


def _build_neighbors_bruteforce(
    item_ids: np.ndarray,
    emb: np.ndarray,
    topn: int,
    batch_q: int = 8192,
) -> pd.DataFrame:
    """
    Fallback без FAISS: батчевый косинус через матмул.
    ВНИМАНИЕ: O(N^2) по времени, но с умеренной памятью (по батчу); используйте в quick-режиме или на сэмпле.
    """
    xb = _normalize_rows(emb.copy())  # [N, D]
    N, D = xb.shape
    out_rows = []

    for start in range(0, N, batch_q):
        end = min(start + batch_q, N)
        Q = xb[start:end]                    # [B, D]
        S = Q @ xb.T                         # [B, N]
        # уберём self-сходства
        for i in range(end - start):
            S[i, start + i] = -np.inf

        # берём topn
        idx = np.argpartition(-S, kth=min(topn, N - 1) - 1, axis=1)[:, :topn]
        # сортируем внутри topn
        part_scores = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top_idx = np.take_along_axis(idx, order, axis=1)
        top_scores = np.take_along_axis(part_scores, order, axis=1)

        for i in range(end - start):
            src = int(item_ids[start + i])
            for j in range(topn):
                dst = int(item_ids[top_idx[i, j]])
                sc = float(top_scores[i, j])
                out_rows.append((src, dst, sc))

    df = pd.DataFrame(out_rows, columns=["item_id", "neighbor_id", "score"])
    df["item_id"] = df["item_id"].astype("int64")
    df["neighbor_id"] = df["neighbor_id"].astype("int64")
    df["score"] = df["score"].astype("float64")
    return df.reset_index(drop=True)


def build_neighbors_from_embeddings(
    item_ids: np.ndarray,
    emb: np.ndarray,
    topn_per_item: int = I2V_TOPN,
    use_faiss: bool = True,
    batch_size_nn: int = 8192,
) -> pd.DataFrame:
    """
    Универсальная обёртка: строит таблицу соседей по косинусу.
    """
    if emb is None or len(emb) == 0:
        return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype({
            "item_id": "int64", "neighbor_id": "int64", "score": "float64"
        })

    if use_faiss and _FAISS:
        return _build_neighbors_faiss(item_ids, emb, topn=int(topn_per_item))
    else:
        print("[item2vec] FAISS недоступен → fallback на батчевый матмул (медленно на больших N).")
        return _build_neighbors_bruteforce(item_ids, emb, topn=int(topn_per_item), batch_q=int(batch_size_nn))


# ----------------------------- Кандидаты для пользователей -----------------------------

def candidates_from_item2vec(
    train_df: pd.DataFrame,
    split,
    neighbors_df: pd.DataFrame,
    k_recent: int = K_RECENT_ITEMS_PER_USER,
    m_per_user: int = CAND_TOP_M_PER_USER,
    exclude_seed: bool = True,
    users: Optional[Sequence[int]] = None,
) -> Dict[int, List[int]]:
    """
    Строит per-user кандидатов: суммирование скор neighbor’ов от последних K семян.
    """
    if users is None:
        # все пользователи из валидации (лучше передать контекстом отдельно)
        users = train_df[COL_USER].unique().tolist()

    # последние K айтемов на пользователя до конца train
    seeds_full = last_k_items_by_user(train_df, k=int(k_recent), upto_day=int(split.train_end))
    seeds = {int(u): seeds_full.get(int(u), []) for u in users}

    cand_map = candidates_from_covis(
        recent_items_by_user=seeds,
        neighbors_df_or_map=neighbors_df,
        M=int(m_per_user),
        exclude_seed=bool(exclude_seed),
    )
    return cand_map


# ----------------------------- Утилиты для сохранения/загрузки -----------------------------

def embeddings_to_df(item_ids: np.ndarray, emb: np.ndarray) -> pd.DataFrame:
    """
    Преобразует эмбеддинги в плоский DataFrame с колонками:
      item_id, f0, f1, ... f{D-1}
    Удобно хранить как parquet.
    """
    item_ids = item_ids.reshape(-1, 1).astype(np.int64)
    df = pd.DataFrame(np.concatenate([item_ids, emb.astype(np.float32)], axis=1))
    cols = ["item_id"] + [f"f{i}" for i in range(emb.shape[1])]
    df.columns = cols
    return df


def df_to_embeddings(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обратное преобразование: DataFrame → (item_ids, emb)
    """
    if "item_id" not in df.columns:
        raise ValueError("DataFrame must contain 'item_id' column.")
    item_ids = df["item_id"].to_numpy(dtype=np.int64)
    feat_cols = [c for c in df.columns if c != "item_id"]
    emb = df[feat_cols].to_numpy(dtype=np.float32)
    return item_ids, emb
