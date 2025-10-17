# -*- coding: utf-8 -*-
"""
exp105_item2vec.py — Stage 1 / Эксперимент 105: Item2Vec (SGNS-lite)

Пайплайн:
- предложения из корзин (user,date)
- Word2Vec (skip-gram + negative sampling)
- извлечь эмбеддинги → top-N соседей по косинусу (FAISS если доступен, иначе батчевый матмул)
- пер-юзер кандидаты из последних K семян
- оценка recall@M

Артефакты:
- artifacts/exp105_item2vec/item2vec_emb_dim<D>_ep<E>.parquet
- artifacts/exp105_item2vec/neighbors_i2v_topN_<N>.parquet
- artifacts/candidates/exp105_item2vec/val_candidates_i2v_K<K>_M<M>.parquet
- metrics/exp105_item2vec.csv
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..artifacts import ensure_dir, log_artifact, save_df
from ..base_exp import BaseExperiment
from ..config import (
    CAND_TOP_M_PER_USER,
    COL_DATE,
    COL_ITEM,
    COL_USER,
    FORCE_REBUILD,
    I2V_DIM,
    I2V_NS,
    I2V_TOPN,
    K_RECENT_ITEMS_PER_USER,
    PATHS,
    QUICK_MODE,
    QUICK_USERS,
    SEED,
)
from ..metrics import recall_at_m
from ..candidates.item2vec_lite import (
    I2VConfig,
    build_neighbors_from_embeddings,
    build_sentences_from_baskets,
    candidates_from_item2vec,
    embeddings_to_df,
    extract_item_embeddings,
    train_item2vec,
)

# W&B — мягко
try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None
    _WANDB = False


# ----------------------------- Конфиг и состояние -----------------------------

@dataclass
class Exp105Config:
    # Word2Vec (SGNS)
    vector_size: int = int(I2V_DIM)
    window: int = 5
    negative: int = int(I2V_NS)
    epochs: int = 5
    min_count: int = 1
    sample: float = 1e-5
    ns_exponent: float = 0.75
    alpha: float = 0.025
    min_alpha: float = 0.0007
    workers: int = 4

    # NN
    topn_neighbors_per_item: int = int(I2V_TOPN)
    use_faiss: bool = True
    batch_size_nn: int = 8192

    # кандидаты
    k_recent: int = int(K_RECENT_ITEMS_PER_USER)
    m_per_user: int = int(CAND_TOP_M_PER_USER)
    exclude_seed: bool = True

    # оценка
    eval_M_list: Sequence[int] = (200, 500, 1000)

    # удобства
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED
    force_rebuild: bool = FORCE_REBUILD  # пересчитывать даже если артефакты есть


@dataclass
class Exp105State:
    embeddings_df: pd.DataFrame
    neighbors_df: pd.DataFrame
    cand_map: Optional[Dict[int, List[int]]]
    metrics_recall: Optional[pd.DataFrame]
    out_dir: Path
    cand_dir: Path
    emb_path: Path
    neighbors_path: Path


# ----------------------------- Реализация эксперимента -----------------------------

class Exp105Item2Vec(BaseExperiment):
    def __init__(self, cfg: Optional[Exp105Config] = None):
        super().__init__(exp_name="exp105_item2vec")
        self.cfg = cfg or Exp105Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        # имена файлов под параметры
        self.emb_path = self.out_dir / f"item2vec_emb_dim{self.cfg.vector_size}_ep{self.cfg.epochs}.parquet"
        self.neighbors_path = self.out_dir / f"neighbors_i2v_topN_{self.cfg.topn_neighbors_per_item}.parquet"
        self.state: Optional[Exp105State] = None

    # ---- fit: тренировка i2v + соседи ----
    def fit(self, context: dict) -> Exp105State:
        """
        Требует: basket_items, train_df, split
        """
        self.require_context_keys(context, ["basket_items", "train_df", "split"])
        basket_items: pd.DataFrame = context["basket_items"]
        train_df: pd.DataFrame = context["train_df"]
        split = context["split"]

        rng = np.random.default_rng(self.cfg.seed)

        # quick-режим: можно подсэмплировать пользователей для обучения (для скорости)
        if self.cfg.quick_mode and self.cfg.quick_users and self.cfg.quick_users > 0:
            uniq_users = basket_items[COL_USER].unique()
            if len(uniq_users) > self.cfg.quick_users:
                keep_users = set(rng.choice(uniq_users, size=self.cfg.quick_users, replace=False))
                basket_items = basket_items[basket_items[COL_USER].isin(keep_users)].copy()

        # если эмбеддинги уже посчитаны и не нужно пересчитывать — просто загрузим
        if (not self.cfg.force_rebuild) and self.emb_path.exists() and self.neighbors_path.exists():
            emb_df = pd.read_parquet(self.emb_path)
            neighbors_df = pd.read_parquet(self.neighbors_path)
            self.state = Exp105State(
                embeddings_df=emb_df,
                neighbors_df=neighbors_df,
                cand_map=None,
                metrics_recall=None,
                out_dir=self.out_dir,
                cand_dir=self.cand_dir,
                emb_path=self.emb_path,
                neighbors_path=self.neighbors_path,
            )
            return self.state

        # 1) предложения из корзин
        sentences = build_sentences_from_baskets(
            basket_items=basket_items,
            shuffle_within_basket=True,
            as_str_tokens=True,
            seed=self.cfg.seed,
        )

        # 2) тренировка SGNS
        i2v_cfg = I2VConfig(
            vector_size=self.cfg.vector_size,
            window=self.cfg.window,
            negative=self.cfg.negative,
            epochs=self.cfg.epochs,
            min_count=self.cfg.min_count,
            sample=self.cfg.sample,
            ns_exponent=self.cfg.ns_exponent,
            alpha=self.cfg.alpha,
            min_alpha=self.cfg.min_alpha,
            workers=self.cfg.workers,
            seed=self.cfg.seed,
            topn_neighbors_per_item=self.cfg.topn_neighbors_per_item,
            use_faiss=self.cfg.use_faiss,
            batch_size_nn=self.cfg.batch_size_nn,
            k_recent=self.cfg.k_recent,
            m_per_user=self.cfg.m_per_user,
            exclude_seed=self.cfg.exclude_seed,
        )
        model = train_item2vec(sentences, i2v_cfg)

        # 3) извлечь эмбеддинги
        item_ids, emb = extract_item_embeddings(model)
        emb_df = embeddings_to_df(item_ids, emb)
        save_df(self.emb_path, emb_df, index=False)
        log_artifact(self.emb_path, name=f"{self.exp_name}_embeddings", type_="model")

        # 4) соседи по косинусу
        neighbors_df = build_neighbors_from_embeddings(
            item_ids=item_ids,
            emb=emb,
            topn_per_item=self.cfg.topn_neighbors_per_item,
            use_faiss=self.cfg.use_faiss,
            batch_size_nn=self.cfg.batch_size_nn,
        )
        save_df(self.neighbors_path, neighbors_df, index=False)
        log_artifact(self.neighbors_path, name=f"{self.exp_name}_neighbors", type_="dataset")

        self.state = Exp105State(
            embeddings_df=emb_df,
            neighbors_df=neighbors_df,
            cand_map=None,
            metrics_recall=None,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
            emb_path=self.emb_path,
            neighbors_path=self.neighbors_path,
        )
        return self.state

    # ---- candidates: per-user top-M ----
    def candidates(
        self,
        context: dict,
        users: Optional[Sequence[int]] = None,
        M: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])

        train_df: pd.DataFrame = context["train_df"]
        split = context["split"]
        val_truth: Mapping[int, set] = context["val_truth"]

        # таргетные пользователи
        if users is None:
            users = list(val_truth.keys())

        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        cand_map = candidates_from_item2vec(
            train_df=train_df,
            split=split,
            neighbors_df=self.state.neighbors_df,
            k_recent=self.cfg.k_recent,
            m_per_user=int(M or self.cfg.m_per_user),
            exclude_seed=self.cfg.exclude_seed,
            users=users,
        )

        if self.state.cand_map is None:
            self.state.cand_map = {}
        self.state.cand_map.update(cand_map)
        return cand_map

    # ---- evaluate: recall@M ----
    def evaluate(self, context: dict) -> pd.DataFrame:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["val_truth"])

        val_truth: Mapping[int, set] = context["val_truth"]
        if self.state.cand_map is None:
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        rows = []
        for m in self.cfg.eval_M_list:
            r = recall_at_m(val_truth, self.state.cand_map, m=int(m), averaging="micro")
            rows.append({"M": int(m), "recall": float(r)})
        metrics_df = pd.DataFrame(rows)

        self.wandb_log_table("exp105_recall", metrics_df)
        self.state.metrics_recall = metrics_df
        return metrics_df

    # ---- save: метрики + кандидаты ----
    def save(self, context: dict):
        assert self.state is not None, "Run fit() first."
        metrics_path = self.save_metrics_df(
            self.state.metrics_recall if self.state.metrics_recall is not None else pd.DataFrame(),
            filename=f"{self.exp_name}.csv",
            artifact_name=f"{self.exp_name}_metrics",
        )
        if self.state.cand_map is None:
            val_truth: Mapping[int, set] = context["val_truth"]
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        cand_path = self.save_candidates_map(
            self.state.cand_map,
            filename=f"val_candidates_i2v_K{self.cfg.k_recent}_M{self.cfg.m_per_user}.parquet",
        )
        return cand_path, metrics_path
