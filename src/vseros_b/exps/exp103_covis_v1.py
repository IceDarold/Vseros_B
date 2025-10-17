# -*- coding: utf-8 -*-
"""
exp103_covis_v1.py — Stage 1 / Эксперимент 103: item-item Co-Vis (v1)

Что делает:
- считает co-vis пары по корзинам (user, date) с time-decay весами
- мера сходства: cosine|jaccard|lift (по умолчанию cosine)
- формирует topN соседей на item
- генерирует per-user кандидатов из последних K семян
- оценивает recall@M на валидации
- сохраняет: neighbors_topN.parquet, val_candidates_*.parquet, metrics csv

Контракты:
    .fit(context)            → строит соседей
    .candidates(context, …)  → делает per-user кандидатов
    .evaluate(context)       → считает recall@M
    .save(context)           → сохраняет метрики и кандидатов
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from ..artifacts import ensure_dir, log_artifact, save_df
from ..base_exp import BaseExperiment
from ..config import (
    CAND_TOP_M_PER_USER,
    COL_DATE,
    COL_ITEM,
    COL_USER,
    COVIS_MIN_PAIR_COUNT,
    DECAY_LAMBDA_COVIS,
    K_RECENT_ITEMS_PER_USER,
    PATHS,
    QUICK_MODE,
    QUICK_USERS,
    SEED,
    TOPN_NEIGHBORS_PER_ITEM,
)
from ..metrics import recall_at_m
from ..candidates.covis import (
    CoVisConfig,
    build_neighbors,
    candidates_from_covis,
    last_k_items_by_user,
)


# ----------------------------- Конфиг и состояние -----------------------------

@dataclass
class Exp103Config:
    # построение соседей
    decay_lambda: float = float(DECAY_LAMBDA_COVIS)
    score: Literal["cosine", "jaccard", "lift"] = "cosine"
    topn_per_item: int = int(TOPN_NEIGHBORS_PER_ITEM)
    min_pair_count: int = int(COVIS_MIN_PAIR_COUNT)
    weighted_support: bool = True

    # генерация кандидатов
    k_recent: int = int(K_RECENT_ITEMS_PER_USER)
    m_per_user: int = int(CAND_TOP_M_PER_USER)
    exclude_seed: bool = True

    # оценка
    eval_M_list: Sequence[int] = (200, 500, 1000)

    # удобства
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED


@dataclass
class Exp103State:
    neighbors_df: pd.DataFrame
    cand_map: Optional[Dict[int, List[int]]]
    metrics_recall: Optional[pd.DataFrame]
    neighbors_path: Path
    out_dir: Path
    cand_dir: Path


# ----------------------------- Реализация эксперимента -----------------------------

class Exp103CoVisV1(BaseExperiment):
    def __init__(self, cfg: Optional[Exp103Config] = None):
        super().__init__(exp_name="exp103_covis_v1")
        self.cfg = cfg or Exp103Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.neighbors_path = self.out_dir / f"neighbors_topN_{self.cfg.score}_l{str(self.cfg.decay_lambda).replace('.','')}.parquet"
        self.state: Optional[Exp103State] = None

    # ---- fit: строим соседей на item ----
    def fit(self, context: dict) -> Exp103State:
        """
        Требует в context: basket_items (из TRAIN), split.
        """
        self.require_context_keys(context, ["basket_items", "split"])
        basket_items: pd.DataFrame = context["basket_items"]
        split = context["split"]

        cfg = CoVisConfig(
            decay_lambda=float(self.cfg.decay_lambda),
            ref_day=int(split.train_end),
            score=self.cfg.score,
            topn_per_item=int(self.cfg.topn_per_item),
            min_pair_count=int(self.cfg.min_pair_count),
            weighted_support=bool(self.cfg.weighted_support),
        )

        neighbors_df = build_neighbors(basket_items, cfg=cfg)  # item_id, neighbor_id, score
        save_df(self.neighbors_path, neighbors_df, index=False)
        log_artifact(self.neighbors_path, name=f"{self.exp_name}_neighbors", type_="dataset")

        self.state = Exp103State(
            neighbors_df=neighbors_df,
            cand_map=None,
            metrics_recall=None,
            neighbors_path=self.neighbors_path,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
        )
        return self.state

    # ---- candidates: per-user top-M из последних K семян ----
    def candidates(
        self,
        context: dict,
        users: Optional[Sequence[int]] = None,
        M: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        """
        Генерирует кандидатов для указанных users (или для всех пользователей валидации).
        """
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])

        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame = context["val_df"]
        split = context["split"]
        val_truth: Mapping[int, set] = context["val_truth"]

        # список пользователей для генерации
        if users is None:
            users = list(val_truth.keys())

        # quick-режим: ограничим число пользователей
        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        # последние K айтемов у каждого (до конца train)
        seeds_full = last_k_items_by_user(train_df, k=int(self.cfg.k_recent), upto_day=int(split.train_end))
        seeds = {int(u): seeds_full.get(int(u), []) for u in users}

        # кандидаты
        cand_map = candidates_from_covis(
            recent_items_by_user=seeds,
            neighbors_df_or_map=self.state.neighbors_df,
            M=int(M or self.cfg.m_per_user),
            exclude_seed=bool(self.cfg.exclude_seed),
        )

        # сохраним в состоянии (для evaluate/save)
        if self.state.cand_map is None:
            self.state.cand_map = {}
        self.state.cand_map.update(cand_map)

        return cand_map

    # ---- evaluate: recall@M ----
    def evaluate(self, context: dict) -> pd.DataFrame:
        """
        Считает recall@M по списку M. Если кандидаты ещё не построены — строит для всех вал-пользователей.
        """
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])

        val_truth: Mapping[int, set] = context["val_truth"]

        # гарантируем, что есть кандидаты для всех вал-пользователей
        if self.state.cand_map is None:
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        rows = []
        for m in self.cfg.eval_M_list:
            r = recall_at_m(val_truth, self.state.cand_map, m=int(m), averaging="micro")
            rows.append({"M": int(m), "recall": float(r)})
        metrics_df = pd.DataFrame(rows)

        # логи и сохранение метрик (позже сохраним ещё в save())
        self.wandb_log_table("exp103_recall", metrics_df)
        self.state.metrics_recall = metrics_df
        return metrics_df

    # ---- save: кандидаты + метрики ----
    def save(self, context: dict) -> Tuple[Path, Path]:
        """
        Сохраняет:
          - metrics csv
          - candidates parquet (для всех пользователей, что есть в self.state.cand_map)
        """
        assert self.state is not None, "Run fit() first."
        # метрики
        metrics_path = self.save_metrics_df(
            self.state.metrics_recall if self.state.metrics_recall is not None else pd.DataFrame(),
            filename=f"{self.exp_name}.csv",
            artifact_name=f"{self.exp_name}_metrics",
        )

        # кандидаты
        if self.state.cand_map is None:
            # как fallback — соберём всех пользователей валидации
            val_truth: Mapping[int, set] = context["val_truth"]
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        cand_path = self.save_candidates_map(
            self.state.cand_map,
            filename=f"val_candidates_covis_K{self.cfg.k_recent}_M{self.cfg.m_per_user}.parquet",
        )

        return cand_path, metrics_path
