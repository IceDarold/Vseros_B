# -*- coding: utf-8 -*-
"""
exp107_ppr.py — Stage 1 / Эксперимент 107: Personalized PageRank на user–item графе

Пайплайн:
- строим GraphPPR (R, C) по TRAIN (веса: unit|count|decay)
- собираем seeds: последние K item_id для каждого пользователя
- для заданного списка пользователей считаем PPR по item-пространству и берём top-M кандидатов
- считаем recall@M на вал
- сохраняем артефакты: user_map/item_map, graph_meta.json, кандидаты, метрики

Артефакты:
- artifacts/exp107_ppr/user_map.parquet
- artifacts/exp107_ppr/item_map.parquet
- artifacts/exp107_ppr/graph_meta.json
- artifacts/candidates/exp107_ppr/val_candidates_ppr_K<K>_M<M>_a<alpha>_it<iters>.parquet
- metrics/exp107_ppr.csv
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import json

from ..artifacts import ensure_dir, log_artifact, save_json
from ..base_exp import BaseExperiment
from ..candidates.ppr import (
    GraphPPR,
    build_bipartite_matrices,
    ppr_candidates_from_seeds,
    recent_items_by_user_ids,
)
from ..config import (
    CAND_TOP_M_PER_USER,
    COL_DATE,
    COL_ITEM,
    COL_USER,
    K_RECENT_ITEMS_PER_USER,
    PATHS,
    QUICK_MODE,
    QUICK_USERS,
    SEED,
)
from ..metrics import recall_at_m


# ----------------------------- Конфиг и состояние -----------------------------

@dataclass
class Exp107Config:
    # построение графа
    weight_mode: Literal["unit", "count", "decay"] = "decay"
    decay_lambda: float = 0.05         # для weight_mode="decay"
    ref_day: Optional[int] = None      # если None → split.train_end

    # PPR
    alpha: float = 0.15
    iters: int = 20

    # seeds и кандидаты
    k_recent: int = int(K_RECENT_ITEMS_PER_USER)
    m_per_user: int = int(CAND_TOP_M_PER_USER)
    exclude_seen: bool = True          # исключать просмотренные из выдачи

    # оценка
    eval_M_list: Sequence[int] = (200, 500, 1000)

    # удобства
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED


@dataclass
class Exp107State:
    G: GraphPPR
    seen_idx_map: Optional[Dict[int, set]]
    seeds_by_user_ids: Dict[int, List[int]]
    cand_map: Optional[Dict[int, List[int]]]
    metrics_recall: Optional[pd.DataFrame]
    out_dir: Path
    cand_dir: Path
    user_map_path: Path
    item_map_path: Path
    graph_meta_path: Path


# ----------------------------- Реализация эксперимента -----------------------------

class Exp107PPR(BaseExperiment):
    def __init__(self, cfg: Optional[Exp107Config] = None):
        super().__init__(exp_name="exp107_ppr")
        self.cfg = cfg or Exp107Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.state: Optional[Exp107State] = None

    # ---- fit: строим граф и подготовим все вспомогательные структуры ----
    def fit(self, context: dict) -> Exp107State:
        self.require_context_keys(context, ["train_df", "val_df", "split"])
        train_df: pd.DataFrame = context["train_df"]
        split = context["split"]

        ref = int(self.cfg.ref_day if self.cfg.ref_day is not None else split.train_end)

        # 1) граф (R, C) и маппинги
        G = build_bipartite_matrices(
            train_df=train_df,
            weight_mode=self.cfg.weight_mode,
            decay_lambda=float(self.cfg.decay_lambda),
            ref_day=ref,
            users=None,
            items=None,
        )

        # 2) seeds: последние k айтемов на пользователя (item_id)
        seeds = recent_items_by_user_ids(train_df, split, k=int(self.cfg.k_recent))

        # 3) карта seen на ИНДЕКСАХ (u_idx → set(i_idx)), если нужно исключать просмотренные
        seen_idx = None
        if self.cfg.exclude_seen:
            # замапим train_df через маппинги G
            tmp = (train_df.merge(G.user_map, on=COL_USER, how="inner")
                           .merge(G.item_map, on=COL_ITEM, how="inner"))[["u_idx", "i_idx"]]
            seen_idx = {}
            for u, g in tmp.groupby("u_idx", sort=False):
                seen_idx[int(u)] = set(int(x) for x in g["i_idx"].values)

        # 4) сохраняем маппинги и метаданные графа
        user_map_path = self.out_dir / "user_map.parquet"
        item_map_path = self.out_dir / "item_map.parquet"
        graph_meta_path = self.out_dir / "graph_meta.json"

        G.user_map.to_parquet(user_map_path, index=False)
        G.item_map.to_parquet(item_map_path, index=False)

        meta = {
            "U": int(G.U),
            "I": int(G.I),
            "R_nnz": int(G.R.nnz),
            "C_nnz": int(G.C.nnz),
            "weight_mode": self.cfg.weight_mode,
            "decay_lambda": float(self.cfg.decay_lambda),
            "alpha": float(self.cfg.alpha),
            "iters": int(self.cfg.iters),
            "k_recent": int(self.cfg.k_recent),
            "ref_day": int(ref),
        }
        save_json(graph_meta_path, meta)

        # логнем артефакты
        log_artifact(user_map_path, name=f"{self.exp_name}_user_map", type_="dataset")
        log_artifact(item_map_path, name=f"{self.exp_name}_item_map", type_="dataset")
        log_artifact(graph_meta_path, name=f"{self.exp_name}_graph_meta", type_="metadata")

        # state
        self.state = Exp107State(
            G=G,
            seen_idx_map=seen_idx,
            seeds_by_user_ids=seeds,
            cand_map=None,
            metrics_recall=None,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
            user_map_path=user_map_path,
            item_map_path=item_map_path,
            graph_meta_path=graph_meta_path,
        )
        return self.state

    # ---- candidates: считаем PPR per-user и формируем топы ----
    def candidates(
        self,
        context: dict,
        users: Optional[Sequence[int]] = None,
        M: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["val_df", "val_truth"])

        val_truth: Mapping[int, set] = context["val_truth"]
        # таргетные пользователи
        if users is None:
            users = list(val_truth.keys())

        # quick-режим — ограничим число пользователей
        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        # выделим seeds только для нужных пользователей
        seeds_sub = {int(u): self.state.seeds_by_user_ids.get(int(u), []) for u in users}

        cand_map = ppr_candidates_from_seeds(
            G=self.state.G,
            seeds_by_user_item_ids=seeds_sub,
            M=int(M or self.cfg.m_per_user),
            alpha=float(self.cfg.alpha),
            iters=int(self.cfg.iters),
            exclude_seen_map=self.state.seen_idx_map,
            return_item_ids=True,
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

        # гарантируем, что есть кандидаты для всех вал-пользователей
        if self.state.cand_map is None:
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        rows = []
        for m in self.cfg.eval_M_list:
            r = recall_at_m(val_truth, self.state.cand_map, m=int(m), averaging="micro")
            rows.append({"M": int(m), "recall": float(r)})
        metrics_df = pd.DataFrame(rows)

        self.wandb_log_table("exp107_recall", metrics_df)
        self.state.metrics_recall = metrics_df
        # сохраним единой таблицей (как во всех эксп.)
        self.save_metrics_df(metrics_df, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")
        return metrics_df

    # ---- save: кандидаты + метрики ----
    def save(self, context: dict):
        assert self.state is not None, "Run fit() first."

        # метрики уже сохранялись в evaluate(); повторим на всякий случай
        if self.state.metrics_recall is not None:
            self.save_metrics_df(self.state.metrics_recall, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")

        # кандидаты
        if self.state.cand_map is None:
            val_truth: Mapping[int, set] = context["val_truth"]
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        fname = f"val_candidates_ppr_K{self.cfg.k_recent}_M{self.cfg.m_per_user}_a{str(self.cfg.alpha).replace('.','')}_it{self.cfg.iters}.parquet"
        cand_path = self.save_candidates_map(self.state.cand_map, filename=fname)
        return cand_path, self.metrics_path
