# -*- coding: utf-8 -*-
"""
exp106_lightgcn_import.py — Stage 1 / Эксперимент 106: импорт эмбеддингов LightGCN

Что делает:
- грузит user/item эмбеддинги + маппинги (локально или из W&B-артефакта)
- строит карту seen по TRAIN
- ранжирует пользователей по dot-product (батчево), опционально по ограниченному пулу айтемов
- считает recall@M на вал
- сохраняет кандидатов и метрики, логирует в W&B (если ран активен)

Артефакты:
- artifacts/candidates/exp106_lightgcn_import/val_candidates_lgcn_M<M>_pool-<mode>.parquet
- metrics/exp106_lightgcn_import.csv
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal, Union

import numpy as np
import pandas as pd

from .base_exp import BaseExperiment
from .config import (
    PATHS, COL_USER, COL_ITEM,
    CAND_TOP_M_PER_USER, QUICK_MODE, QUICK_USERS, SEED,
)
from .artifacts import ensure_dir
from .metrics import recall_at_m
from .pop_decay import compute_pop_static, build_global_top
from .lightgcn_import import (
    ImportConfig, LGCNEmbeddings,
    load_embeddings, build_seen_map, rank_users,
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
class Exp106Config:
    # откуда брать эмбеддинги (если все пути None — попытаемся загрузить из стандартных локальных путей)
    import_cfg: ImportConfig = field(default_factory=ImportConfig)

    # пул кандидатов (ограничение для скоринга)
    pool_mode: Literal["none", "popular", "file"] = "popular"
    pool_K: int = 2000                       # размер пула для режима "popular"
    pool_file_path: Optional[Union[str, Path]] = None  # путь к файлу с item_id (csv/txt, один id на строку)

    # кандидаты и оценка
    m_per_user: int = int(CAND_TOP_M_PER_USER)
    eval_M_list: Sequence[int] = (200, 500, 1000)
    exclude_seen: bool = True                # исключать ли просмотренные в train

    # удобства
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED


@dataclass
class Exp106State:
    emb: LGCNEmbeddings
    seen_map_idx: Optional[Dict[int, set]]
    pool_items: Optional[Sequence[int]]      # item_id (сырые) или None
    cand_map: Optional[Dict[int, List[int]]]
    metrics_recall: Optional[pd.DataFrame]
    out_dir: Path
    cand_dir: Path


# ----------------------------- Реализация эксперимента -----------------------------

class Exp106LightGCNImport(BaseExperiment):
    def __init__(self, cfg: Optional[Exp106Config] = None):
        super().__init__(exp_name="exp106_lightgcn_import")
        self.cfg = cfg or Exp106Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.state: Optional[Exp106State] = None

    # ---- fit: грузим эмбеддинги, собираем карту seen и пул ----
    def fit(self, context: dict) -> Exp106State:
        self.require_context_keys(context, ["train_df", "val_df", "split"])
        train_df: pd.DataFrame = context["train_df"]

        # 1) эмбеддинги
        emb = load_embeddings(self.cfg.import_cfg)
        U, I, D = emb.dims()
        if _WANDB and wandb.run is not None:
            wandb.summary["exp106/U"] = U
            wandb.summary["exp106/I"] = I
            wandb.summary["exp106/D"] = D

        # 2) карта seen на индексах (u_idx → set(i_idx))
        seen_map_idx = build_seen_map(train_df, emb.user_map, emb.item_map) if self.cfg.exclude_seen else None

        # 3) пул айтемов (в item_id)
        pool_items = self._build_pool_items(train_df)

        self.state = Exp106State(
            emb=emb,
            seen_map_idx=seen_map_idx,
            pool_items=pool_items,
            cand_map=None,
            metrics_recall=None,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
        )
        return self.state

    # ---- candidates: ранжирование per-user ----
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

        # quick-режим — подсэмпл пользователей
        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        # ранжирование
        cand_map = rank_users(
            emb=self.state.emb,
            users=users,
            M=int(M or self.cfg.m_per_user),
            pool_items=self.state.pool_items,
            exclude_seen_map=self.state.seen_map_idx,
            use_indices_for_pool=False,
            cfg=self.cfg.import_cfg,
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

        # гарантируем кандидатов (на всех вал-пользователей)
        if self.state.cand_map is None:
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        rows = []
        for m in self.cfg.eval_M_list:
            r = recall_at_m(val_truth, self.state.cand_map, m=int(m), averaging="micro")
            rows.append({"M": int(m), "recall": float(r)})
        metrics_df = pd.DataFrame(rows)

        self.wandb_log_table("exp106_recall", metrics_df)
        self.state.metrics_recall = metrics_df
        # сохраним также сводную таблицу (как принято в base)
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

        pool_tag = self.cfg.pool_mode if self.cfg.pool_mode != "file" else "file"
        cand_path = self.save_candidates_map(
            self.state.cand_map,
            filename=f"val_candidates_lgcn_M{self.cfg.m_per_user}_pool-{pool_tag}.parquet",
        )
        return cand_path, self.metrics_path

    # ----------------------------- ВСПОМОГАТЕЛЬНЫЕ -----------------------------

    def _build_pool_items(self, train_df: pd.DataFrame) -> Optional[Sequence[int]]:
        """
        Собирает пул айтемов в item_id по конфигу:
          - "none": None (ранжирование по всем айтемам)
          - "popular": топ-K популярных по TRAIN
          - "file": читаем список item_id из файла (csv/txt, один id на строку или колонка item_id)
        """
        mode = self.cfg.pool_mode
        if mode == "none":
            return None
        if mode == "popular":
            pop = compute_pop_static(train_df)
            return build_global_top(pop, K=int(self.cfg.pool_K))
        if mode == "file":
            if not self.cfg.pool_file_path:
                raise ValueError("pool_mode='file', но pool_file_path не указан.")
            p = Path(self.cfg.pool_file_path)
            if not p.exists():
                raise FileNotFoundError(f"pool_file_path не найден: {p}")
            if p.suffix.lower() in (".csv", ".tsv"):
                df = pd.read_csv(p)
                # предполагаем колонку item_id, иначе берём первую колонку
                col = "item_id" if "item_id" in df.columns else df.columns[0]
                arr = df[col].astype("int64").tolist()
                return arr
            else:
                # простой txt: один id в строке
                vals = []
                with open(p, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        vals.append(int(line))
                return vals
        raise ValueError(f"Неизвестный pool_mode: {mode}")
