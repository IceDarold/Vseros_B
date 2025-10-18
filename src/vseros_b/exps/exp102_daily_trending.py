# -*- coding: utf-8 -*-
"""
exp102_daily_trending.py — Stage 1 / Эксперимент 102:
Ежедневные тренды (скользящая популярность) с базовым контрактом BaseExperiment.

Режимы:
- frozen_train: один замороженный топ из последних W дней TRAIN (без утечек)
- rolling_online: для каждого вал-дня d строим топ по окну [d-W .. d-1] (train + вал-история до d)

Артефакты:
- artifacts/exp102_daily_trending/day_tops_mode-<mode>_W<window>.parquet   (date, items="i1 i2 ...")
- artifacts/candidates/exp102_daily_trending/val_candidates_trending_W<bestW>_K<M>.parquet
- metrics/exp102_daily_trending.csv
- submissions/sub_exp102_trending_frozen_W<bestW>_k20.csv  (если вызван predict_submission)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd

from .base_exp import BaseExperiment
from .config import (
    PATHS, COL_USER, COL_ITEM, COL_DATE,
    TRENDING_WINDOWS, CAND_TOP_M_PER_USER, VAL_DAYS,
    WANDB_PROJECT, WANDB_GROUP, SEED, QUICK_MODE, QUICK_USERS,
)
from .artifacts import ensure_dir, save_df, log_artifact
from .trending import (
    build_val_day_toplists,
    candidates_from_day_toplists,
    evaluate_trending_candidates,
    coverage_from_frozen_trending,
    trending_global_from_last_train_window,
)


@dataclass
class Exp102Config:
    windows: Sequence[int] = tuple(TRENDING_WINDOWS)           # окна W
    day_top_k: int = 200                                       # сколько айтемов держать в дневном топе
    per_user_M: int = CAND_TOP_M_PER_USER                      # сколько кандидатов на пользователя
    mode: Literal["frozen_train", "rolling_online"] = "frozen_train"
    eval_M_list: Sequence[int] = (200, 500, 1000)              # recall@M
    coverage_ks: Sequence[int] = (20, 50, 100, 200, 500, 1000) # coverage@K (только frozen_train)
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED


@dataclass
class Exp102State:
    day_tops: Dict[int, Dict[int, List[int]]]      # W -> {date -> [item_ids]}
    cand_by_user: Dict[int, Dict[int, List[int]]]  # W -> {user -> [item_ids]}
    metrics_recall: pd.DataFrame                   # [window, M, recall]
    metrics_cov: Optional[pd.DataFrame]            # [window, K, coverage] (или None)
    best_window_by_recall: int
    out_dir: Path
    cand_dir: Path


class Exp102DailyTrending(BaseExperiment):
    def __init__(self, cfg: Optional[Exp102Config] = None):
        super().__init__(exp_name="exp102_daily_trending")
        self.cfg = cfg or Exp102Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.state: Optional[Exp102State] = None

    # ------------------- основной цикл эксперимента -------------------

    def fit(self, context: dict) -> Exp102State:
        """
        Строит дневные топы для каждого окна W и собирает per-user кандидатов на вал.
        context: ожидает train_df, val_df, split, val_truth, val_item_cnt.
        """
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth", "val_item_cnt"])
        rng = np.random.default_rng(self.cfg.seed)

        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame   = context["val_df"]
        split                  = context["split"]
        val_truth: Mapping[int, set] = context["val_truth"]
        val_item_cnt: pd.Series      = context["val_item_cnt"]

        # rolling_online нуждается в общем df_all (train + вал-история)
        df_all = pd.concat([train_df, val_df], axis=0, ignore_index=True)

        # быстрый режим: подсэмплировать пользователей валидации
        users_val = list(val_truth.keys())
        if self.cfg.quick_mode and self.cfg.quick_users and len(users_val) > self.cfg.quick_users:
            users_val = list(rng.choice(users_val, size=self.cfg.quick_users, replace=False))

        day_tops_by_W: Dict[int, Dict[int, List[int]]] = {}
        cand_by_user_by_W: Dict[int, Dict[int, List[int]]] = {}
        metrics_rows: List[pd.DataFrame] = []
        cov_rows: List[pd.DataFrame] = []

        for W in self.cfg.windows:
            # 1) day tops
            day_top = build_val_day_toplists(
                df_all=(train_df if self.cfg.mode == "frozen_train" else df_all),
                split=split, window=int(W), K=int(self.cfg.day_top_k),
                mode=self.cfg.mode
            )
            day_tops_by_W[W] = day_top
            self._save_day_tops(W, day_top)  # артефакт day_tops

            # 2) per-user candidates на вал
            #    (объединяем топы по вал-дням пользователя; ограничиваемся users_val при quick)
            val_df_subset = val_df[val_df[COL_USER].isin(users_val)] if users_val else val_df
            cand = candidates_from_day_tops(val_df_subset, day_top, K=self.cfg.per_user_M)
            cand_by_user_by_W[W] = cand

            # 3) recall@M
            rec_tbl = evaluate_trending_candidates(
                val_truth={u: val_truth[u] for u in users_val if u in val_truth} if users_val else val_truth,
                cand_by_user=cand,
                m_list=self.cfg.eval_M_list,
                averaging="micro",
            )
            rec_tbl["window"] = int(W)
            metrics_rows.append(rec_tbl)

            # 4) coverage@K (только для frozen_train)
            if self.cfg.mode == "frozen_train":
                cov_tbl = coverage_from_frozen_trending(train_df, val_item_cnt, split, window=W, ks=self.cfg.coverage_ks)
                cov_tbl["window"] = int(W)
                cov_rows.append(cov_tbl)

        metrics_recall = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
        metrics_cov = pd.concat(cov_rows, ignore_index=True) if cov_rows else None

        # лучший W по recall@max(M)
        if not metrics_recall.empty:
            Mmax = int(metrics_recall["M"].max())
            best_row = metrics_recall[metrics_recall["M"] == Mmax].sort_values("recall", ascending=False).iloc[0]
            best_W = int(best_row["window"])
        else:
            best_W = int(self.cfg.windows[0])

        # лог в W&B (таблицы и summary)
        self.wandb_log_table("exp102_recall", metrics_recall)
        if metrics_cov is not None and not metrics_cov.empty:
            self.wandb_log_table("exp102_coverage", metrics_cov)
        if not metrics_recall.empty:
            self.wandb_summary(**{f"exp102/best_window@M{Mmax}": best_W})

        self.state = Exp102State(
            day_tops=day_tops_by_W,
            cand_by_user=cand_by_user_by_W,
            metrics_recall=metrics_recall,
            metrics_cov=metrics_cov,
            best_window_by_recall=best_W,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
        )
        return self.state

    def evaluate(self, context: dict) -> pd.DataFrame:
        """Возвращает таблицу recall@M по окнам; логирует в W&B."""
        assert self.state is not None, "Run fit() first."
        self.wandb_log_table("exp102_recall", self.state.metrics_recall)
        if self.state.metrics_cov is not None and not self.state.metrics_cov.empty:
            self.wandb_log_table("exp102_coverage", self.state.metrics_cov)
        # сохранить метрики одним csv
        metrics_df = self._assemble_metrics_df(self.state.metrics_recall, self.state.metrics_cov)
        self.save_metrics_df(metrics_df, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")
        return self.state.metrics_recall

    def candidates(self, context: dict, users: Optional[Sequence[int]] = None, W: Optional[int] = None, M: Optional[int] = None) -> Dict[int, List[int]]:
        """Вернёт per-user кандидатов (из fit); по умолчанию для лучшего W."""
        assert self.state is not None, "Run fit() first."
        W = int(W or self.state.best_window_by_recall)
        cand_map = self.state.cand_by_user[W]
        if users is None:
            return cand_map
        out = {}
        limit = int(M or self.cfg.per_user_M)
        for u in users:
            out[int(u)] = cand_map.get(int(u), [])[:limit]
        return out

    def save(self, context: dict) -> Tuple[Path, Path]:
        """Сохраняет: метрики (csv) + кандидатов для лучшего W."""
        assert self.state is not None, "Run fit() first."
        # метрики
        metrics_df = self._assemble_metrics_df(self.state.metrics_recall, self.state.metrics_cov)
        metrics_path = self.save_metrics_df(metrics_df, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")
        # кандидаты
        best_W = self.state.best_window_by_recall
        cand_map = self.state.cand_by_user[best_W]
        cand_path = self.save_candidates_map(cand_map, filename=f"val_candidates_trending_W{best_W}_K{self.cfg.per_user_M}.parquet")
        return cand_path, metrics_path

    # ------------------- сабмит -------------------

    def predict_submission(self, context: dict, sample_df: pd.DataFrame, W: Optional[int] = None, k_top: int = 20) -> pd.DataFrame:
        """
        Делает baseline-сабмит только для mode='frozen_train': всем пользователям один frozen-топ.
        """
        assert self.state is not None, "Run fit() first."
        if self.cfg.mode != "frozen_train":
            raise ValueError("Submission is implemented only for mode='frozen_train'.")
        train_df: pd.DataFrame = context["train_df"]
        split                  = context["split"]
        W = int(W or self.state.best_window_by_recall)

        global_top = trending_global_from_last_train_window(train_df, split, window=W, K=max(k_top, 1000))

        tmp = sample_df.copy()
        tmp["rank"] = tmp.groupby(COL_USER).cumcount()
        rank2item = np.array(global_top[:k_top], dtype=np.int64)
        tmp[COL_ITEM] = rank2item[tmp["rank"].values]
        tmp = tmp.drop(columns=["rank"]).reset_index(drop=True)

        sub_dir = ensure_dir(PATHS.sub_dir)
        sub_path = sub_dir / f"sub_exp102_trending_frozen_W{W}_k{k_top}.csv"
        save_df(sub_path, tmp, index=False)
        log_artifact(sub_path, name=f"{self.exp_name}_submission_W{W}", type_="submission")
        return tmp

    # ------------------- утилиты -------------------

    def _save_day_tops(self, W: int, day_top: Mapping[int, Sequence[int]]) -> Path:
        """Сохраняет карту {date -> топ} в parquet (date, items="i1 i2 ...") и логирует как артефакт."""
        df = pd.DataFrame({
            COL_DATE: list(day_top.keys()),
            "items": [" ".join(map(str, day_top[d])) for d in day_top.keys()]
        }).sort_values(COL_DATE).reset_index(drop=True)
        path = self.out_dir / f"day_tops_mode-{self.cfg.mode}_W{W}.parquet"
        save_df(path, df, index=False)
        log_artifact(path, name=f"{self.exp_name}_daytops_W{W}", type_="dataset")
        return path

    @staticmethod
    def _assemble_metrics_df(metrics_recall: pd.DataFrame, metrics_cov: Optional[pd.DataFrame]) -> pd.DataFrame:
        rec = metrics_recall.copy()
        rec = rec.rename(columns={"recall": "metric_value"})
        rec["metric"] = "recall"
        rec = rec[["window", "M", "metric", "metric_value"]]
        if metrics_cov is not None and not metrics_cov.empty:
            cov = metrics_cov.copy().rename(columns={"coverage": "metric_value", "K": "M"})
            cov["metric"] = "coverage"
            cov = cov[["window", "M", "metric", "metric_value"]]
            return pd.concat([rec, cov], ignore_index=True)
        return rec
