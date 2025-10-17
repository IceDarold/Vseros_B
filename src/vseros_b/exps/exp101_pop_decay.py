# -*- coding: utf-8 -*-
"""
exp101_pop_decay.py
Stage 1 — Эксперимент 101: Global Popularity (Static vs Time-Decay, λ-sweep)

Артефакты:
- artifacts/exp101_pop_decay/pop_table.parquet          (все скоринги популярности)
- artifacts/exp101_pop_decay/coverage_curves.csv        (K, variant → coverage)
- metrics/exp101_pop_decay.csv                           (метрики по вариантам)
- submissions/sub_exp101_<variant>_<filter>.csv         (если вызван predict_submission)

W&B:
- лог таблиц coverage/metrics (если ран активен)
- лог файлов как Artifacts
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..artifacts import ensure_dir, log_artifact, log_table_df, save_df
from ..base_exp import BaseExperiment
from ..config import (
    COL_DATE,
    COL_ITEM,
    COL_USER,
    DECAY_REF,
    LAMBDA_LIST,
    PATHS,
    SEED,
)
from ..metrics import coverage_at_k, jaccard_topk, spearman_rank_corr
from ..candidates.pop_decay import (
    assemble_pop_table,
    best_variant_by_map,
    build_global_top,
    compute_pop_static,
    coverage_curves,
    evaluate_global_predictions,
    predict_per_user_global,
    sweep_pop_decay,
)


# ----------------------------- конфиг и состояние -----------------------------

@dataclass
class Exp101Config:
    lambdas: Sequence[float] = tuple(LAMBDA_LIST)
    decay_ref: Optional[int] = DECAY_REF   # если None → возьмём конец train
    eval_k: int = 20                      # mAP@K, Hit@K, NDCG@K
    coverage_ks: Sequence[int] = (20, 50, 100, 200, 500, 1000)
    eval_filter_seen_in_train: bool = True
    global_K_pool: int = 2000             # длина глобального пула для сабмита/фильтрации
    seed: int = SEED


@dataclass
class Exp101State:
    pop_static: pd.Series
    pop_decay_dict: Dict[float, pd.Series]
    metrics_df: pd.DataFrame
    coverage_df: pd.DataFrame
    best_variant: str
    out_dir: Path


# ----------------------------- реализация эксперимента -----------------------------

class Exp101PopDecay(BaseExperiment):
    def __init__(self, cfg: Optional[Exp101Config] = None):
        super().__init__(exp_name="exp101_pop_decay")
        self.cfg = cfg or Exp101Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.coverage_path = self.out_dir / "coverage_curves.csv"
        self.pop_table_path = self.out_dir / "pop_table.parquet"
        self.state: Optional[Exp101State] = None

    # ---- обязательные методы ----

    def fit(self, context: dict) -> Exp101State:
        """
        Строим глобальные скоринги популярности по TRAIN (статик + λ-свип decay).
        """
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth", "val_item_cnt"])
        train_df: pd.DataFrame = context["train_df"]
        split                  = context["split"]
        val_item_cnt: pd.Series = context["val_item_cnt"]

        ref = self.cfg.decay_ref
        if ref is None:
            ref = split.train_end  # конец train

        # 1) скоринги
        pop_static = compute_pop_static(train_df, ensure_unique_triplets=False)
        pop_decay_dict = sweep_pop_decay(train_df, self.cfg.lambdas, ref_day=ref, ensure_unique_triplets=False)

        # 2) coverage@K для каждого варианта
        cov_rows = []
        cov_s = coverage_curves(pop_static, val_item_cnt, ks=self.cfg.coverage_ks)
        cov_s["variant"] = "static"
        cov_rows.append(cov_s)

        for lam, s in pop_decay_dict.items():
            cov_d = coverage_curves(s, val_item_cnt, ks=self.cfg.coverage_ks)
            cov_d["variant"] = f"decay@{lam}"
            cov_rows.append(cov_d)

        coverage_all = pd.concat(cov_rows, ignore_index=True)

        # 3) сохраним общий pop_table
        pop_tbl = assemble_pop_table(pop_static, pop_decay_dict)
        save_df(self.pop_table_path, pop_tbl, index=True)
        log_artifact(self.pop_table_path, name=f"{self.exp_name}_pop_table", type_="dataset")

        # 4) сохраним coverage
        save_df(self.coverage_path, coverage_all, index=False)
        log_artifact(self.coverage_path, name=f"{self.exp_name}_coverage", type_="dataset")
        self.wandb_log_table("exp101_coverage", coverage_all)

        # наполним state
        self.state = Exp101State(
            pop_static=pop_static,
            pop_decay_dict=pop_decay_dict,
            metrics_df=pd.DataFrame(),
            coverage_df=coverage_all,
            best_variant="static",
            out_dir=self.out_dir,
        )
        return self.state

    def evaluate(self, context: dict) -> pd.DataFrame:
        """
        Считает ранжировочные метрики на валидации для каждого варианта глобального топа.
        """
        assert self.state is not None, "Call fit() first."
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])

        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame = context["val_df"]
        split = context["split"]
        val_truth: Mapping[int, set] = context["val_truth"]

        users_val = list(val_truth.keys())
        k_eval = self.cfg.eval_k

        # подготовим карту seen (если включена оценка с фильтром)
        seen_map = None
        if self.cfg.eval_filter_seen_in_train:
            seen_map = train_df.groupby(COL_USER)[COL_ITEM].apply(lambda s: set(map(int, s.values))).to_dict()

        rows = []

        # STATIC
        global_static = build_global_top(self.state.pop_static, K=self.cfg.global_K_pool)
        for filter_mode, smap in (("no_filter", None), ("filter_seen", seen_map)):
            preds = predict_per_user_global(users_val, global_static, k=k_eval, seen_map=smap)
            res = evaluate_global_predictions(val_truth, preds, k_eval=k_eval, variant_name="static")
            rows.append({
                "variant": "static", "filter": filter_mode,
                "map@20": res.map_at_k, "hit@20": res.hit_at_k, "ndcg@20": res.ndcg_at_k
            })

        # DECAY — каждый λ
        for lam, s in self.state.pop_decay_dict.items():
            global_top = build_global_top(s, K=self.cfg.global_K_pool)
            for filter_mode, smap in (("no_filter", None), ("filter_seen", seen_map)):
                preds = predict_per_user_global(users_val, global_top, k=k_eval, seen_map=smap)
                res = evaluate_global_predictions(val_truth, preds, k_eval=k_eval, variant_name=f"decay@{lam}")
                rows.append({
                    "variant": f"decay@{lam}", "filter": filter_mode,
                    "map@20": res.map_at_k, "hit@20": res.hit_at_k, "ndcg@20": res.ndcg_at_k
                })

        metrics_df = pd.DataFrame(rows).sort_values(["filter", "map@20", "ndcg@20"], ascending=[True, False, False]).reset_index(drop=True)
        self.state.metrics_df = metrics_df

        # Логи и сохранение
        self.wandb_log_table("exp101_metrics", metrics_df)
        self.save_metrics_df(metrics_df, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")

        # Найдём лучший вариант по mAP@20 (по no_filter — чаще главный)
        best_no_filter = metrics_df[metrics_df["filter"] == "no_filter"]
        if not best_no_filter.empty:
            self.state.best_variant = best_variant_by_map(best_no_filter, variant_col="variant", map_col="map@20") or "static"
        else:
            self.state.best_variant = best_variant_by_map(metrics_df, variant_col="variant", map_col="map@20") or "static"

        # Доп: дрейф популярности (train vs val) — для отчёта
        train_item_cnt = train_df.groupby(COL_ITEM)[COL_DATE].count()
        val_item_cnt = val_df.groupby(COL_ITEM)[COL_DATE].count()
        try:
            sp = spearman_rank_corr(train_item_cnt, val_item_cnt)
            jacc_tbl = jaccard_topk(train_item_cnt, val_item_cnt, ks=self.cfg.coverage_ks)
            self.wandb_log_table("exp101_drift_jaccard", jacc_tbl)
            self.wandb_summary(**{"exp101/best_variant": self.state.best_variant, "exp101/spearman_train_val": float(sp)})
        except Exception:
            pass

        return metrics_df

    def save(self, context: dict) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Уже сохраняли coverage и pop_table в fit(); здесь ничего доп. не требуется.
        Вернём пути на всякий случай.
        """
        return self.pop_table_path, self.coverage_path

    # ---- сабмит ----

    def predict_submission(self, context: dict, sample_df: pd.DataFrame, k_top: int = 20, use_filter_seen: bool = False) -> pd.DataFrame:
        """
        Делает сабмит для лучшего варианта (по mAP@20). По умолчанию без фильтра "seen".
        """
        assert self.state is not None and not self.state.metrics_df.empty, "Run fit() and evaluate() first."

        best = self.state.best_variant or "static"
        if best == "static":
            score_series = self.state.pop_static
        else:
            # формат 'decay@<λ>'
            lam = float(str(best).split("@")[1])
            score_series = self.state.pop_decay_dict[lam]

        global_top = build_global_top(score_series, K=max(k_top, self.cfg.global_K_pool))

        seen_map = None
        if use_filter_seen:
            train_df: pd.DataFrame = context["train_df"]
            seen_map = train_df.groupby(COL_USER)[COL_ITEM].apply(lambda s: set(map(int, s.values))).to_dict()

        # заполняем sample
        tmp = sample_df.copy()
        tmp["rank"] = tmp.groupby(COL_USER).cumcount()
        rank2item = np.array(global_top[:k_top], dtype=np.int64)

        if not use_filter_seen:
            tmp[COL_ITEM] = rank2item[tmp["rank"].values]
        else:
            # медленнее: фильтруем per-user
            users = tmp[COL_USER].unique()
            # построим список top-k на пользователя
            top_by_user: Dict[int, List[int]] = predict_per_user_global(users, global_top, k=k_top, seen_map=seen_map)
            tmp[COL_ITEM] = [top_by_user[int(u)][int(r)] for u, r in zip(tmp[COL_USER].values, tmp["rank"].values)]

        tmp = tmp.drop(columns=["rank"]).reset_index(drop=True)

        # сохраняем сабмит
        suffix = "filterseen" if use_filter_seen else "nofilter"
        sub_dir = ensure_dir(PATHS.sub_dir)
        sub_path = sub_dir / f"sub_exp101_{best}_{suffix}.csv"
        save_df(sub_path, tmp, index=False)
        log_artifact(sub_path, name=f"{self.exp_name}_submission_{best}_{suffix}", type_="submission")

        return tmp
