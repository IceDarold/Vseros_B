# -*- coding: utf-8 -*-
"""
exp202_ranker_catboost — YetiRank (CatBoost) на features v1.

Совместим с BaseExperiment:
  - fit(ctx)        : собирает матрицы, обучает CatBoostRanker
  - evaluate(ctx)   : HR@K/NDCG@K на валидации
  - save(ctx)       : сохраняет модель/импортансы/val_scores
  - predict_submission(ctx, sample_df, k_top)

Зависимости: catboost
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import json
import glob
import os

try:
    from catboost import CatBoostRanker, Pool
except Exception:
    CatBoostRanker = None
    Pool = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.builders import build_matrix
from vseros_b.features.schema import apply_norm, load_schema, coerce_dtypes, FEATURES_VERSION
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.candidates.io import load_compact_map


DEFAULT_FEATURES: List[str] = [
    "sources_basic",
    "pop_basic",
    "calendar_basic",
    "user_basic",
    "i2v_sim",
    "covis_local",
    "markov_basic",
    "regularizers_basic",
    "interact_strong",
    "diag_basic",
]


@dataclass
class Exp202Config:
    features: Sequence[str] = tuple(DEFAULT_FEATURES)
    out_tag: str = FEATURES_VERSION
    max_neg_per_pos: int = 100
    seed: int = 42

    # CatBoost params (базовые — дальше через свипы)
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 8
    l2_leaf_reg: float = 3.0
    loss_function: str = "YetiRank"  # или "YetiRankPairwise"
    random_strength: float = 1.0
    border_count: int = 254
    # metrics@K
    k_list: Sequence[int] = (20, 50, 100)


class Exp202RankerCatBoost(BaseExperiment):
    exp_name = "exp202_ranker_catboost"

    def __init__(self, cfg: Optional[Exp202Config] = None):
        self.cfg = cfg or Exp202Config()
        self.model: Optional[CatBoostRanker] = None
        self.feature_cols: Optional[List[str]] = None
        self.norm_stats: Optional[Dict[str, dict]] = None

    # ----------------- utils -----------------

    def _pick_latest_cand(self, exp_dir: Path, mode: str) -> Optional[Path]:
        if not exp_dir.exists():
            return None
        patt = "val_candidates*.parquet" if mode == "val" else "test_candidates*.parquet"
        files = glob.glob(str(exp_dir / patt))
        if not files:
            return None
        files = sorted(files, key=lambda p: os.path.getsize(p))
        return Path(files[-1])

    def _load_fusion_map(self, mode: str = "val") -> Dict[int, List[int]]:
        ctx = self.ctx
        cand_map = ctx.get("fusion_map_" + mode) or ctx.get("cand_map_" + mode) or ctx.get("fusion_map")
        if cand_map:
            return cand_map
        base = PATHS.cand_dir / "exp108_recall_fusion"
        p = self._pick_latest_cand(base, mode=mode)
        if not p:
            raise FileNotFoundError(f"[{self.exp_name}] no fusion candidates for mode={mode} in {base}")
        return load_compact_map(p)

    def _split_Xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        feats = [c for c in df.columns if c not in (COL_USER, COL_ITEM, "label", "group_id")]
        X = df[feats]
        y = df["label"].astype("float32").values
        group_id = df["group_id"].astype("int64").values  # для CatBoost нужен query/group id для каждой строки
        self.feature_cols = feats
        return X, y, group_id

    # ----------------- API -----------------

    def fit(self, ctx: dict):
        if CatBoostRanker is None or Pool is None:
            raise ImportError("catboost is required for Exp202RankerCatBoost")

        self.ctx = ctx
        cfg = self.cfg

        # 1) кандидаты
        fusion_map = self._load_fusion_map(mode="val")

        # 2) фичи/нормализации
        train_mat, val_mat, norm_stats = build_matrix(
            ctx,
            cand_map=fusion_map,
            feature_names=cfg.features,
            out_tag=cfg.out_tag,
            max_neg_per_pos=cfg.max_neg_per_pos,
            seed=cfg.seed,
            save_artifacts=True,
        )
        self.norm_stats = norm_stats

        X_tr, y_tr, gid_tr = self._split_Xy(train_mat)
        X_va, y_va, gid_va = self._split_Xy(val_mat)

        train_pool = Pool(
            data=X_tr,
            label=y_tr,
            group_id=gid_tr,
        )
        val_pool = Pool(
            data=X_va,
            label=y_va,
            group_id=gid_va,
        )

        params = dict(
            loss_function=cfg.loss_function,
            iterations=cfg.iterations,
            learning_rate=cfg.learning_rate,
            depth=cfg.depth,
            l2_leaf_reg=cfg.l2_leaf_reg,
            random_seed=cfg.seed,
            random_strength=cfg.random_strength,
            border_count=cfg.border_count,
            od_type="Iter",
            od_wait=100,
            task_type="CPU",
            verbose=False,
        )
        model = CatBoostRanker(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=False)
        self.model = model

        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            wb.summary["feature_count"] = len(self.feature_cols or [])

        return self

    def evaluate(self, ctx: dict):
        self.ctx = ctx
        if self.model is None:
            return {}

        cfg = self.cfg
        fusion_map = self._load_fusion_map(mode="val")

        # пересчёт фичей на валидации (без лейблов), нормализация по сохранённой схеме
        out_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        schema_path = out_dir / "feature_schema.json"
        stats = self.norm_stats or (load_schema(schema_path) if schema_path.exists() else {})

        # build features
        from vseros_b.features.builders import _cand_map_to_df as _to_df
        base = _to_df(fusion_map)
        autodiscover()
        res = FeatureResources(ctx=ctx)
        reg = select(list(cfg.features))
        feats_df = base.copy()
        for _, (spec, builder) in reg.items():
            part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
            feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        feats_df = coerce_dtypes(feats_df, feat_cols)
        feats_df = apply_norm(feats_df, stats)

        scores = self.model.predict(feats_df[feat_cols])
        pred_df = feats_df[[COL_USER, COL_ITEM]].copy()
        pred_df["score"] = scores.astype("float32")

        # метрики
        pred_sorted = (
            pred_df.sort_values([COL_USER, "score"], ascending=[True, False])
                  .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        val_truth: Dict[int, set] = ctx["val_truth"]
        rows = []
        for k in cfg.k_list:
            rows.append({"k": k, "HR@k": recall_at_m(pred_sorted, val_truth, m=k),
                              "NDCG@k": ndcg_at_k(pred_sorted, val_truth, k=k)})

        # лог + сохранения
        wb = ctx.get("wandb_run")
        if wb is not None:
            for r in rows:
                wb.summary[f"HR@{r['k']}"] = r["HR@k"]
                wb.summary[f"NDCG@{r['k']}"] = r["NDCG@k"]

        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_scores.parquet", pred_df)

        return {"metrics": rows}

    def save(self, ctx: dict):
        self.ctx = ctx
        if self.model is None:
            return {}
        cfg = self.cfg
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)

        model_path = out_dir / "model.cbm"
        self.model.save_model(str(model_path))

        # импортансы
        if self.feature_cols:
            imp = self.model.get_feature_importance(type="FeatureImportance")
            imp_df = pd.DataFrame({"feature": self.feature_cols, "gain": imp}).sort_values("gain", ascending=False)
            save_df(out_dir / "feature_importance.csv", imp_df, index=False)

        # продублируем схему нормализации
        feat_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        src_schema = feat_dir / "feature_schema.json"
        if src_schema.exists():
            (out_dir / "feature_schema.json").write_text(src_schema.read_text(encoding="utf-8"), encoding="utf-8")

        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(model_path))
                wb.save(str(out_dir / "feature_importance.csv"))
                wb.save(str(out_dir / "val_scores.parquet"))
            except Exception:
                pass

        return {"model_path": str(model_path)}

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not fitted")

        self.ctx = ctx
        cfg = self.cfg

        fusion_map_test = self._load_fusion_map(mode="test")

        # фичи для теста
        out_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        schema_path = out_dir / "feature_schema.json"
        stats = self.norm_stats or (load_schema(schema_path) if schema_path.exists() else {})

        from vseros_b.features.builders import _cand_map_to_df as _to_df
        base = _to_df(fusion_map_test)
        autodiscover()
        res = FeatureResources(ctx=ctx)
        reg = select(list(cfg.features))
        feats_df = base.copy()
        for _, (spec, builder) in reg.items():
            part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
            feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        feats_df = coerce_dtypes(feats_df, feat_cols)
        feats_df = apply_norm(feats_df, stats)

        scores = self.model.predict(feats_df[feat_cols])
        scored = feats_df[[COL_USER, COL_ITEM]].copy()
        scored["score"] = scores.astype("float32")

        topk = (
            scored.sort_values([COL_USER, "score"], ascending=[True, False])
                  .groupby(COL_USER)[COL_ITEM].apply(lambda s: " ".join(map(str, s.head(k_top).tolist())))
                  .reset_index().rename(columns={COL_ITEM: "items"})
        )

        sub = sample_df.copy()
        if "items" not in sub.columns:
            sub["items"] = ""
        sub = sub.drop(columns=["items"], errors="ignore").merge(topk, on="user_id", how="left")
        sub["items"] = sub["items"].fillna("")

        sub_dir = ensure_dir(PATHS.submission_dir)
        out_csv = sub_dir / f"sub_{self.exp_name}_k{k_top}.csv"
        sub.to_csv(out_csv, index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(out_csv))
            except Exception:
                pass

        return sub
