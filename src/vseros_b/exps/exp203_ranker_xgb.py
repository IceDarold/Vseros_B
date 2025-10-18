# -*- coding: utf-8 -*-
"""
exp203_ranker_xgb — XGBoost Ranker (pairwise) на features v1.

Совместим с BaseExperiment:
  - fit/evaluate/save/predict_submission

Зависимости: xgboost>=1.6 (XGBRanker)
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
    from xgboost import XGBRanker
except Exception:
    XGBRanker = None

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
class Exp203Config:
    features: Sequence[str] = tuple(DEFAULT_FEATURES)
    out_tag: str = FEATURES_VERSION
    max_neg_per_pos: int = 100
    seed: int = 42

    # XGB params (базовые — дальше через свипы)
    n_estimators: int = 1200
    learning_rate: float = 0.05
    max_depth: int = 8
    min_child_weight: float = 20.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    tree_method: str = "hist"

    # metrics@K
    k_list: Sequence[int] = (20, 50, 100)


class Exp203RankerXGB(BaseExperiment):
    exp_name = "exp203_ranker_xgb"

    def __init__(self, cfg: Optional[Exp203Config] = None):
        self.cfg = cfg or Exp203Config()
        self.model: Optional[XGBRanker] = None
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

    def _group_sizes(self, df: pd.DataFrame) -> np.ndarray:
        return df.groupby("group_id").size().astype("int32").values

    def _split_Xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        feats = [c for c in df.columns if c not in (COL_USER, COL_ITEM, "label", "group_id")]
        X = df[feats]
        y = df["label"].astype("float32").values
        group = self._group_sizes(df)
        self.feature_cols = feats
        return X, y, group

    # ----------------- API -----------------

    def fit(self, ctx: dict):
        if XGBRanker is None:
            raise ImportError("xgboost>=1.6 is required for Exp203RankerXGB")

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

        X_tr, y_tr, g_tr = self._split_Xy(train_mat)
        X_va, y_va, g_va = self._split_Xy(val_mat)

        params = dict(
            objective="rank:pairwise",
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            tree_method=cfg.tree_method,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        model = XGBRanker(**params)
        model.fit(
            X_tr, y_tr,
            group=g_tr.tolist(),
            eval_set=[(X_va, y_va)],
            eval_group=[g_va.tolist()],
            verbose=False,
        )
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

        # фичи на валидации
        out_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        schema_path = out_dir / "feature_schema.json"
        stats = self.norm_stats or (load_schema(schema_path) if schema_path.exists() else {})

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

        pred_sorted = (
            pred_df.sort_values([COL_USER, "score"], ascending=[True, False])
                  .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        val_truth: Dict[int, set] = ctx["val_truth"]
        rows = []
        for k in cfg.k_list:
            rows.append({"k": k, "HR@k": recall_at_m(pred_sorted, val_truth, m=k),
                              "NDCG@k": ndcg_at_k(pred_sorted, val_truth, k=k)})

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

        # модель
        model_path = out_dir / "model.json"
        self.model.get_booster().save_model(str(model_path))

        # импортансы (gain)
        if self.feature_cols:
            booster = self.model.get_booster()
            imp_gain = booster.get_score(importance_type="gain")
            # превращаем в aligned таблицу
            imp_df = pd.DataFrame({"feature": list(imp_gain.keys()), "gain": list(imp_gain.values())})
            # дополним отсутствующие нулями
            missing = [f for f in self.feature_cols if f not in imp_gain]
            if missing:
                imp_df = pd.concat([imp_df, pd.DataFrame({"feature": missing, "gain": [0.0]*len(missing)})], axis=0)
            imp_df = imp_df.sort_values("gain", ascending=False)
            save_df(out_dir / "feature_importance.csv", imp_df, index=False)

        # схема нормализации рядом
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
