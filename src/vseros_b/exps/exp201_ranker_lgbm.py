# -*- coding: utf-8 -*-
"""
exp201_ranker_lgbm — LambdaMART-ранкер на features v1.

API (совместим с BaseExperiment):
  - exp_name = "exp201_ranker_lgbm"
  - fit(ctx): строит фич-матрицы (через builders), обучает LGBMRanker
  - evaluate(ctx): считает HR@K/NDCG@K на валиде, сохраняет val_scores
  - save(ctx): сохраняет модель, импортансы; пишет артефакты
  - predict_submission(ctx, sample_df, k_top): ранжирует test-пул и отдаёт сабмит

Зависимости: lightgbm
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    lgb = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.builders import build_matrix
from vseros_b.features.schema import apply_norm, load_schema, coerce_dtypes, FEATURES_VERSION
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.candidates.io import load_compact_map
import glob
import os
import time

# --------------------------------
# Config
# --------------------------------

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
class Exp201Config:
    features: Sequence[str] = tuple(DEFAULT_FEATURES)
    out_tag: str = FEATURES_VERSION
    max_neg_per_pos: int = 100    # 0/None = не сэмплить
    seed: int = 42

    # LightGBM params (базовые, дальше — через свипы)
    learning_rate: float = 0.05
    num_leaves: int = 127
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.85
    n_estimators: int = 1200
    max_depth: int = -1
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0

    # metrics@K
    k_list: Sequence[int] = (20, 50, 100)

# --------------------------------
# Experiment
# --------------------------------

class Exp201RankerLGBM(BaseExperiment):
    exp_name = "exp201_ranker_lgbm"

    def __init__(self, cfg: Optional[Exp201Config] = None):
        self.cfg = cfg or Exp201Config()
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_cols: Optional[List[str]] = None
        self.norm_stats: Optional[Dict[str, dict]] = None
        self.train_users: Optional[np.ndarray] = None

    # ---------- utilities ----------

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
        # 1) из ctx (если runner уже сформировал)
        ctx = self.ctx
        cand_map = ctx.get("fusion_map_" + mode) or ctx.get("cand_map_" + mode) or ctx.get("fusion_map")
        if cand_map:
            return cand_map
        # 2) из артефактов exp108
        base = PATHS.cand_dir / "exp108_recall_fusion"
        path = self._pick_latest_cand(base, mode=mode)
        if not path:
            raise FileNotFoundError(f"[{self.exp_name}] no fusion candidates found for mode={mode} in {base}")
        return load_compact_map(path)

    def _prepare_group(self, df: pd.DataFrame) -> np.ndarray:
        # LightGBM Ranker expects group sizes array
        return df.groupby("group_id").size().astype("int32").values

    def _split_Xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        feats = [c for c in df.columns if c not in (COL_USER, COL_ITEM, "label", "group_id")]
        X = df[feats]
        y = df["label"].astype("float32").values
        group = self._prepare_group(df)
        self.feature_cols = feats
        return X, y, group

    # ---------- core API ----------

    def fit(self, ctx: dict):
        if lgb is None:
            raise ImportError("lightgbm is required for Exp201RankerLGBM")

        self.ctx = ctx
        cfg = self.cfg

        # 1) загрузим fusion-кандидатов (val)
        fusion_map = self._load_fusion_map(mode="val")

        # 2) соберём матрицы и нормализации
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

        # 3) обучим LambdaMART
        params = dict(
            objective="lambdarank",
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_data_in_leaf=cfg.min_data_in_leaf,
            feature_fraction=cfg.feature_fraction,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            lambda_l1=cfg.lambda_l1,
            lambda_l2=cfg.lambda_l2,
            random_state=cfg.seed,
            n_jobs=-1,
            subsample=1.0,
            reg_sqrt=False,
            importance_type="gain",
        )
        model = lgb.LGBMRanker(**params)
        model.fit(
            X_tr, y_tr,
            group=g_tr.tolist(),
            eval_set=[(X_va, y_va)],
            eval_group=[g_va.tolist()],
            eval_at=[20],
            verbose=False,
        )
        self.model = model

        # 4) лог в W&B (если есть)
        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            wb.summary["feature_count"] = len(self.feature_cols or [])

        return self

    def evaluate(self, ctx: dict):
        self.ctx = ctx
        if self.model is None:
            return {}

        # прогон на валидации
        cfg = self.cfg
        fusion_map = self._load_fusion_map(mode="val")

        # пересоберём признаки в том же порядке (без лейблов)
        # используем уже сохранённую схему нормализации
        out_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        schema_path = out_dir / "feature_schema.json"
        stats = self.norm_stats or (load_schema(schema_path) if schema_path.exists() else {})

        # построим cand_df и посчитаем фичи только (без label)
        from vseros_b.features.builders import _apply_features as _fe_apply, _cand_map_to_df as _to_df  # reuse internals

        base = _to_df(fusion_map)
        autodiscover()
        res = FeatureResources(ctx=ctx)
        # последовательное применение фич (как в _apply_features), но без NaN-лишнего
        reg = select(list(cfg.features))
        feats_df = base.copy()
        for fname, (spec, builder) in reg.items():
            part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
            feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

        # приведение типов и нормализация по сохранённой схеме
        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        feats_df = coerce_dtypes(feats_df, feat_cols)
        feats_df = apply_norm(feats_df, stats)

        # скоры
        X = feats_df[[c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]]
        scores = self.model.predict(X, raw_score=False)

        pred_df = feats_df[[COL_USER, COL_ITEM]].copy()
        pred_df["score"] = scores.astype("float32")

        # метрики
        # построим топ-листы
        k_list = list(cfg.k_list)
        metrics_rows = []
        # свернём предсказания в user -> [items sorted]
        pred_sorted = (
            pred_df.sort_values([COL_USER, "score"], ascending=[True, False])
            .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        val_truth: Dict[int, set] = ctx["val_truth"]

        for k in k_list:
            hr = recall_at_m(pred_sorted, val_truth, m=k)
            nd = ndcg_at_k(pred_sorted, val_truth, k=k)
            metrics_rows.append({"k": k, "HR@k": hr, "NDCG@k": nd})

        # лог в W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            for row in metrics_rows:
                wb.summary[f"HR@{row['k']}"] = row["HR@k"]
                wb.summary[f"NDCG@{row['k']}"] = row["NDCG@k"]

        # сохраним val_scores
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_scores.parquet", pred_df)

        return {"metrics": metrics_rows}

    def save(self, ctx: dict):
        self.ctx = ctx
        if self.model is None:
            return {}
        cfg = self.cfg
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)

        # модель
        model_path = out_dir / "model.txt"
        try:
            self.model.booster_.save_model(str(model_path))
        except Exception:
            # fallback для новых API
            self.model._Booster.save_model(str(model_path))

        # импортансы
        if self.feature_cols:
            imp_gain = self.model.booster_.feature_importance(importance_type="gain")
            imp_split = self.model.booster_.feature_importance(importance_type="split")
            imp_df = pd.DataFrame({
                "feature": self.feature_cols,
                "gain": imp_gain,
                "split": imp_split,
            }).sort_values("gain", ascending=False)
            save_df(out_dir / "feature_importance.csv", imp_df, index=False)

        # схема нормализации (дублируем рядом с моделью для удобства)
        feat_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        src_schema = feat_dir / "feature_schema.json"
        if src_schema.exists():
            (out_dir / "feature_schema.json").write_text(src_schema.read_text(encoding="utf-8"), encoding="utf-8")

        # W&B: залогировать файлы
        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(model_path))
                wb.save(str(out_dir / "feature_importance.csv"))
                wb.save(str(out_dir / "val_scores.parquet"))
            except Exception:
                pass

        return {"model_path": str(model_path)}

    # ---------- submissions ----------

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: int = 20) -> pd.DataFrame:
        """
        Ранжируем test-пул и заполняем sample_submission.csv
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted")

        self.ctx = ctx
        cfg = self.cfg

        # 1) загрузим test-кандидатов fusion
        fusion_map_test = self._load_fusion_map(mode="test")

        # 2) посчитаем фичи на тесте и применим сохранённую схему нормализации
        out_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        schema_path = out_dir / "feature_schema.json"
        stats = self.norm_stats or (load_schema(schema_path) if schema_path.exists() else {})

        # применим те же фичи
        from vseros_b.features.builders import _cand_map_to_df as _to_df
        base = _to_df(fusion_map_test)

        autodiscover()
        res = FeatureResources(ctx=ctx)
        reg = select(list(cfg.features))
        feats_df = base.copy()
        for fname, (spec, builder) in reg.items():
            part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
            feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        feats_df = coerce_dtypes(feats_df, feat_cols)
        feats_df = apply_norm(feats_df, stats)

        # 3) предикт и топ-K
        X_test = feats_df[[c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]]
        scores = self.model.predict(X_test, raw_score=False)
        scored = feats_df[[COL_USER, COL_ITEM]].copy()
        scored["score"] = scores.astype("float32")

        # сортировка и взятие top-K
        topk = (
            scored.sort_values([COL_USER, "score"], ascending=[True, False])
            .groupby(COL_USER)[COL_ITEM].apply(lambda s: " ".join(map(str, s.head(k_top).tolist())))
            .reset_index().rename(columns={COL_ITEM: "items"})
        )

        # 4) собрать сабмит на базе sample_df
        sub = sample_df.copy()
        if "items" not in sub.columns:
            sub["items"] = ""
        sub = sub.drop(columns=["items"], errors="ignore").merge(topk, on="user_id", how="left")
        sub["items"] = sub["items"].fillna("")

        # 5) сохранение
        sub_dir = ensure_dir(PATHS.submission_dir)
        out_csv = sub_dir / f"sub_{self.exp_name}_k{k_top}.csv"
        sub.to_csv(out_csv, index=False)

        # W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(out_csv))
            except Exception:
                pass

        return sub
