# -*- coding: utf-8 -*-
"""
exp204_ranker_mlp — pointwise MLP (PyTorch) для ранжирования по features v1.

Контракт:
  - fit(ctx): строит фич-матрицы (builders.build_matrix), тренит MLP c ранней остановкой
  - evaluate(ctx): HR@K/NDCG@K на валидации
  - save(ctx): сохраняет state_dict, importances (псевдо: по |w| первого слоя), val_scores
  - predict_submission(ctx, sample_df, k_top): ранжирует test-пул и делает CSV

Зависимости: torch (PyTorch)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import glob
import os
import math
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.builders import build_matrix
from vseros_b.features.schema import apply_norm, load_schema, coerce_dtypes, FEATURES_VERSION
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.candidates.io import load_compact_map


# --------------------------------
# Defaults
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
class Exp204Config:
    features: Sequence[str] = tuple(DEFAULT_FEATURES)
    out_tag: str = FEATURES_VERSION
    max_neg_per_pos: int = 100
    seed: int = 42

    # Модель
    hidden_dims: Sequence[int] = (512, 256, 128)
    dropout: float = 0.1
    use_batchnorm: bool = True

    # Оптимизация
    epochs: int = 40
    batch_size: int = 4096
    lr: float = 3e-3
    weight_decay: float = 1e-5
    pos_weight: Optional[float] = None      # если None — авто (neg/pos)

    # Early stop по NDCG@20
    patience: int = 5
    eval_every: int = 1

    # Metrics@K
    k_list: Sequence[int] = (20, 50, 100)

    # GPU
    use_gpu: Optional[bool] = None  # None=auto, True/False — явно


# --------------------------------
# Torch Dataset
# --------------------------------

class RankDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32, copy=False)
        self.y = None if y is None else y.astype(np.float32, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# --------------------------------
# Model
# --------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.head(z).squeeze(-1)
        return logits


# --------------------------------
# Experiment
# --------------------------------

class Exp204RankerMLP(BaseExperiment):
    exp_name = "exp204_ranker_mlp"

    def __init__(self, cfg: Optional[Exp204Config] = None):
        self.cfg = cfg or Exp204Config()
        self.model: Optional[MLP] = None
        self.feature_cols: Optional[List[str]] = None
        self.norm_stats: Optional[Dict[str, dict]] = None
        self.train_mat: Optional[pd.DataFrame] = None
        self.val_mat: Optional[pd.DataFrame] = None

    # ---------- utilities ----------

    def _device(self) -> torch.device:
        use_gpu = self.cfg.use_gpu
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

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

    def _split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feats = [c for c in df.columns if c not in (COL_USER, COL_ITEM, "label", "group_id")]
        self.feature_cols = feats
        X = df[feats].astype("float32").values
        y = df["label"].astype("float32").values
        return X, y

    def _predict_scores_df(self, feats_df: pd.DataFrame) -> pd.DataFrame:
        """Utility: предсказывает score для feats_df с [user_id,item_id, feature_cols...]"""
        self.model.eval()
        device = self._device()
        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        X = feats_df[feat_cols].astype("float32").values
        ds = RankDataset(X)
        loader = DataLoader(ds, batch_size=131072, shuffle=False, num_workers=0, pin_memory=False)
        scores = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                logits = self.model(xb)
                probs = torch.sigmoid(logits)
                scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        out = feats_df[[COL_USER, COL_ITEM]].copy()
        out["score"] = scores.astype("float32")
        return out

    # ---------- core API ----------

    def fit(self, ctx: dict):
        if torch is None:
            raise ImportError("PyTorch is required for Exp204RankerMLP")

        self.ctx = ctx
        cfg = self.cfg

        # reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # 1) кандидаты
        fusion_map = self._load_fusion_map(mode="val")

        # 2) собрать матрицы/нормализации
        train_mat, val_mat, norm_stats = build_matrix(
            ctx,
            cand_map=fusion_map,
            feature_names=cfg.features,
            out_tag=cfg.out_tag,
            max_neg_per_pos=cfg.max_neg_per_pos,
            seed=cfg.seed,
            save_artifacts=True,
        )
        self.train_mat, self.val_mat, self.norm_stats = train_mat, val_mat, norm_stats

        X_tr, y_tr = self._split(train_mat)
        X_va, y_va = self._split(val_mat)

        # 3) модель
        in_dim = X_tr.shape[1]
        model = MLP(in_dim, cfg.hidden_dims, dropout=cfg.dropout, use_bn=cfg.use_batchnorm).to(self._device())
        self.model = model

        # 4) оптимизация
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # pos_weight
        if cfg.pos_weight is None:
            pos = float((y_tr > 0.5).sum())
            neg = float(len(y_tr) - pos)
            pos_weight = (neg / max(pos, 1.0))
        else:
            pos_weight = float(cfg.pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self._device()))

        # 5) loaders
        train_ds = RankDataset(X_tr, y_tr)
        val_ds = RankDataset(X_va, y_va)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=131072, shuffle=False, num_workers=0, pin_memory=False)

        # 6) train loop с ранней остановкой по NDCG@20
        best_metric = -1.0
        best_state = None
        patience = cfg.patience
        epochs_no_improve = 0

        wb = ctx.get("wandb_run")
        k_primary = 20

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self._device())
                yb = yb.to(self._device())
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n_batches += 1
            train_loss = total_loss / max(1, n_batches)

            # eval
            eval_now = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
            if eval_now:
                model.eval()
                with torch.no_grad():
                    # вал-логлосс
                    vloss_sum, vbatches = 0.0, 0
                    for xb, yb in val_loader:
                        xb = xb.to(self._device()); yb = yb.to(self._device())
                        logits = model(xb)
                        vloss = criterion(logits, yb)
                        vloss_sum += float(vloss.item()); vbatches += 1
                    val_loss = vloss_sum / max(1, vbatches)

                # вал-рэнкинг метрики
                # получим вероятности для всех вал-строк
                pred_scores = []
                model.eval()
                with torch.no_grad():
                    for xb, _ in val_loader:
                        xb = xb.to(self._device())
                        probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
                        pred_scores.append(probs)
                yhat = np.concatenate(pred_scores, axis=0).astype("float32")

                pred = self.val_mat[[COL_USER, COL_ITEM]].copy()
                pred["score"] = yhat
                pred_sorted = (
                    pred.sort_values([COL_USER, "score"], ascending=[True, False])
                        .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
                )
                val_truth: Dict[int, set] = ctx["val_truth"]
                metrics = {}
                for k in cfg.k_list:
                    metrics[f"HR@{k}"] = recall_at_m(pred_sorted, val_truth, m=k)
                    metrics[f"NDCG@{k}"] = ndcg_at_k(pred_sorted, val_truth, k=k)

                primary = metrics.get(f"NDCG@{k_primary}", 0.0)

                # W&B
                if wb is not None:
                    wb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics})

                # early stop
                improved = primary > best_metric + 1e-6
                if improved:
                    best_metric = primary
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

        # восстановим лучшую
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # финальный лог
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            wb.summary["feature_count"] = len(self.feature_cols or [])
            wb.summary["best_NDCG@20"] = best_metric

        return self

    def evaluate(self, ctx: dict):
        self.ctx = ctx
        if self.model is None:
            return {}

        # используем сохранённую self.val_mat (быстрее)
        if self.val_mat is None:
            # fallback: восстановим фичи
            cfg = self.cfg
            fusion_map = self._load_fusion_map(mode="val")
            _, val_mat, _ = build_matrix(
                ctx,
                cand_map=fusion_map,
                feature_names=cfg.features,
                out_tag=cfg.out_tag,
                max_neg_per_pos=cfg.max_neg_per_pos,
                seed=cfg.seed,
                save_artifacts=False,
            )
            self.val_mat = val_mat

        # предсказать и посчитать метрики
        model = self.model.eval()
        device = self._device()
        feat_cols = [c for c in self.val_mat.columns if c not in (COL_USER, COL_ITEM, "label", "group_id")]
        X = self.val_mat[feat_cols].astype("float32").values
        ds = RankDataset(X)
        loader = DataLoader(ds, batch_size=131072, shuffle=False, num_workers=0)

        scores = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb))
                scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)

        pred_df = self.val_mat[[COL_USER, COL_ITEM]].copy()
        pred_df["score"] = scores.astype("float32")

        pred_sorted = (
            pred_df.sort_values([COL_USER, "score"], ascending=[True, False])
                   .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        val_truth: Dict[int, set] = ctx["val_truth"]

        rows = []
        for k in self.cfg.k_list:
            rows.append({"k": k, "HR@k": recall_at_m(pred_sorted, val_truth, m=k),
                              "NDCG@k": ndcg_at_k(pred_sorted, val_truth, k=k)})

        wb = ctx.get("wandb_run")
        if wb is not None:
            for r in rows:
                wb.summary[f"HR@{r['k']}"] = r["HR@k"]
                wb.summary[f"NDCG@{r['k']}"] = r["NDCG@k"]

        # сохраним val_scores
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
        model_path = out_dir / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # простая «важность» фич: |веса| первого слоя (если есть)
        imp_df = None
        try:
            first_linear = None
            for m in self.model.backbone:
                if isinstance(m, nn.Linear):
                    first_linear = m
                    break
            if first_linear is None:
                first_linear = getattr(self.model, "head", None)
            if first_linear is not None and self.feature_cols is not None:
                w = first_linear.weight.detach().cpu().numpy()
                if w.ndim == 2:
                    # сводим по абсолютным значениям весов (max по выходным нейронам)
                    imp = np.abs(w).max(axis=0)
                else:
                    imp = np.abs(w)
                imp_df = pd.DataFrame({"feature": self.feature_cols, "abs_weight": imp})
                imp_df = imp_df.sort_values("abs_weight", ascending=False)
                save_df(out_dir / "feature_importance_linear.csv", imp_df, index=False)
        except Exception:
            pass

        # продублируем схему нормализации
        feat_dir = PATHS.artifact_dir / "features" / cfg.out_tag
        src_schema = feat_dir / "feature_schema.json"
        if src_schema.exists():
            (out_dir / "feature_schema.json").write_text(src_schema.read_text(encoding="utf-8"), encoding="utf-8")

        # W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(model_path))
                if imp_df is not None:
                    wb.save(str(out_dir / "feature_importance_linear.csv"))
                wb.save(str(out_dir / "val_scores.parquet"))
            except Exception:
                pass

        return {"model_path": str(model_path)}

    # ---------- submissions ----------

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: int = 20) -> pd.DataFrame:
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
        for _, (spec, builder) in reg.items():
            part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
            feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        feats_df = coerce_dtypes(feats_df, feat_cols)
        feats_df = apply_norm(feats_df, stats)

        # 3) предикт и top-K
        self.model.eval()
        device = self._device()
        X_test = feats_df[feat_cols].astype("float32").values
        ds = RankDataset(X_test)
        loader = DataLoader(ds, batch_size=131072, shuffle=False, num_workers=0)

        scores = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(device)
                probs = torch.sigmoid(self.model(xb))
                scores.append(probs.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)

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
