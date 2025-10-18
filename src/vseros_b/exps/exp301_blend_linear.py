# -*- coding: utf-8 -*-
"""
exp301_blend_linear — простой линейный бленд нескольких источников скоров.

Что делает:
  • Берёт кандидатов (обычно из exp108_recall_fusion).
  • Подтягивает score-таблицы из base_sources:
      - если есть artifacts/<src>/val_scores.parquet — используем;
      - если src ∈ {exp201,exp202,exp203} и артефакта нет — пересчитываем по фичам.
  • Нормализует каждый источник per-user (rank|minmax|zscore|none).
  • Подбирает веса на валидации (least-squares → clip ≥0 → normalize sum=1).
  • Считает blended-скор = Σ w_s * s_norm_src.
  • Сохраняет:
      artifacts/models/exp301_blend_linear/val_scores.parquet
      artifacts/models/exp301_blend_linear/val_metrics.csv
      artifacts/models/exp301_blend_linear/blend_weights.json
     (опц.) test_scores.parquet, если доступны тестовые кандидаты.
  • predict_submission(sample_df): формирует сабмит по blended-скорам.

Зависимости: numpy, pandas (+ опц. lightgbm/catboost/xgboost только если нужно пересчитать 201/202/203).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import glob
import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd

# optional model libs for 201/202/203 fallback
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostRanker, Pool as CatPool
except Exception:
    CatBoostRanker, CatPool = None, None
try:
    from xgboost import XGBRanker
except Exception:
    XGBRanker = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM, COL_DATE
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.features.schema import FEATURES_VERSION, load_schema, apply_norm, coerce_dtypes
from vseros_b.candidates.io import load_compact_map, save_compact_map


# =============================================================================
# Конфиг
# =============================================================================

DEFAULT_FEATURES_FOR_BASES: Sequence[str] = (
    "sources_basic", "pop_basic", "calendar_basic", "user_basic",
    "i2v_sim", "covis_local", "markov_basic", "regularizers_basic",
    "interact_strong", "diag_basic",
)

@dataclass
class Exp301Config:
    # кандидаты для валидации/теста
    candidate_source: str = "exp108_recall_fusion"
    topk_limit_per_user: int = 0  # 0 = не ограничивать; иначе обрезать кандидатов после мерджа для скорости

    # источники для бленда (ожидаются их score-таблицы)
    base_sources: Sequence[str] = (
        "exp201_ranker_lgbm",
        "exp202_ranker_catboost",
        "exp203_ranker_xgb",
        "exp106_lightgcn_import",
        "exp107_ppr",
        # можешь добавить "exp102_daily_trending" и т.п., если они сохраняют val_scores/test_scores
    )

    # если нужно пересчитывать 201/202/203 — набор фич
    features_for_sources: Sequence[str] = DEFAULT_FEATURES_FOR_BASES
    features_tag: str = FEATURES_VERSION

    # нормализация каждого источника per-user
    norm_mode: str = "rank"  # 'rank' | 'minmax_user' | 'zscore_user' | 'none'

    # подстройка весов
    non_negative: bool = True     # клип к >=0
    normalize_weights: bool = True  # нормировать сумму к 1
    with_intercept: bool = False  # добавлять свободный член

    # метрики/сабмит
    k_list: Sequence[int] = (20, 50, 100)
    k_top_submit: int = 20

    seed: int = 42


# =============================================================================
# Вспомогалки: кандидаты и скоры
# =============================================================================

def _pick_latest_cand(exp_dir: Path, mode: str) -> Optional[Path]:
    if not exp_dir.exists(): return None
    patt = "val_candidates*.parquet" if mode == "val" else "test_candidates*.parquet"
    files = glob.glob(str(exp_dir / patt))
    if not files: return None
    files = sorted(files, key=lambda p: os.path.getsize(p))
    return Path(files[-1])

def _load_candidates(ctx: dict, exp_name: str, mode: str) -> Dict[int, List[int]]:
    m = ctx.get("fusion_map_" + mode) or ctx.get("cand_map_" + mode) or ctx.get("fusion_map")
    if m: return m
    base = PATHS.cand_dir / exp_name
    p = _pick_latest_cand(base, mode=mode)
    if not p:
        raise FileNotFoundError(f"[exp301] no candidates for mode={mode} in {base}")
    return load_compact_map(p)

def _artifact_model_dir(exp_name: str) -> Path:
    return PATHS.artifact_dir / "models" / exp_name

def _load_scores_artifact(exp_name: str, mode: str) -> Optional[pd.DataFrame]:
    mdir = _artifact_model_dir(exp_name)
    p = mdir / ("val_scores.parquet" if mode == "val" else "test_scores.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        cols = set(df.columns)
        if "score" in cols:
            return df[[COL_USER, COL_ITEM, "score"]].copy()
        if "blended" in cols:
            # дружелюбно поддерживаем старое имя
            return df[[COL_USER, COL_ITEM]].assign(score=df["blended"].astype("float32"))
    return None

def _cand_map_to_df(map_ui: Dict[int, List[int]]) -> pd.DataFrame:
    rows = []
    for u, items in map_ui.items():
        for it in items:
            rows.append((int(u), int(it)))
    return pd.DataFrame(rows, columns=[COL_USER, COL_ITEM])

# ---------------- пересчёт 201/202/203 при отсутствии артефактов ----------------

def _build_features(ctx: dict, user2items: Dict[int, List[int]], feature_names: Sequence[str]) -> pd.DataFrame:
    base = _cand_map_to_df(user2items)
    autodiscover()
    res = FeatureResources(ctx=ctx)
    reg = select(list(feature_names))
    feats_df = base.copy()
    for _, (spec, builder) in reg.items():
        part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
        feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)
    for c in feats_df.columns:
        if c in (COL_USER, COL_ITEM): continue
        if pd.api.types.is_float_dtype(feats_df[c]): feats_df[c] = feats_df[c].fillna(0.0).astype("float32")
        else: feats_df[c] = feats_df[c].fillna(0)
    return feats_df

def _load_schema_from_model_dir(model_dir: Path) -> Dict[str, dict]:
    sp = model_dir / "feature_schema.json"
    if not sp.exists():
        feat_dir = PATHS.artifact_dir / "features" / FEATURES_VERSION
        sp = feat_dir / "feature_schema.json"
    return load_schema(sp) if sp.exists() else {}

def _predict_exp201(ctx, mode, cand_map, feats):
    if lgb is None: return None
    mdir = _artifact_model_dir("exp201_ranker_lgbm")
    mp = mdir / "model.txt"
    if not mp.exists(): return None
    booster = lgb.Booster(model_file=str(mp))
    feat_names = booster.feature_name()
    stats = _load_schema_from_model_dir(mdir)
    df = _build_features(ctx, cand_map, feats)
    df = coerce_dtypes(df, [c for c in df.columns if c not in (COL_USER, COL_ITEM)])
    df = apply_norm(df, stats)
    X = pd.DataFrame(index=df.index)
    for f in feat_names:
        X[f] = df[f] if f in df.columns else 0.0
    s = booster.predict(X, raw_score=False)
    return df[[COL_USER, COL_ITEM]].assign(score=s.astype("float32"))

def _predict_exp202(ctx, mode, cand_map, feats):
    if CatBoostRanker is None: return None
    mdir = _artifact_model_dir("exp202_ranker_catboost")
    mp = mdir / "model.cbm"
    if not mp.exists(): return None
    model = CatBoostRanker(); model.load_model(str(mp))
    feat_names = list(getattr(model, "feature_names_", []))
    stats = _load_schema_from_model_dir(mdir)
    df = _build_features(ctx, cand_map, feats)
    df = coerce_dtypes(df, [c for c in df.columns if c not in (COL_USER, COL_ITEM)])
    df = apply_norm(df, stats)
    if feat_names and all(f in df.columns for f in feat_names):
        X = df[feat_names]
    else:
        feat_cols = [c for c in df.columns if c not in (COL_USER, COL_ITEM)]
        X = df[sorted(feat_cols)]
    pool = CatPool(data=X)
    s = model.predict(pool)
    return df[[COL_USER, COL_ITEM]].assign(score=np.asarray(s, dtype="float32"))

def _predict_exp203(ctx, mode, cand_map, feats):
    if XGBRanker is None: return None
    mdir = _artifact_model_dir("exp203_ranker_xgb")
    mp = mdir / "model.json"
    if not mp.exists(): return None
    model = XGBRanker(); model.load_model(str(mp))
    feat_names = list(model.get_booster().feature_names) if model.get_booster() is not None else None
    stats = _load_schema_from_model_dir(mdir)
    df = _build_features(ctx, cand_map, feats)
    df = coerce_dtypes(df, [c for c in df.columns if c not in (COL_USER, COL_ITEM)])
    df = apply_norm(df, stats)
    if feat_names and all(f in df.columns for f in feat_names):
        X = df[feat_names]
    else:
        feat_cols = [c for c in df.columns if c not in (COL_USER, COL_ITEM)]
        X = df[sorted(feat_cols)]
    s = model.predict(X)
    return df[[COL_USER, COL_ITEM]].assign(score=np.asarray(s, dtype="float32"))

def _get_source_scores(ctx: dict, exp_name: str, mode: str,
                       cand_map: Dict[int, List[int]],
                       feats_for_source: Sequence[str]) -> Optional[pd.DataFrame]:
    df = _load_scores_artifact(exp_name, mode)
    if df is not None:
        return df
    if exp_name == "exp201_ranker_lgbm":
        return _predict_exp201(ctx, mode, cand_map, feats_for_source)
    if exp_name == "exp202_ranker_catboost":
        return _predict_exp202(ctx, mode, cand_map, feats_for_source)
    if exp_name == "exp203_ranker_xgb":
        return _predict_exp203(ctx, mode, cand_map, feats_for_source)
    # для других источников требуем parquet
    return None


# =============================================================================
# Нормализация per-user
# =============================================================================

def _normalize_userwise(df: pd.DataFrame, mode: str, score_col: str = "score") -> pd.Series:
    s = df[score_col].astype("float32").values
    if mode == "none":
        return df[score_col].astype("float32")

    if mode == "rank":
        out = np.empty_like(s, dtype=np.float32)
        for _, g in df.groupby(COL_USER, sort=False):
            idx = g.index.values
            vals = g[score_col].values
            order = np.argsort(-vals, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            out[idx] = 1.0 if len(order) == 1 else (1.0 - ranks.astype(np.float32) / (len(order) - 1))
        return pd.Series(out, index=df.index, dtype="float32")

    if mode == "minmax_user":
        out = np.empty_like(s, dtype=np.float32)
        for _, g in df.groupby(COL_USER, sort=False):
            idx = g.index.values
            v = g[score_col].values.astype(np.float32)
            vmin, vmax = float(np.min(v)), float(np.max(v))
            out[idx] = 0.5 if abs(vmax - vmin) < 1e-12 else (v - vmin) / (vmax - vmin)
        return pd.Series(out, index=df.index, dtype="float32")

    if mode == "zscore_user":
        out = np.empty_like(s, dtype=np.float32)
        for _, g in df.groupby(COL_USER, sort=False):
            idx = g.index.values
            v = g[score_col].values.astype(np.float32)
            m, sd = float(np.mean(v)), float(np.std(v) + 1e-6)
            z = (v - m) / sd
            out[idx] = 1.0 / (1.0 + np.exp(-z))  # map z -> [0,1]
        return pd.Series(out, index=df.index, dtype="float32")

    # fallback
    return df[score_col].astype("float32")


# =============================================================================
# Эксперимент
# =============================================================================

class Exp301BlendLinear(BaseExperiment):
    exp_name = "exp301_blend_linear"

    def __init__(self, cfg: Optional[Exp301Config] = None):
        self.cfg = cfg or Exp301Config()
        self.weights_: Dict[str, float] = {}
        self.intercept_: float = 0.0
        self.sources_used_: List[str] = []

    # --------- helpers ---------

    def _collect_sources(self, ctx: dict, mode: str):
        cand_map = _load_candidates(ctx, self.cfg.candidate_source, mode=mode)

        # подтянем score-таблицы
        src_frames: Dict[str, pd.DataFrame] = {}
        for src in self.cfg.base_sources:
            sdf = _get_source_scores(ctx, src, mode, cand_map, self.cfg.features_for_sources)
            if sdf is None or sdf.empty:
                continue
            # нормализуем per-user
            s2 = sdf.copy()
            s2["s_norm"] = _normalize_userwise(s2, mode=self.cfg.norm_mode, score_col="score").astype("float32")
            s2 = s2[[COL_USER, COL_ITEM, "s_norm"]].rename(columns={"s_norm": f"s_{src}"})
            src_frames[src] = s2

        if not src_frames:
            raise RuntimeError("[exp301] no source scores were found/derived for blend")

        # собираем общий фрейм (u,i) + колонки s_<src>
        base = _cand_map_to_df(cand_map)
        df = base.copy()
        for src, s2 in src_frames.items():
            df = df.merge(s2, on=[COL_USER, COL_ITEM], how="left")
        # отсутствующие — 0
        for c in df.columns:
            if c.startswith("s_"):
                df[c] = df[c].fillna(0.0).astype("float32")

        # ограничим topK per user, если требуется (по сумме нормированных скороов)
        if self.cfg.topk_limit_per_user and self.cfg.topk_limit_per_user > 0:
            score_cols = [c for c in df.columns if c.startswith("s_")]
            df["sum_s"] = df[score_cols].sum(axis=1).astype("float32")
            df = (df.sort_values([COL_USER, "sum_s"], ascending=[True, False])
                    .groupby(COL_USER, sort=False).head(int(self.cfg.topk_limit_per_user)).reset_index(drop=True))
            df = df.drop(columns=["sum_s"])

        return cand_map, df

    def _build_targets(self, cand_map: Dict[int, List[int]], truth: Dict[int, set]) -> pd.Series:
        rows = []
        for u, items in cand_map.items():
            tgt = truth.get(int(u), set())
            for it in items:
                rows.append(1 if int(it) in tgt else 0)
        return pd.Series(rows, dtype="int8")

    def _fit_weights(self, X: np.ndarray, y: np.ndarray, src_names: List[str]) -> Tuple[np.ndarray, float]:
        """
        Находим w, b: y ~ Xw + b. По умолчанию b выключен (cfg.with_intercept).
        Решаем LS, затем w>=0 (если нужно), потом нормируем к сумме 1 (если нужно).
        """
        cfg = self.cfg
        if cfg.with_intercept:
            Xext = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
        else:
            Xext = X

        # LS
        try:
            w_ext, *_ = np.linalg.lstsq(Xext, y, rcond=None)
            w_ext = w_ext.astype(np.float32)
        except Exception:
            # fallback: равные веса
            w_ext = np.ones(Xext.shape[1], dtype=np.float32) / float(Xext.shape[1])

        if cfg.with_intercept:
            w, b = w_ext[:-1], float(w_ext[-1])
        else:
            w, b = w_ext, 0.0

        if cfg.non_negative:
            w = np.maximum(0.0, w)

        if cfg.normalize_weights:
            s = float(np.sum(w))
            if s > 1e-12:
                w = w / s

        # на всякий случай стабилизация
        w = w.astype(np.float32)
        return w, b

    def _blend_scores(self, df: pd.DataFrame, weights: Dict[str, float], intercept: float = 0.0) -> pd.Series:
        cols = [f"s_{src}" for src in self.sources_used_]
        M = df[cols].to_numpy(dtype=np.float32)
        w = np.asarray([weights[src] for src in self.sources_used_], dtype=np.float32)
        s = (M @ w) + float(intercept)
        return pd.Series(s, index=df.index, dtype="float32")

    # --------- API ---------

    def fit(self, ctx: dict):
        self.ctx = ctx
        cfg = self.cfg

        # соберём валидационный фрейм со всеми источниками
        cand_map, df = self._collect_sources(ctx, mode="val")

        # список источников, реально попавших в датафрейм
        self.sources_used_ = [src for src in self.cfg.base_sources if f"s_{src}" in df.columns]
        if not self.sources_used_:
            raise RuntimeError("[exp301] no usable sources after merge")

        # подготовим X, y
        X = df[[f"s_{src}" for src in self.sources_used_]].to_numpy(dtype=np.float32)
        y = self._build_targets(cand_map, ctx["val_truth"]).to_numpy(dtype=np.float32)

        # fit weights
        w, b = self._fit_weights(X, y, self.sources_used_)
        self.weights_ = {src: float(w[i]) for i, src in enumerate(self.sources_used_)}
        self.intercept_ = float(b)

        # blended вал-скор
        val_df = df[[COL_USER, COL_ITEM]].copy()
        val_df["score"] = self._blend_scores(df, self.weights_, self.intercept_).astype("float32")

        # метрики
        truth: Dict[int, set] = ctx["val_truth"]
        pred_map = (
            val_df.sort_values([COL_USER, "score"], ascending=[True, False])
                  .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        rows = []
        for k in cfg.k_list:
            rows.append({
                "k": k,
                "HR@k": recall_at_m(pred_map, truth, m=k),
                "NDCG@k": ndcg_at_k(pred_map, truth, k=k),
            })

        # сохранения
        model_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(model_dir / "val_scores.parquet", val_df)
        save_df(model_dir / "val_metrics.csv", pd.DataFrame(rows), index=False)
        with open(model_dir / "blend_weights.json", "w", encoding="utf-8") as f:
            json.dump({
                "sources": self.sources_used_,
                "weights": [self.weights_[s] for s in self.sources_used_],
                "intercept": self.intercept_,
                "norm_mode": cfg.norm_mode,
                "non_negative": cfg.non_negative,
                "normalize_weights": cfg.normalize_weights,
                "with_intercept": cfg.with_intercept,
            }, f, ensure_ascii=False, indent=2)

        # W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            for r in rows:
                wb.summary[f"HR@{r['k']}"] = r["HR@k"]
                wb.summary[f"NDCG@{r['k']}"] = r["NDCG@k"]
            for s in self.sources_used_:
                wb.summary[f"weight/{s}"] = self.weights_[s]
            wb.summary["intercept"] = self.intercept_

        return {"metrics": rows, "weights": self.weights_}

    def evaluate(self, ctx: dict):
        self.ctx = ctx
        p = PATHS.artifact_dir / "models" / self.exp_name / "val_metrics.csv"
        if p.exists():
            return {"metrics_csv": str(p)}
        return {}

    def save(self, ctx: dict):
        """Инференс на тесте (если доступны кандидаты/скоры) и сохранение test_scores.parquet."""
        self.ctx = ctx
        cfg = self.cfg

        # загрузим веса, если не обучались в этом процессе
        if not self.weights_:
            model_dir = PATHS.artifact_dir / "models" / self.exp_name
            jp = model_dir / "blend_weights.json"
            if jp.exists():
                obj = json.loads(jp.read_text(encoding="utf-8"))
                self.sources_used_ = list(obj.get("sources", []))
                ws = obj.get("weights", [])
                self.weights_ = {s: float(w) for s, w in zip(self.sources_used_, ws)}
                self.intercept_ = float(obj.get("intercept", 0.0))

        # кандидаты теста могут отсутствовать (тогда просто выходим)
        try:
            cand_map, df = self._collect_sources(ctx, mode="test")
        except Exception:
            return {"saved": True}

        # проверим, что есть нужные источники
        missing = [s for s in self.sources_used_ if f"s_{s}" not in df.columns]
        for s in missing:
            # если чего-то из sources_used_ нет — просто добавим нулевую колонку
            df[f"s_{s}"] = 0.0

        test_df = df[[COL_USER, COL_ITEM]].copy()
        test_df["score"] = self._blend_scores(df, self.weights_, self.intercept_).astype("float32")

        model_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(model_dir / "test_scores.parquet", test_df)
        return {"saved": True}

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: Optional[int] = None) -> pd.DataFrame:
        if sample_df is None or sample_df.empty:
            raise ValueError("[exp301] sample_df is required")

        self.ctx = ctx
        K = int(k_top or self.cfg.k_top_submit)

        model_dir = PATHS.artifact_dir / "models" / self.exp_name
        p = model_dir / "test_scores.parquet"
        if p.exists():
            df = pd.read_parquet(p)
        else:
            # попробуем посчитать прямо сейчас
            try:
                _, df_all = self._collect_sources(ctx, mode="test")
                # подхватим веса
                if not self.weights_:
                    jp = model_dir / "blend_weights.json"
                    if jp.exists():
                        obj = json.loads(jp.read_text(encoding="utf-8"))
                        self.sources_used_ = list(obj.get("sources", []))
                        ws = obj.get("weights", [])
                        self.weights_ = {s: float(w) for s, w in zip(self.sources_used_, ws)}
                        self.intercept_ = float(obj.get("intercept", 0.0))
                # добьём отсутствующие колонки
                for s in self.sources_used_:
                    c = f"s_{s}"
                    if c not in df_all.columns:
                        df_all[c] = 0.0
                df = df_all[[COL_USER, COL_ITEM]].copy()
                df["score"] = self._blend_scores(df_all, self.weights_, self.intercept_).astype("float32")
            except Exception:
                # нет теста — пустой сабмит (или можно кинуть исключение)
                df = pd.DataFrame(columns=[COL_USER, COL_ITEM, "score"])

        topk = (
            df.sort_values([COL_USER, "score"], ascending=[True, False])
              .groupby(COL_USER)[COL_ITEM].apply(lambda s: " ".join(map(str, s.head(K).tolist())))
              .reset_index().rename(columns={COL_ITEM: "items"})
        )

        sub = sample_df.copy()
        if "items" not in sub.columns:
            sub["items"] = ""
        sub = sub.drop(columns=["items"], errors="ignore").merge(topk, on=COL_USER, how="left")
        sub["items"] = sub["items"].fillna("")

        out_csv = ensure_dir(PATHS.submission_dir) / f"sub_{self.exp_name}_k{K}.csv"
        sub.to_csv(out_csv, index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            try: wb.save(str(out_csv))
            except Exception: pass

        return sub
