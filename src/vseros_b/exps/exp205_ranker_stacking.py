# -*- coding: utf-8 -*-
"""
exp205_ranker_stacking — level-2 ранкер, который учится на выходах базовых моделей
(201/202/203 и/или готовых score-таблицах других экспериментов) + опционально фичах,
и выдаёт финальный скор. ВАЖНО: сохраняет val_scores/test_scores.parquet, чтобы
реранкеры/блендеры могли их брать без пересчёта.

Пайплайн (val):
  1) Берём кандидатов (обычно из exp108_recall_fusion).
  2) Для каждого source из base_sources собираем колонку score_<src>:
     - если есть артефакт <src>/val_scores.parquet — берём его;
     - если src ∈ {exp201,exp202,exp203} и артефакта нет — пересчитываем по фичам.
  3) (опц.) достраиваем дополнительные фичи (features_for_level2).
  4) Собираем таргет (u,i ∈ truth → 1, иначе 0). Обучаем L2-модель (LightGBM или
     numpy ridge fallback). Считаем метрики, сохраняем артефакты:
     - artifacts/models/exp205_ranker_stacking/model.txt (если lgbm)
     - artifacts/models/exp205_ranker_stacking/feature_schema.json
     - artifacts/models/exp205_ranker_stacking/val_scores.parquet
     - artifacts/models/exp205_ranker_stacking/val_metrics.csv
  5) Для test (если известны тест-пользователи/кандидаты) — сохраняем test_scores.parquet.
  6) predict_submission() формирует сабмит по test_scores.

Безопасность утечек: все фичи строятся строго из train_df, метки — из val_truth.
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

# optional libs
try:
    import lightgbm as lgb
except Exception:
    lgb = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM, COL_DATE
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.features.schema import FEATURES_VERSION, save_schema, coerce_dtypes, apply_norm, infer_schema_from_df
from vseros_b.candidates.io import load_compact_map, save_compact_map


# =============================================================================
# Конфиг
# =============================================================================

DEFAULT_LEVEL2_FEATURES: Sequence[str] = (
    # Можно держать пустым — чистый стэкинг по score_* колонкам тоже ок.
    "sources_basic", "pop_basic", "calendar_basic", "user_basic",
    "i2v_sim", "covis_local", "markov_basic", "regularizers_basic",
    "interact_strong", "diag_basic",
)

@dataclass
class Exp205Config:
    # откуда брать кандидатов
    candidate_source: str = "exp108_recall_fusion"   # где лежит compact-map
    topk_candidates: int = 400                       # ограничим пул per user (ускорение)

    # какие базовые источники скоров стэкаем
    base_sources: Sequence[str] = (
        "exp201_ranker_lgbm",
        "exp202_ranker_catboost",
        "exp203_ranker_xgb",
        # при желании можно добавить "exp106_lightgcn_import", "exp107_ppr", "exp102_daily_trending", ...
    )

    # фичи для level-2 (опционально)
    use_extra_features: bool = True
    features_for_level2: Sequence[str] = DEFAULT_LEVEL2_FEATURES
    out_tag_features: str = FEATURES_VERSION

    # модель L2
    lgb_params: dict = None     # если None — дефолтный сет ниже
    num_boost_round: int = 200
    valid_fraction: float = 0.1 # от валидационного набора выделим кусок под lgbm-валидацию (ранний стоп)

    # метрики/сабмит
    k_list: Sequence[int] = (20, 50, 100)
    k_top_submit: int = 20

    seed: int = 42

    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = dict(
                objective="binary",
                metric="auc",
                boosting="gbdt",
                learning_rate=0.05,
                num_leaves=63,
                min_data_in_leaf=50,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                lambda_l1=0.0,
                lambda_l2=1.0,
                verbose=-1,
                seed=int(self.seed),
                num_threads=0,
            )


# =============================================================================
# Вспомогалки: кандидаты и источники скоров
# =============================================================================

def _pick_latest_cand(exp_dir: Path, mode: str) -> Optional[Path]:
    if not exp_dir.exists(): return None
    patt = "val_candidates*.parquet" if mode == "val" else "test_candidates*.parquet"
    files = glob.glob(str(exp_dir / patt))
    if not files: return None
    files = sorted(files, key=lambda p: os.path.getsize(p))
    return Path(files[-1])

def _load_candidates(ctx: dict, exp_name: str, mode: str) -> Dict[int, List[int]]:
    # 1) из ctx, если уже положили
    m = ctx.get("fusion_map_" + mode) or ctx.get("cand_map_" + mode) or ctx.get("fusion_map")
    if m: return m
    # 2) compact parquet артефакт
    base = PATHS.cand_dir / exp_name
    p = _pick_latest_cand(base, mode=mode)
    if not p:
        raise FileNotFoundError(f"[exp205] no candidates for mode={mode} in {base}")
    return load_compact_map(p)

def _artifact_model_dir(exp_name: str) -> Path:
    return PATHS.artifact_dir / "models" / exp_name

def _load_scores_artifact(exp_name: str, mode: str) -> Optional[pd.DataFrame]:
    """Если источник уже сохранял val_scores/test_scores — забираем."""
    mdir = _artifact_model_dir(exp_name)
    p = mdir / ("val_scores.parquet" if mode == "val" else "test_scores.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        cols = set(df.columns)
        if "score" in cols:
            return df[[COL_USER, COL_ITEM, "score"]].copy()
        if "blended" in cols:
            return df[[COL_USER, COL_ITEM]].assign(score=df["blended"].astype("float32"))
    return None

# ------- пересчёт базовых ранкеров (201/202/203), если нет артефактов -------

def _cand_map_to_df(map_ui: Dict[int, List[int]]) -> pd.DataFrame:
    rows = []
    for u, items in map_ui.items():
        for it in items:
            rows.append((int(u), int(it)))
    return pd.DataFrame(rows, columns=[COL_USER, COL_ITEM])

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
    if sp.exists():
        with open(sp, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _predict_exp201(ctx, mode, cand_map, feats):
    try:
        import lightgbm as lgb_local
    except Exception:
        return None
    mdir = _artifact_model_dir("exp201_ranker_lgbm")
    mp = mdir / "model.txt"
    if not mp.exists(): return None
    booster = lgb_local.Booster(model_file=str(mp))
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
    try:
        from catboost import CatBoostRanker, Pool as CatPool
    except Exception:
        return None
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
    try:
        from xgboost import XGBRanker
    except Exception:
        return None
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
    # сначала — готовые артефакты
    df = _load_scores_artifact(exp_name, mode)
    if df is not None:
        return df

    # пересчёт только для базовых деревьев
    if exp_name == "exp201_ranker_lgbm":
        return _predict_exp201(ctx, mode, cand_map, feats_for_source)
    if exp_name == "exp202_ranker_catboost":
        return _predict_exp202(ctx, mode, cand_map, feats_for_source)
    if exp_name == "exp203_ranker_xgb":
        return _predict_exp203(ctx, mode, cand_map, feats_for_source)

    # для других источников (exp106, exp107, exp102...) ожидаем val/test_scores.parquet
    return None


# =============================================================================
# Утилиты level-2
# =============================================================================

def _assemble_level2_frame(cand_map: Dict[int, List[int]],
                           source_scores: Dict[str, pd.DataFrame],
                           extra_features: Optional[pd.DataFrame] = None,
                           score_fill: float = 0.0) -> pd.DataFrame:
    """
    Собирает единую таблицу (u,i, score_src1, score_src2, ..., + extra features).
    """
    base = _cand_map_to_df(cand_map)
    df = base.copy()
    for src, sdf in source_scores.items():
        col = f"score_{src}"
        sdf2 = sdf.rename(columns={"score": col})
        df = df.merge(sdf2, on=[COL_USER, COL_ITEM], how="left")
        if col in df.columns:
            if pd.api.types.is_float_dtype(df[col]): df[col] = df[col].fillna(score_fill).astype("float32")
            else: df[col] = df[col].fillna(score_fill)
    if extra_features is not None and len(extra_features):
        # ожидание: extra_features содержит [user_id, item_id, ...feature columns...]
        feat_cols = [c for c in extra_features.columns if c not in (COL_USER, COL_ITEM)]
        df = df.merge(extra_features[[COL_USER, COL_ITEM] + feat_cols], on=[COL_USER, COL_ITEM], how="left")
        for c in feat_cols:
            if pd.api.types.is_float_dtype(df[c]): df[c] = df[c].fillna(0.0).astype("float32")
            else: df[c] = df[c].fillna(0)
    return df

def _build_extra_features(ctx: dict, cand_map: Dict[int, List[int]], feature_names: Sequence[str]) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame(columns=[COL_USER, COL_ITEM])
    return _build_features(ctx, cand_map, feature_names)

def _build_targets(cand_map: Dict[int, List[int]], truth: Dict[int, set]) -> pd.Series:
    """1 если item ∈ truth[u], иначе 0."""
    rows = []
    for u, items in cand_map.items():
        tgt = truth.get(int(u), set())
        for it in items:
            rows.append(1 if int(it) in tgt else 0)
    return pd.Series(rows, dtype="int8")

def _train_l2_lgbm(X: pd.DataFrame, y: pd.Series, params: dict, num_boost_round: int, valid_fraction: float, seed: int):
    n = len(X)
    if n == 0:
        raise RuntimeError("[exp205] empty training frame for L2")
    # простой holdout из val (без утечки, мы и так валидационную часть используем как train для l2)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(max(32, n * (1.0 - float(valid_fraction))))
    tr_idx, va_idx = idx[:cut], idx[cut:]
    dtr = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
    dva = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx]) if len(va_idx) > 0 else None
    booster = lgb.train(
        params=params,
        train_set=dtr,
        valid_sets=[dtr, dva] if dva is not None else [dtr],
        valid_names=["train", "valid"] if dva is not None else ["train"],
        num_boost_round=int(num_boost_round),
        early_stopping_rounds=50 if dva is not None else None,
        verbose_eval=False,
    )
    return booster

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# =============================================================================
# Эксперимент
# =============================================================================

class Exp205RankerStacking(BaseExperiment):
    exp_name = "exp205_ranker_stacking"

    def __init__(self, cfg: Optional[Exp205Config] = None):
        self.cfg = cfg or Exp205Config()
        self.model = None
        self.feature_list: List[str] = []   # порядок колонок для инференса
        self.schema_stats: Dict[str, dict] = {}

    # ------------- ядро -------------

    def _collect_level2_inputs(self, ctx: dict, mode: str):
        # кандидаты
        cand_map = _load_candidates(ctx, self.cfg.candidate_source, mode=mode)

        # базовые скоры
        src_scores: Dict[str, pd.DataFrame] = {}
        for src in self.cfg.base_sources:
            df = _get_source_scores(ctx, src, mode, cand_map, self.cfg.features_for_level2)
            if df is None:
                # если источник не может пересчитаться и не дал артефакт — просто пропускаем
                continue
            src_scores[src] = df[[COL_USER, COL_ITEM, "score"]].copy()

        # фичи (опционально)
        feats_df = None
        if self.cfg.use_extra_features:
            feats_df = _build_extra_features(ctx, cand_map, self.cfg.features_for_level2)

        return cand_map, src_scores, feats_df

    def _make_train_frame(self, ctx: dict):
        cand_map, src_scores, feats_df = self._collect_level2_inputs(ctx, mode="val")

        # соберём фрейм level-2
        df = _assemble_level2_frame(cand_map, src_scores, feats_df, score_fill=0.0)

        # таргет
        y = _build_targets(cand_map, ctx["val_truth"])

        # готовим X
        feat_cols = [c for c in df.columns if c not in (COL_USER, COL_ITEM)]
        X = df[feat_cols].copy()

        # схема нормализации (пока — просто типы/заполнение)
        self.schema_stats = infer_schema_from_df(X)
        X = coerce_dtypes(X, feat_cols)  # приведение типов (float32/int32) под схему

        self.feature_list = list(X.columns)
        return df[[COL_USER, COL_ITEM]], X, y

    # ------------- API -------------

    def fit(self, ctx: dict):
        self.ctx = ctx
        cfg = self.cfg

        # построим train-фрейм уровня 2 (на основе вал-кандидатов)
        ui_df, X, y = self._make_train_frame(ctx)

        if lgb is None:
            # fallback: простая линейная регрессия на numpy (ridge отсутствует — сделаем L2 вручную)
            # solution: w = (X^T X + λI)^(-1) X^T y
            lam = 1e-3
            Xn = X.to_numpy(dtype=np.float32)
            yn = y.to_numpy(dtype=np.float32)
            XtX = Xn.T @ Xn
            XtX[np.diag_indices_from(XtX)] += lam
            w = np.linalg.solve(XtX, Xn.T @ yn)
            self.model = ("linear_numpy", w)
            raw = Xn @ w
            pred = _sigmoid(raw)
        else:
            booster = _train_l2_lgbm(
                X=X,
                y=y,
                params=cfg.lgb_params,
                num_boost_round=int(cfg.num_boost_round),
                valid_fraction=float(cfg.valid_fraction),
                seed=int(cfg.seed),
            )
            self.model = ("lgbm", booster)
            pred = booster.predict(X, raw_score=False)

        # соберём вал-скоры
        val_df = ui_df.copy()
        val_df["score"] = pred.astype("float32")

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

        # модель
        if self.model and self.model[0] == "lgbm":
            booster: lgb.Booster = self.model[1]
            booster.save_model(str(model_dir / "model.txt"))
            # сохраним список фичей именно в том порядке, в каком их ждёт бустер
            with open(model_dir / "feature_list.json", "w", encoding="utf-8") as f:
                json.dump(booster.feature_name(), f, ensure_ascii=False, indent=2)
        else:
            # линейные веса
            with open(model_dir / "linear_weights.json", "w", encoding="utf-8") as f:
                json.dump({
                    "feature_list": self.feature_list,
                    "weights": [float(x) for x in (self.model[1].tolist() if isinstance(self.model[1], np.ndarray) else [])],
                }, f, ensure_ascii=False, indent=2)

        # схема фичей (для совместимости с apply_norm/coerce_dtypes)
        schema_path = model_dir / "feature_schema.json"
        save_schema(self.schema_stats, schema_path)

        # вал-скоры/метрики
        save_df(model_dir / "val_scores.parquet", val_df)
        save_df(model_dir / "val_metrics.csv", pd.DataFrame(rows), index=False)

        # лог в W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            for r in rows:
                wb.summary[f"HR@{r['k']}"] = r["HR@k"]
                wb.summary[f"NDCG@{r['k']}"] = r["NDCG@k"]
            wb.summary["l2_features"] = len(self.feature_list)

        return {"metrics": rows, "val_rows": int(len(val_df))}

    def evaluate(self, ctx: dict):
        # метрики уже посчитали в fit()
        self.ctx = ctx
        out = {}
        p = PATHS.artifact_dir / "models" / self.exp_name / "val_metrics.csv"
        if p.exists():
            out["metrics_csv"] = str(p)
        return out

    def save(self, ctx: dict):
        """Инференс на test-кандидатах (если доступны), сохранение test_scores.parquet."""
        self.ctx = ctx
        cfg = self.cfg

        # если модель ещё не обучена в этом процессе — попробуем загрузить (для lgbm) или проигнорируем
        if self.model is None:
            model_dir = PATHS.artifact_dir / "models" / self.exp_name
            mp = model_dir / "model.txt"
            if lgb is not None and mp.exists():
                booster = lgb.Booster(model_file=str(mp))
                self.model = ("lgbm", booster)
                # поднимем порядок фич из файла; иначе возьмём то, что в бустере
                flp = model_dir / "feature_list.json"
                if flp.exists():
                    with open(flp, "r", encoding="utf-8") as f:
                        self.feature_list = json.load(f)

        # тестовые пользователи подразумевают наличие candidate_source/test_candidates*
        try:
            cand_map = _load_candidates(ctx, self.cfg.candidate_source, mode="test")
        except Exception:
            return {"saved": True}

        # соберём источники скоров и (опц.) фичи
        src_scores: Dict[str, pd.DataFrame] = {}
        for src in self.cfg.base_sources:
            df = _get_source_scores(ctx, src, "test", cand_map, self.cfg.features_for_level2)
            if df is None:
                continue
            src_scores[src] = df[[COL_USER, COL_ITEM, "score"]].copy()

        feats_df = _build_extra_features(ctx, cand_map, self.cfg.features_for_level2) if self.cfg.use_extra_features else None
        # level-2 фрейм
        df2 = _assemble_level2_frame(cand_map, src_scores, feats_df, score_fill=0.0)

        # приведём типы и порядок фичей в точности как при обучении
        feat_cols = [c for c in df2.columns if c not in (COL_USER, COL_ITEM)]
        X = df2[feat_cols].copy()
        X = coerce_dtypes(X, feat_cols)
        # если есть lgbm — возьмём только те фичи, которые он знает (в порядке feature_name())
        if self.model and self.model[0] == "lgbm":
            booster: lgb.Booster = self.model[1]
            feat_order = list(booster.feature_name())
            # подстрахуемся: некоторые имена могли отсутствовать — добавим пустые
            for f in feat_order:
                if f not in X.columns:
                    X[f] = 0.0
            X = X[feat_order]
            pred = booster.predict(X, raw_score=False)
        elif self.model and self.model[0] == "linear_numpy":
            # восстановим линейные веса
            model_dir = PATHS.artifact_dir / "models" / self.exp_name
            jp = model_dir / "linear_weights.json"
            with open(jp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            feat_order = obj.get("feature_list", list(X.columns))
            w = np.asarray(obj.get("weights", [0.0]*len(feat_order)), dtype=np.float32)
            for f in feat_order:
                if f not in X.columns:
                    X[f] = 0.0
            X = X[feat_order]
            pred = _sigmoid(X.to_numpy(dtype=np.float32) @ w)
        else:
            # нет модели — отдаём нули
            pred = np.zeros(len(X), dtype=np.float32)

        test_scores = df2[[COL_USER, COL_ITEM]].copy()
        test_scores["score"] = pred.astype("float32")

        model_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(model_dir / "test_scores.parquet", test_scores)

        return {"saved": True}

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: Optional[int] = None) -> pd.DataFrame:
        if sample_df is None or sample_df.empty:
            raise ValueError("[exp205] sample_df is required")

        self.ctx = ctx
        K = int(k_top or self.cfg.k_top_submit)

        # попробуем готовые test_scores
        mdir = PATHS.artifact_dir / "models" / self.exp_name
        p = mdir / "test_scores.parquet"
        if p.exists():
            df = pd.read_parquet(p)
        else:
            # если нет — сделаем save(), который должен их сгенерить
            self.save(ctx)
            if p.exists():
                df = pd.read_parquet(p)
            else:
                # последний шанс: посчитаем на лету
                cand_map = _load_candidates(ctx, self.cfg.candidate_source, mode="test")
                src_scores = {}
                for src in self.cfg.base_sources:
                    sdf = _get_source_scores(ctx, src, "test", cand_map, self.cfg.features_for_level2)
                    if sdf is not None:
                        src_scores[src] = sdf[[COL_USER, COL_ITEM, "score"]].copy()
                feats_df = _build_extra_features(ctx, cand_map, self.cfg.features_for_level2) if self.cfg.use_extra_features else None
                df2 = _assemble_level2_frame(cand_map, src_scores, feats_df, score_fill=0.0)
                feat_cols = [c for c in df2.columns if c not in (COL_USER, COL_ITEM)]
                X = df2[feat_cols].copy()
                X = coerce_dtypes(X, feat_cols)
                if self.model and self.model[0] == "lgbm":
                    booster: lgb.Booster = self.model[1]
                    feat_order = list(booster.feature_name())
                    for f in feat_order:
                        if f not in X.columns:
                            X[f] = 0.0
                    X = X[feat_order]
                    pred = booster.predict(X, raw_score=False)
                else:
                    pred = np.zeros(len(X), dtype=np.float32)
                df = df2[[COL_USER, COL_ITEM]].copy()
                df["score"] = pred.astype("float32")

        # top-K → сабмит
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
