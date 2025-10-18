# -*- coding: utf-8 -*-
"""
exp206_calibration — калибровка скоринга выбранной базовой модели:
  • source_exp ∈ {exp201_ranker_lgbm, exp202_ranker_catboost, exp203_ranker_xgb, exp205_ranker_stacking}
  • method: 'isotonic' | 'platt' | 'auto' (подбираем по logloss на валидации)
  • fit: считаем сырые скоры модели на вал, учим калибратор
  • evaluate: логируем метрики калибровки (logloss/brier/auc), HR/NDCG (для справки)
  • save: сохраняем параметры калибратора и вал-скоры
  • predict_submission: сырые тест-скоры → калибратор → сабмит

Зависимости: scikit-learn (+ lightgbm/catboost/xgboost — смотря что выбрано как источник).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import glob
import os
import json
import numpy as np
import pandas as pd

# base libs (optional)
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

# sklearn for calibration
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
    log_loss = None
    roc_auc_score = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.features.schema import load_schema, apply_norm, coerce_dtypes, FEATURES_VERSION
from vseros_b.features.builders import _cand_map_to_df as cand_map_to_df  # reuse helpers
from vseros_b.candidates.io import load_compact_map


# -------------------------
# Конфиг
# -------------------------

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
class Exp206Config:
    source_exp: str = "exp205_ranker_stacking"     # что калибруем
    features_for_source: Sequence[str] = tuple(DEFAULT_FEATURES)  # под что считались исходные модели
    out_tag_source: str = FEATURES_VERSION
    method: str = "auto"                            # 'isotonic'|'platt'|'auto'
    seed: int = 42
    k_list: Sequence[int] = (20, 50, 100)
    k_top_submit: int = 20


# -------------------------
# Вспомогалки
# -------------------------

def _pick_latest_cand(exp_dir: Path, mode: str) -> Optional[Path]:
    if not exp_dir.exists():
        return None
    patt = "val_candidates*.parquet" if mode == "val" else "test_candidates*.parquet"
    files = glob.glob(str(exp_dir / patt))
    if not files:
        return None
    files = sorted(files, key=lambda p: os.path.getsize(p))
    return Path(files[-1])

def _load_fusion_map_from_ctx_or_artifacts(ctx: dict, mode: str = "val") -> Dict[int, List[int]]:
    cand_map = ctx.get("fusion_map_" + mode) or ctx.get("cand_map_" + mode) or ctx.get("fusion_map")
    if cand_map:
        return cand_map
    base = PATHS.cand_dir / "exp108_recall_fusion"
    p = _pick_latest_cand(base, mode=mode)
    if not p:
        raise FileNotFoundError(f"[exp206] no fusion candidates for mode={mode} in {base}")
    return load_compact_map(p)

def _artifact_model_dir(exp_name: str) -> Path:
    return PATHS.artifact_dir / "models" / exp_name

def _load_schema_from_model_dir(model_dir: Path) -> Dict[str, dict]:
    sp = model_dir / "feature_schema.json"
    if not sp.exists():
        feat_dir = PATHS.artifact_dir / "features" / FEATURES_VERSION
        sp = feat_dir / "feature_schema.json"
    return load_schema(sp) if sp.exists() else {}

def _build_features(ctx: dict, user2items: Dict[int, List[int]], feature_names: Sequence[str]) -> pd.DataFrame:
    base = cand_map_to_df(user2items)
    autodiscover()
    res = FeatureResources(ctx=ctx)
    reg = select(list(feature_names))
    feats_df = base.copy()
    for _, (spec, builder) in reg.items():
        part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
        feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

    # NaN-safe & dtypes
    for c in feats_df.columns:
        if c in (COL_USER, COL_ITEM): 
            continue
        if pd.api.types.is_float_dtype(feats_df[c]):
            feats_df[c] = feats_df[c].fillna(0.0).astype("float32")
        else:
            feats_df[c] = feats_df[c].fillna(0)
    return feats_df


# ---- предсказания исходной модели ----

def _predict_exp201_scores(ctx: dict, cand_map: Dict[int, List[int]], exp_dir: Path, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    if lgb is None:
        return None
    model_path = exp_dir / "model.txt"
    if not model_path.exists():
        return None
    booster = lgb.Booster(model_file=str(model_path))
    feat_names = booster.feature_name()

    stats = _load_schema_from_model_dir(exp_dir)
    feats_df = _build_features(ctx, cand_map, feature_names)
    feats_df = coerce_dtypes(feats_df, [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)])
    feats_df = apply_norm(feats_df, stats)

    X = pd.DataFrame(index=feats_df.index)
    for f in feat_names:
        X[f] = feats_df[f] if f in feats_df.columns else 0.0
    scores = booster.predict(X, raw_score=False)
    out = feats_df[[COL_USER, COL_ITEM]].copy()
    out["raw_score"] = scores.astype("float32")
    return out

def _predict_exp202_scores(ctx: dict, cand_map: Dict[int, List[int]], exp_dir: Path, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    if CatBoostRanker is None:
        return None
    p = exp_dir / "model.cbm"
    if not p.exists():
        return None
    model = CatBoostRanker()
    model.load_model(str(p))
    feat_names = list(getattr(model, "feature_names_", []))

    stats = _load_schema_from_model_dir(exp_dir)
    feats_df = _build_features(ctx, cand_map, feature_names)
    feats_df = coerce_dtypes(feats_df, [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)])
    feats_df = apply_norm(feats_df, stats)

    if feat_names and all(f in feats_df.columns for f in feat_names):
        X = feats_df[feat_names]
    else:
        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        X = feats_df[sorted(feat_cols)]
    pool = CatPool(data=X)
    scores = model.predict(pool)
    out = feats_df[[COL_USER, COL_ITEM]].copy()
    out["raw_score"] = np.asarray(scores, dtype="float32")
    return out

def _predict_exp203_scores(ctx: dict, cand_map: Dict[int, List[int]], exp_dir: Path, feature_names: Sequence[str]) -> Optional[pd.DataFrame]:
    if XGBRanker is None:
        return None
    p = exp_dir / "model.json"
    if not p.exists():
        return None
    model = XGBRanker()
    model.load_model(str(p))
    feat_names = list(model.get_booster().feature_names) if model.get_booster() is not None else None

    stats = _load_schema_from_model_dir(exp_dir)
    feats_df = _build_features(ctx, cand_map, feature_names)
    feats_df = coerce_dtypes(feats_df, [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)])
    feats_df = apply_norm(feats_df, stats)

    if feat_names and all(f in feats_df.columns for f in feat_names):
        X = feats_df[feat_names]
    else:
        feat_cols = [c for c in feats_df.columns if c not in (COL_USER, COL_ITEM)]
        X = feats_df[sorted(feat_cols)]
    scores = model.predict(X)
    out = feats_df[[COL_USER, COL_ITEM]].copy()
    out["raw_score"] = np.asarray(scores, dtype="float32")
    return out

# Для exp205 (стэкинг) — это тоже LGBM с другими входными фичами.
# Чтобы не дублировать всю сборку мета-признаков, здесь даём упрощённый путь:
# используем сохранённые val_scores, если есть. Если нет — падаем с подсказкой.
def _load_exp205_scores_if_available(mode: str, exp_dir: Path) -> Optional[pd.DataFrame]:
    p = exp_dir / ("val_scores.parquet" if mode == "val" else "val_scores.parquet")
    # Для test у exp205 у нас обычно нет сохранённых score-файлов — их можно получить вызовом exp205.predict_submission(),
    # но калибровке для сабмита достаточно применить калибратор к тем же «сырым» очкам, которые прошли в сабмит.
    return pd.read_parquet(p) if p.exists() else None


def _predict_source_scores(ctx: dict, mode: str, source_exp: str, cand_map: Dict[int, List[int]], feature_names: Sequence[str]) -> pd.DataFrame:
    mdir = _artifact_model_dir(source_exp)
    if not mdir.exists():
        raise FileNotFoundError(f"[exp206] model dir not found for source_exp={source_exp}: {mdir}")

    if source_exp == "exp201_ranker_lgbm":
        df = _predict_exp201_scores(ctx, cand_map, mdir, feature_names)
    elif source_exp == "exp202_ranker_catboost":
        df = _predict_exp202_scores(ctx, cand_map, mdir, feature_names)
    elif source_exp == "exp203_ranker_xgb":
        df = _predict_exp203_scores(ctx, cand_map, mdir, feature_names)
    elif source_exp == "exp205_ranker_stacking":
        # пробуем взять val_scores напрямую
        if mode == "val":
            df = _load_exp205_scores_if_available(mode, mdir)
            if df is not None and "score" in df.columns:
                df = df.rename(columns={"score": "raw_score"})[[COL_USER, COL_ITEM, "raw_score"]]
            else:
                raise RuntimeError("[exp206] exp205 has no saved val_scores; run exp205.evaluate/save first.")
        else:
            # Для теста у exp205 часто нет score-файла. В таком случае пользователь должен сначала вызвать
            # exp205.predict_submission — тогда сабмит есть, но «сырые» score у нас не сохранены.
            # Поэтому для source=exp205 мы калибруем только вал-скоры и при сабмите применяем к исходным score другого ранкера
            # (рекомендуется использовать source_exp=exp201/202/203 для стабильности).
            raise RuntimeError("[exp206] test-time scores for exp205 are not available; prefer source_exp=exp201/202/203.")
    else:
        raise ValueError(f"[exp206] unsupported source_exp={source_exp}")

    if df is None or df.empty:
        raise RuntimeError(f"[exp206] cannot obtain raw scores for {source_exp} ({mode})")
    return df


# -------------------------
# Калибраторы
# -------------------------

class _Isotonic:
    def __init__(self):
        if IsotonicRegression is None:
            raise ImportError("scikit-learn is required for isotonic calibration")
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        # Isotonic ожидает вещественный скор, таргет ∈ [0,1]
        self.model.fit(y_pred.astype("float64"), y_true.astype("float64"))
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        return self.model.predict(y_pred.astype("float64")).astype("float32")

    def dump(self) -> dict:
        # сохраняем «ступеньки»
        return {
            "type": "isotonic",
            "X_thresholds_": self.model.X_thresholds_.tolist(),
            "y_thresholds_": self.model.y_thresholds_.tolist(),
        }


class _Platt:
    def __init__(self):
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for platt calibration")
        # Простая логистическая регрессия с одним признаком
        self.model = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            max_iter=1000,
            n_jobs=None,
            random_state=42,
        )

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        X = y_pred.reshape(-1, 1).astype("float64")
        y = y_true.astype("int32")
        self.model.fit(X, y)
        return self

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        X = y_pred.reshape(-1, 1).astype("float64")
        proba = self.model.predict_proba(X)[:, 1]
        return proba.astype("float32")

    def dump(self) -> dict:
        # извлечём коэффициенты
        coef = float(self.model.coef_.ravel()[0])
        bias = float(self.model.intercept_.ravel()[0])
        return {
            "type": "platt",
            "coef": coef,
            "bias": bias,
        }


def _choose_and_fit_calibrator(method: str, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[object, dict, Dict[str, float]]:
    """
    Возвращает (калибратор, dump_dict, metrics_on_val)
    """
    def _eval_metrics(p):
        m = {}
        if log_loss is not None:
            m["logloss"] = float(log_loss(y_true, np.clip(p, 1e-7, 1-1e-7)))
        if roc_auc_score is not None:
            # AUC по вероятностям
            try:
                m["auc"] = float(roc_auc_score(y_true, p))
            except Exception:
                m["auc"] = float("nan")
        # Brier
        m["brier"] = float(np.mean((p - y_true) ** 2))
        return m

    methods = []
    if method == "auto":
        methods = ["isotonic", "platt"]
    else:
        methods = [method]

    best = None
    best_name = None
    best_metrics = None
    best_dump = None

    for name in methods:
        if name == "isotonic":
            cal = _Isotonic().fit(y_pred, y_true)
        elif name == "platt":
            cal = _Platt().fit(y_pred, y_true)
        else:
            raise ValueError("method must be one of ['isotonic','platt','auto']")

        p = cal.predict(y_pred)
        m = _eval_metrics(p)

        if best is None or (m.get("logloss", np.inf) < best_metrics.get("logloss", np.inf)):
            best = cal
            best_name = name
            best_metrics = m
            best_dump = cal.dump()

    best_dump["chosen"] = best_name
    return best, best_dump, best_metrics


# -------------------------
# Эксперимент
# -------------------------

class Exp206Calibration(BaseExperiment):
    exp_name = "exp206_calibration"

    def __init__(self, cfg: Optional[Exp206Config] = None):
        self.cfg = cfg or Exp206Config()
        self.calibrator = None
        self.meta: Optional[dict] = None

    # ---------- core ----------

    def fit(self, ctx: dict):
        self.ctx = ctx
        cfg = self.cfg

        # 1) кандидаты (валидация)
        fusion_map = _load_fusion_map_from_ctx_or_artifacts(ctx, mode="val")

        # 2) получить «сырые» скоры источника на вал
        df = _predict_source_scores(ctx, "val", cfg.source_exp, fusion_map, cfg.features_for_source)
        # разметить
        val_truth: Dict[int, set] = ctx["val_truth"]
        df["label"] = np.fromiter(
            (1 if int(i) in val_truth.get(int(u), set()) else 0 for u, i in zip(df[COL_USER].values, df[COL_ITEM].values)),
            count=len(df),
            dtype=np.int8,
        )

        # 3) обучить калибратор
        y_pred = df["raw_score"].astype("float32").values
        y_true = df["label"].astype("float32").values
        cal, dump, cal_metrics = _choose_and_fit_calibrator(cfg.method, y_pred, y_true)
        self.calibrator = cal
        self.meta = {"dump": dump, "cal_metrics_val": cal_metrics}

        # 4) лог в W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            for k, v in cal_metrics.items():
                wb.summary[f"cal_{k}"] = v

        # 5) сохраним вал-таблицу с калиброванными вероятностями
        df_out = df[[COL_USER, COL_ITEM, "raw_score", "label"]].copy()
        df_out["calibrated"] = cal.predict(y_pred)
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_calibrated.parquet", df_out)

        return self

    def evaluate(self, ctx: dict):
        self.ctx = ctx
        if self.calibrator is None:
            return {}

        cfg = self.cfg
        # Для полноты — посчитаем HR/NDCG до/после (они совпадут, т.к. монотонная трансформация),
        # но мы их всё равно залогируем.
        fusion_map = _load_fusion_map_from_ctx_or_artifacts(ctx, mode="val")
        df = _predict_source_scores(ctx, "val", cfg.source_exp, fusion_map, cfg.features_for_source)
        val_truth: Dict[int, set] = ctx["val_truth"]

        # сырые
        raw_pred = (
            df.sort_values([COL_USER, "raw_score"], ascending=[True, False])
              .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        # калиброванные
        df_cal = df.copy()
        df_cal["calibrated"] = self.calibrator.predict(df["raw_score"].astype("float32").values)
        cal_pred = (
            df_cal.sort_values([COL_USER, "calibrated"], ascending=[True, False])
                  .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )

        rows = []
        for k in cfg.k_list:
            rows.append({
                "k": k,
                "HR_raw": recall_at_m(raw_pred, val_truth, m=k),
                "NDCG_raw": ndcg_at_k(raw_pred, val_truth, k=k),
                "HR_cal": recall_at_m(cal_pred, val_truth, m=k),
                "NDCG_cal": ndcg_at_k(cal_pred, val_truth, k=k),
            })

        # W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            for r in rows:
                wb.summary[f"HR_raw@{r['k']}"] = r["HR_raw"]
                wb.summary[f"NDCG_raw@{r['k']}"] = r["NDCG_raw"]
                wb.summary[f"HR_cal@{r['k']}"] = r["HR_cal"]
                wb.summary[f"NDCG_cal@{r['k']}"] = r["NDCG_cal"]

        # сохраним табличку
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_rank_metrics.csv", pd.DataFrame(rows), index=False)

        return {"rank_metrics": rows, **(self.meta or {})}

    def save(self, ctx: dict):
        self.ctx = ctx
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)

        # параметры калибратора
        if self.meta and "dump" in self.meta:
            (out_dir / "calibrator.json").write_text(json.dumps(self.meta["dump"], ensure_ascii=False, indent=2), encoding="utf-8")

        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(out_dir / "calibrator.json"))
                wb.save(str(out_dir / "val_calibrated.parquet"))
                wb.save(str(out_dir / "val_rank_metrics.csv"))
            except Exception:
                pass

        return {"calibrator_path": str(out_dir / "calibrator.json")}

    # ---------- submissions ----------

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: Optional[int] = None) -> pd.DataFrame:
        """
        Применяем калибратор к тест-скорам источника и делаем сабмит.
        Примечание: монотонная калибровка не меняет порядок; это в основном для «проб»/порогов/блендов.
        """
        if self.calibrator is None:
            raise RuntimeError("Calibrator is not fitted")

        self.ctx = ctx
        cfg = self.cfg
        k = int(k_top or cfg.k_top_submit)

        # test кандидаты
        fusion_map_test = _load_fusion_map_from_ctx_or_artifacts(ctx, mode="test")
        # получить сырые скоры источника
        df = _predict_source_scores(ctx, "test", cfg.source_exp, fusion_map_test, cfg.features_for_source)

        # калибруем
        df["calibrated"] = self.calibrator.predict(df["raw_score"].astype("float32").values)

        # top-K по калиброванному скору
        topk = (
            df.sort_values([COL_USER, "calibrated"], ascending=[True, False])
              .groupby(COL_USER)[COL_ITEM].apply(lambda s: " ".join(map(str, s.head(k).tolist())))
              .reset_index().rename(columns={COL_ITEM: "items"})
        )

        sub = sample_df.copy()
        if "items" not in sub.columns:
            sub["items"] = ""
        sub = sub.drop(columns=["items"], errors="ignore").merge(topk, on="user_id", how="left")
        sub["items"] = sub["items"].fillna("")

        out_dir = ensure_dir(PATHS.submission_dir)
        out_csv = out_dir / f"sub_{self.exp_name}_{cfg.source_exp}_k{k}.csv"
        sub.to_csv(out_csv, index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(out_csv))
            except Exception:
                pass

        return sub
