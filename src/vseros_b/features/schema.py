# -*- coding: utf-8 -*-
"""
schema.py — единая политика для фич:
- версии (FEATURES_VERSION)
- подсказки по нормализации/типам (regex-хинты)
- fit/apply нормализаций
- сохранение/загрузка схемы

Использование:
from vseros_b.features.schema import FEATURES_VERSION, fit_norm_stats, apply_norm, save_schema, load_schema
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Sequence
import json
import re
import numpy as np
import pandas as pd

Norm = Literal["none", "minmax", "robust"]

FEATURES_VERSION: str = "v1"

@dataclass(frozen=True)
class FeatureMeta:
    norm: Norm = "robust"
    dtype: str = "float32"
    note: str = ""

# -----------------------------------------------------------------------------
# Хинты (regex -> FeatureMeta)
# -----------------------------------------------------------------------------
# Порядок важен: матчится первый подходящий шаблон
_HINTS: list[tuple[re.Pattern, FeatureMeta]] = [
    # индикаторы и бинарные
    (re.compile(r"^(in_|overlap_|diag_any_|seen_in_train_user|diag_seen_in_user_hist)"), FeatureMeta("none", "int8", "binary")),
    # маленькие целые
    (re.compile(r"^(votes|votes_.*|covis_hits|lastK_len|u_days_active|diag_num_sources)$"), FeatureMeta("none", "int16", "small-int")),
    # ранги
    (re.compile(r"^rank_"), FeatureMeta("minmax", "int32", "rank")),
    (re.compile(r"^diag_rank_min_any$"), FeatureMeta("minmax", "int32", "rank-agg")),
    # reciprocal ranks, косинусы, dot’ы, граф-скоры
    (re.compile(r"^(rr_|i2v_cos_|lgcn_dot|ppr_score|markov_)"), FeatureMeta("minmax", "float32", "bounded")),
    # кол-ва кликов и частоты
    (re.compile(r"^(u_clicks_|u_uniqs_|u_i_cnt|item_cnt_)"), FeatureMeta("robust", "int32", "counts")),
    # «возраст»/реценси
    (re.compile(r"^(days_since_|age_)"), FeatureMeta("robust", "int32", "recency/age")),
    # тренды/новизна/регуляризаторы/интеракции
    (re.compile(r"^(trend_|novelty|anti_|.*__.*|recency_bias)$"), FeatureMeta("robust", "float32", "trend/regularizer")),
    # календарь
    (re.compile(r"^(dow_(sin|cos))$"), FeatureMeta("none", "float32", "global-const")),
    # диагностика
    (re.compile(r"^(diag_)"), FeatureMeta("robust", "float32", "diagnostic")),
]

def _hint_for(col: str) -> FeatureMeta:
    for pat, meta in _HINTS:
        if pat.search(col):
            return meta
    # дефолт
    return FeatureMeta("robust", "float32", "default")

# -----------------------------------------------------------------------------
# Нормализации
# -----------------------------------------------------------------------------

def _fit_minmax(s: pd.Series) -> dict:
    vmin, vmax = float(s.min()), float(s.max())
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax): vmax = 1.0
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return {"norm": "minmax", "min": vmin, "max": vmax}

def _fit_robust(s: pd.Series) -> dict:
    arr = s.astype("float32").values
    med = float(np.nanmedian(arr))
    q25, q75 = np.nanpercentile(arr, [25, 75]) if arr.size else (0.0, 1.0)
    iqr = float(max(q75 - q25, 1e-6))
    return {"norm": "robust", "median": med, "iqr": iqr}

def fit_norm_stats(train_df: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, dict]:
    """
    Строит словарь {col -> {norm: .., params..}} по train-матрице.
    Политика берётся из regex-хинтов выше.
    """
    stats: Dict[str, dict] = {}
    for c in feature_cols:
        meta = _hint_for(c)
        s = train_df[c]
        # dtype-гармонизация
        if meta.dtype.startswith("float"):
            s = s.astype("float32")
        if meta.norm == "none":
            stats[c] = {"norm": "none"}
        elif meta.norm == "minmax":
            stats[c] = _fit_minmax(s)
        else:
            stats[c] = _fit_robust(s)
        # прокинем ожидаемый dtype как подсказку
        stats[c]["dtype"] = meta.dtype
        stats[c]["note"] = meta.note
    return stats

def apply_norm(df: pd.DataFrame, stats: Dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    for c, st in stats.items():
        if c not in out.columns:
            continue
        mode = st.get("norm", "none")
        if mode == "none":
            # приведение к ожидаемому типу (если есть)
            dtype = st.get("dtype")
            if dtype:
                if dtype.startswith("float"):
                    out[c] = out[c].astype("float32")
                elif dtype.startswith("int"):
                    out[c] = out[c].fillna(0).astype(dtype)
            continue
        if mode == "minmax":
            vmin, vmax = st["min"], st["max"]
            out[c] = ((out[c].astype("float32") - vmin) / (vmax - vmin)).astype("float32")
        elif mode == "robust":
            med, iqr = st["median"], st["iqr"]
            out[c] = ((out[c].astype("float32") - med) / iqr).astype("float32")
    return out

# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def save_schema(path: Path, stats: Dict[str, dict], version: str | None = None) -> None:
    data = {
        "features_version": version or FEATURES_VERSION,
        "schema": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_schema(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # обратная совместимость: если лежит чистый dict без обёртки
    if "schema" in data:
        return data["schema"]
    return data

# -----------------------------------------------------------------------------
# Вспомогательное: привести типы колонок согласно хинтам (до нормализации)
# -----------------------------------------------------------------------------

def coerce_dtypes(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        meta = _hint_for(c)
        if meta.dtype.startswith("float"):
            out[c] = out[c].astype("float32")
        elif meta.dtype.startswith("int"):
            out[c] = out[c].fillna(0).astype(meta.dtype)
    return out
