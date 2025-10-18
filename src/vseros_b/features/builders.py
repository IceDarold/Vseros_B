# -*- coding: utf-8 -*-
"""
builders.py — сборщик матриц фич для ранкеров.
Использует:
  - features.registry.autodiscover/select
  - FeatureResources
  - features.schema.{fit_norm_stats, apply_norm, save_schema, coerce_dtypes}

Сохранения:
  artifacts/features/<out_tag>/{train_matrix.parquet,val_matrix.parquet,feature_schema.json}
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import json
import numpy as np
import pandas as pd

from vseros_b.config import PATHS, COL_USER, COL_ITEM
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.features.resources import FeatureResources
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.schema import (
    FEATURES_VERSION,
    fit_norm_stats,
    apply_norm,
    save_schema,
    coerce_dtypes,
)

# -----------------------------
# helpers: base, labels, groups
# -----------------------------

def _cand_map_to_df(user2items: Mapping[int, List[int]]) -> pd.DataFrame:
    rows = [(int(u), int(i)) for u, items in user2items.items() for i in items]
    if not rows:
        return pd.DataFrame(columns=[COL_USER, COL_ITEM], dtype="int64")
    df = pd.DataFrame(rows, columns=[COL_USER, COL_ITEM])
    df[COL_USER] = df[COL_USER].astype("int64")
    df[COL_ITEM] = df[COL_ITEM].astype("int64")
    return df


def _apply_features(df_base: pd.DataFrame, ctx: dict, feature_names: Sequence[str]) -> pd.DataFrame:
    """
    Последовательно применяет билдера фич по списку.
    Каждый билдер возвращает DF с ключами [user_id, item_id] и колонками своей фичи.
    """
    autodiscover()  # подхватываем все feat_*.py
    reg = select(list(feature_names))
    res = FeatureResources(ctx=ctx)

    out = df_base.copy()
    for fname, (spec, builder) in reg.items():
        part = builder(out[[COL_USER, COL_ITEM]].copy(), ctx, res)
        if not {COL_USER, COL_ITEM}.issubset(part.columns):
            raise ValueError(f"Feature '{fname}' must return [{COL_USER}, {COL_ITEM}, ...]")
        out = out.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)

    # NaN-safe и приведение типов: float→float32, int→без NaN
    for c in out.columns:
        if c in (COL_USER, COL_ITEM):
            continue
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].fillna(0.0).astype("float32")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].fillna(0)
    return out


def _label_and_group(df: pd.DataFrame, val_truth: Dict[int, set]) -> pd.DataFrame:
    # label = 1, если item ∈ вал-истине юзера
    lab = np.fromiter(
        (1 if int(i) in val_truth.get(int(u), set()) else 0 for u, i in zip(df[COL_USER].values, df[COL_ITEM].values)),
        count=len(df),
        dtype=np.int8,
    )
    df = df.copy()
    df["label"] = lab
    df["group_id"] = df[COL_USER].astype("int64")
    return df


# -----------------------------
# negative sampling
# -----------------------------

def _downsample_negatives(df_labeled: pd.DataFrame, max_neg_per_pos: Optional[int], seed: int = 42) -> pd.DataFrame:
    if not max_neg_per_pos or max_neg_per_pos <= 0:
        return df_labeled
    rng = np.random.default_rng(seed)
    parts = []
    for u, g in df_labeled.groupby(COL_USER, sort=False):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]
        keep_neg = neg
        if len(pos) > 0 and len(neg) > 0:
            limit = max_neg_per_pos * len(pos)
            if len(neg) > limit:
                idx = rng.choice(neg.index.values, size=limit, replace=False)
                keep_neg = neg.loc[idx]
        parts.append(pd.concat([pos, keep_neg], axis=0))
    return pd.concat(parts, axis=0).reset_index(drop=True)


# -----------------------------
# main entry
# -----------------------------

def build_matrix(
    ctx: dict,
    cand_map: Mapping[int, List[int]],
    feature_names: Sequence[str],
    out_tag: str = FEATURES_VERSION,
    max_neg_per_pos: Optional[int] = None,
    seed: int = 42,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, dict]]:
    """
    Собирает train/val матрицы по выбранным фичам и пулу кандидатов.

    В текущей схеме обе матрицы строятся на одном и том же пуле (вал-пользователи),
    label считается по ctx["val_truth"]. Это MVP для LambdaMART на валидации.
    При желании можно расширить: отдельный train-пул по историческому next-click.

    Возвращает: (train_mat, val_mat, norm_stats)
    """
    # 0) база из cand_map
    base = _cand_map_to_df(cand_map)
    if base.empty:
        raise ValueError("cand_map is empty — нет кандидатов для сборки матриц")

    # 1) считаем фичи
    feats = _apply_features(base, ctx, feature_names)

    # 2) лейблы/группы (по вал-истине)
    val_truth: Dict[int, set] = ctx["val_truth"]
    mat_val = _label_and_group(feats, val_truth)
    mat_train = mat_val.copy()

    # 3) (опц.) даунсэмплим негативы для train
    mat_train = _downsample_negatives(mat_train, max_neg_per_pos=max_neg_per_pos, seed=seed)

    # 4) нормализация — fit на train, apply на обе (через features.schema)
    feature_cols = [c for c in mat_train.columns if c not in (COL_USER, COL_ITEM, "label", "group_id")]
    # предварительно привести типы по хинтам
    mat_train = coerce_dtypes(mat_train, feature_cols)
    mat_val = coerce_dtypes(mat_val, feature_cols)

    norm_stats = fit_norm_stats(mat_train, feature_cols)
    mat_train = apply_norm(mat_train, norm_stats)
    mat_val = apply_norm(mat_val, norm_stats)

    # 5) сохраняем
    out_dir = ensure_dir(PATHS.artifact_dir / "features" / out_tag)
    if save_artifacts:
        save_df(out_dir / "train_matrix.parquet", mat_train)
        save_df(out_dir / "val_matrix.parquet", mat_val)
        save_schema(out_dir / "feature_schema.json", norm_stats, version=out_tag)

    return mat_train, mat_val, norm_stats
