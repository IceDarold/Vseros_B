# -*- coding: utf-8 -*-
"""
covis_gpu.py — GPU-ускоренные варианты ко-визитов с помощью RAPIDS (cuDF/cuGraph/cuPy).

Выполняет наиболее тяжёлые операции (подсчёт support и пар) на GPU и возвращает
результат в виде pandas DataFrame, совместимый с CPU-пайплайном.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import COL_DATE, COL_ITEM, COL_USER
from .covis import (
    CoVisConfig,
    score_pairs,
    topn_neighbors_from_pairs,
)


def _require_rapids() -> tuple:
    try:
        import cudf  # type: ignore
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "RAPIDS (cudf/cupy) is required for GPU co-vis pipeline. "
            "Install cudf and cupy or set use_gpu=False."
        ) from exc
    return cudf, cp


def _prepare_basket_gpu(basket_items: pd.DataFrame, cfg: CoVisConfig):
    cudf, cp = _require_rapids()

    needed_cols = [COL_USER, COL_DATE, COL_ITEM]
    if not set(needed_cols).issubset(basket_items.columns):
        raise ValueError(f"basket_items must contain columns {needed_cols}")

    gdf = cudf.DataFrame.from_pandas(basket_items[needed_cols])
    gdf = gdf.drop_duplicates(subset=needed_cols)

    if gdf.empty:
        return gdf, None

    ref_day = cfg.ref_day if cfg.ref_day is not None else int(gdf[COL_DATE].max())
    if cfg.decay_lambda > 0:
        age = (ref_day - gdf[COL_DATE]).clip(lower=0).astype("float32")
        gdf["weight"] = cp.exp(-float(cfg.decay_lambda) * age.values)
    else:
        gdf["weight"] = 1.0

    return gdf, ref_day


def _compute_support_gpu(gdf, cfg: CoVisConfig) -> pd.Series:
    cudf, _ = _require_rapids()
    if gdf.empty:
        return pd.Series(dtype="float64")

    if cfg.decay_lambda > 0 and cfg.weighted_support:
        support = gdf.groupby(COL_ITEM)["weight"].sum().rename("support_w")
    else:
        support = gdf.groupby(COL_ITEM).size().rename("support_cnt")
    support_pd = support.to_pandas()
    return support_pd.astype("float64") if support_pd.dtype != np.int64 else support_pd


def _compute_pairs_gpu(gdf, cfg: CoVisConfig) -> pd.DataFrame:
    cudf, _ = _require_rapids()
    if gdf.empty:
        return pd.DataFrame(columns=["item_i", "item_j", "co_cnt", "co_w"]).astype(
            {"item_i": "int64", "item_j": "int64", "co_cnt": "int64", "co_w": "float64"}
        )

    joined = gdf.merge(
        gdf,
        on=[COL_USER, COL_DATE],
        suffixes=("_i", "_j"),
        how="inner",
    )
    mask = joined["item_id_i"] < joined["item_id_j"]
    pairs = joined.loc[mask, ["item_id_i", "item_id_j", "weight_i"]]
    if pairs.empty:
        return pd.DataFrame(columns=["item_i", "item_j", "co_cnt", "co_w"]).astype(
            {"item_i": "int64", "item_j": "int64", "co_cnt": "int64", "co_w": "float64"}
        )

    pairs["co_cnt"] = 1
    agg = (
        pairs.groupby(["item_id_i", "item_id_j"])
        .agg({"co_cnt": "sum", "weight_i": "sum"})
        .reset_index()
    )
    agg = agg.rename(columns={"item_id_i": "item_i", "item_id_j": "item_j", "weight_i": "co_w"})

    # В случае λ=0 вес совпадает с количеством — приведём к float64 для совместимости
    agg_pd = agg.to_pandas()
    agg_pd["co_cnt"] = agg_pd["co_cnt"].astype("int64")
    agg_pd["co_w"] = agg_pd["co_w"].astype("float64")
    agg_pd["item_i"] = agg_pd["item_i"].astype("int64")
    agg_pd["item_j"] = agg_pd["item_j"].astype("int64")
    return agg_pd


def build_neighbors_gpu(
    basket_items: pd.DataFrame,
    cfg: Optional[CoVisConfig] = None,
) -> pd.DataFrame:
    """
    GPU-ускоренный аналог build_neighbors().
    Выполняет подсчёт support и пар в RAPIDS, после чего возвращает pandas DataFrame.
    """
    cfg = cfg or CoVisConfig()
    cudf, _ = _require_rapids()

    gdf, ref_day = _prepare_basket_gpu(basket_items, cfg)
    if gdf.empty:
        return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype(
            {"item_id": "int64", "neighbor_id": "int64", "score": "float64"}
        )

    # Обновляем ref_day в конфиге, чтобы downstream понимал, какое значение использовано
    cfg = CoVisConfig(
        decay_lambda=cfg.decay_lambda,
        ref_day=ref_day,
        score=cfg.score,
        topn_per_item=cfg.topn_per_item,
        min_pair_count=cfg.min_pair_count,
        weighted_support=cfg.weighted_support,
    )

    support = _compute_support_gpu(gdf, cfg)
    pairs = _compute_pairs_gpu(gdf, cfg)

    scored = score_pairs(
        pairs_df=pairs,
        support=support,
        score=cfg.score,
        use_weighted=(cfg.decay_lambda > 0 and cfg.weighted_support),
    )

    neighbors = topn_neighbors_from_pairs(
        scored_pairs=scored,
        topn_per_item=cfg.topn_per_item,
        min_pair_count=cfg.min_pair_count,
        use_weighted=(cfg.decay_lambda > 0),
    )
    return neighbors

