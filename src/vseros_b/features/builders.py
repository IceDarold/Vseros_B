"""Utilities to assemble feature matrices for ranking models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import COL_ITEM, COL_USER
from .schema import FeatureSchema


@dataclass
class FeatureMatrixBuilder:
    """Combines per-source feature frames into a single training matrix."""

    schema: FeatureSchema
    index_cols: Tuple[str, str] = (COL_USER, COL_ITEM)
    fill_value: float = 0.0

    def build(
        self,
        feature_frames: Mapping[str, pd.DataFrame],
        drop_missing: bool = False,
    ) -> pd.DataFrame:
        """Merge all frames from *feature_frames* according to the schema order."""
        missing = set(self.schema.names()) - set(feature_frames.keys())
        if missing:
            raise ValueError(f"Missing feature frames for schema {self.schema.version}: {sorted(missing)}")

        ordered = [feature_frames[name].copy() for name in self.schema.names()]
        matrix = ordered[0]
        for frame in ordered[1:]:
            matrix = matrix.merge(frame, on=self.index_cols, how="outer")

        if drop_missing:
            matrix = matrix.dropna(subset=list(self.schema.names()))
        else:
            matrix = matrix.fillna(self.fill_value)

        return matrix.sort_values(list(self.index_cols)).reset_index(drop=True)

    @staticmethod
    def normalise_columns(
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
        """Z-score normalisation for the selected feature columns."""
        cols = list(columns or [c for c in df.columns if c not in {COL_USER, COL_ITEM}])
        if not cols:
            return df
        stats = {}
        for col in cols:
            mu = df[col].mean()
            sigma = df[col].std(ddof=0)
            stats[col] = (mu, max(sigma, eps))
            df[col] = (df[col] - mu) / stats[col][1]
        return df

    @staticmethod
    def split_train_val(
        df: pd.DataFrame,
        train_items: Iterable[int],
        val_items: Iterable[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_mask = df[COL_ITEM].isin(list(train_items))
        val_mask = df[COL_ITEM].isin(list(val_items))
        return df.loc[train_mask].copy(), df.loc[val_mask].copy()
