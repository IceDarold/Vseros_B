"""Helpers for working with embedding-based features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..candidates.item2vec_lite import (
    build_neighbors_from_embeddings,
    df_to_embeddings,
    embeddings_to_df,
)

__all__ = [
    "df_to_embeddings",
    "embeddings_to_df",
    "load_embedding_table",
    "save_embedding_table",
    "build_neighbors_from_embeddings",
]


def load_embedding_table(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load embedding table stored as parquet."""
    df = pd.read_parquet(path)
    return df_to_embeddings(df)


def save_embedding_table(path: Path, item_ids: np.ndarray, emb: np.ndarray) -> Path:
    """Persist embedding vectors to parquet."""
    df = embeddings_to_df(item_ids, emb)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def l2_normalise(emb: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return emb / norms
