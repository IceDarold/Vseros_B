"""Feature builders that operate on candidate lists and interaction logs."""

from __future__ import annotations

from functools import reduce
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from ..config import COL_DATE, COL_ITEM, COL_USER


def candidate_rank_features(
    candidate_maps: Mapping[str, Mapping[int, Sequence[int]]],
    max_rank: int = 300,
) -> Dict[str, pd.DataFrame]:
    """Transform candidate maps into per-source reciprocal-rank features."""
    frames: Dict[str, pd.DataFrame] = {}
    limit = max(1, int(max_rank))

    for source, mapping in candidate_maps.items():
        rows = []
        col_rr = f"{source}_rr"
        col_rank = f"{source}_rank"
        for user, items in mapping.items():
            for rank, item_id in enumerate(items[:limit], start=1):
                rows.append(
                    {
                        COL_USER: int(user),
                        COL_ITEM: int(item_id),
                        col_rr: 1.0 / float(rank),
                        col_rank: float(rank),
                    }
                )
        frames[source] = pd.DataFrame(rows) if rows else pd.DataFrame(columns=[COL_USER, COL_ITEM, col_rr, col_rank])
    return frames


def merge_feature_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge a sequence of feature frames on (user, item)."""
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=[COL_USER, COL_ITEM])
    return reduce(lambda left, right: left.merge(right, on=[COL_USER, COL_ITEM], how="outer"), frames)


def aggregate_interaction_features(
    interactions: pd.DataFrame,
    windows: Sequence[int] = (7, 14, 30),
) -> Dict[str, pd.DataFrame]:
    """Aggregate historical interaction counts for different time windows."""
    if interactions.empty:
        return {}

    date_max = int(interactions[COL_DATE].max())
    frames: Dict[str, pd.DataFrame] = {}
    for window in windows:
        window = int(window)
        left = date_max - window + 1
        scope = interactions[interactions[COL_DATE] >= left]
        agg = (
            scope.groupby([COL_USER, COL_ITEM], as_index=False)
            .size()
            .rename(columns={"size": f"hist_clicks_{window}d"})
        )
        frames[f"hist_{window}d"] = agg
    return frames
