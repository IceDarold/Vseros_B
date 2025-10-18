# -*- coding: utf-8 -*-
"""
covis2_gpu.py — GPU-ускоренный расчёт 2-hop ко-визитов для exp104.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _require_rapids():
    try:
        import cudf  # type: ignore
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "RAPIDS (cudf) is required for GPU two-hop pipeline. "
            "Install cudf or set use_gpu=False."
        ) from exc
    return cudf


def build_two_hop_gpu(
    neighbors1_df: pd.DataFrame,
    gamma: float = 0.5,
    branch_topn: int = 50,
    topn_second: int = 300,
    drop_self: bool = True,
) -> pd.DataFrame:
    """
    GPU аналог _build_two_hop: возвращает pandas DataFrame[item_id, neighbor_id, score].
    """
    cudf = _require_rapids()

    if neighbors1_df is None or neighbors1_df.empty:
        return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype(
            {"item_id": "int64", "neighbor_id": "int64", "score": "float64"}
        )

    g = cudf.DataFrame.from_pandas(neighbors1_df[["item_id", "neighbor_id", "score"]])

    # src -> mid (ограничиваем top branch_topn)
    A = g.rename(columns={"item_id": "src", "neighbor_id": "mid", "score": "s1"})
    A = A.sort_values(["src", "s1"], ascending=[True, False])
    A["rank_mid"] = A.groupby("src").cumcount()
    A = A[A["rank_mid"] < int(branch_topn)]
    A = A.drop(columns=["rank_mid"])

    # mid -> dst
    B = g.rename(columns={"item_id": "mid", "neighbor_id": "dst", "score": "s2_raw"})

    # join по mid
    M = A.merge(B, on="mid", how="inner")
    if "dst" not in M.columns or M.empty:
        return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype(
            {"item_id": "int64", "neighbor_id": "int64", "score": "float64"}
        )
    M = M.dropna(subset=["dst", "s2_raw"])

    if drop_self:
        M = M[M["src"] != M["dst"]]

    if M.empty:
        return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype(
            {"item_id": "int64", "neighbor_id": "int64", "score": "float64"}
        )

    M["score2"] = float(gamma) * M["s1"].astype("float64") * M["s2_raw"].astype("float64")

    agg = M.groupby(["src", "dst"]).agg({"score2": "sum"}).reset_index()
    agg = agg.sort_values(["src", "score2"], ascending=[True, False])
    agg["rank"] = agg.groupby("src").cumcount()
    agg = agg[agg["rank"] < int(topn_second)].drop(columns=["rank"])

    out = agg.rename(columns={"src": "item_id", "dst": "neighbor_id", "score2": "score"})
    out_pd = out.to_pandas()
    out_pd["item_id"] = out_pd["item_id"].astype("int64")
    out_pd["neighbor_id"] = out_pd["neighbor_id"].astype("int64")
    out_pd["score"] = out_pd["score"].astype("float64")
    return out_pd.reset_index(drop=True)

