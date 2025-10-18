# -*- coding: utf-8 -*-
"""
feat_lgcn.py — фича из LightGCN:
  - lgcn_dot: скалярное произведение эмбеддингов (user,item) из импортированных весов exp106.

Требования к ресурсам:
  FeatureResources.ensure_lgcn() -> dict {
      "user_vecs": np.ndarray [U, D] float32 (L2-норм. или как есть — не критично),
      "item_vecs": np.ndarray [I, D] float32,
      "user_map" : DataFrame [[user_id, "u_idx"]],
      "item_map" : DataFrame [[item_id, "i_idx"]],
  }
Если ресурса нет — возвращаем нули.

Никаких обращений к валидации — всё по train-обученным эмбеддингам.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from vseros_b.config import COL_USER, COL_ITEM
from .base import FeatureSpec
from .registry import register_feature
from .resources import FeatureResources


SPEC = FeatureSpec(
    name="lgcn_dot",
    cols=["lgcn_dot"],
    doc="Скалярное произведение эмбеддингов LightGCN: <e_u, e_i>",
)


@register_feature(SPEC)
def build_lgcn_dot(cand_df: pd.DataFrame, ctx: dict, res: FeatureResources) -> pd.DataFrame:
    assert {COL_USER, COL_ITEM}.issubset(cand_df.columns), "cand_df must contain user_id,item_id"

    lg = res.ensure_lgcn()
    out = cand_df[[COL_USER, COL_ITEM]].copy()

    if not lg:
        out["lgcn_dot"] = np.float32(0.0)
        return out

    # маппинги id -> индексы
    u_map = lg["user_map"].set_index(COL_USER)["u_idx"].to_dict()
    i_map = lg["item_map"].set_index(COL_ITEM)["i_idx"].to_dict()

    U = lg["user_vecs"]
    I = lg["item_vecs"]

    u_idx = out[COL_USER].map(u_map).fillna(-1).astype("int64").to_numpy()
    i_idx = out[COL_ITEM].map(i_map).fillna(-1).astype("int64").to_numpy()

    d = np.zeros(len(out), dtype=np.float32)
    mask = (u_idx >= 0) & (i_idx >= 0)
    if mask.any():
        d[mask] = np.einsum("ij,ij->i", U[u_idx[mask]], I[i_idx[mask]]).astype(np.float32)

    out["lgcn_dot"] = d
    return out
