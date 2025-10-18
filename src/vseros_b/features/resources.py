# src/vseros_b/features/resources.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd

from vseros_b.config import PATHS, COL_USER, COL_ITEM, COL_DATE
from vseros_b.pop_decay import compute_pop_static
from vseros_b.lightgcn_import import load_embeddings, ImportConfig
# опционально: i2v и ppr
try:
    from vseros_b.item2vec_lite import load_i2v_model
except Exception:
    load_i2v_model = None
try:
    from vseros_b.ppr import load_ppr_cache  # сделай функцию, которая читает кеш user→{item:score}
except Exception:
    load_ppr_cache = None

@dataclass
class FeatureResources:
    ctx: dict
    cache_dir: Path = PATHS.artifact_dir / "cache"

    _lgcn: Optional[dict] = None
    _i2v: Optional[Any] = None
    _ppr: Optional[dict] = None
    _item_pop: Optional[pd.DataFrame] = None
    _user_hist: Optional[dict] = None
    _markov: Optional[dict] = None
    _covis: Optional[dict] = None

    # ---- LightGCN ----
    def ensure_lgcn(self, import_cfg: Optional[ImportConfig] = None) -> dict:
        if self._lgcn is None:
            emb = load_embeddings(import_cfg or ImportConfig(use_faiss=False))
            self._lgcn = {
                "user_vecs": emb.user_vecs.astype(np.float32),
                "item_vecs": emb.item_vecs.astype(np.float32),
                "user_map": emb.user_map[[COL_USER, "u_idx"]],
                "item_map": emb.item_map[[COL_ITEM, "i_idx"]],
            }
        return self._lgcn

    # ---- item2vec ----
    def ensure_i2v(self, path: Optional[Path] = None):
        if self._i2v is None and load_i2v_model is not None:
            self._i2v = load_i2v_model(path)
        return self._i2v

    # ---- PPR ----
    def ensure_ppr_cache(self) -> dict:
        if self._ppr is None:
            self._ppr = load_ppr_cache(self.cache_dir) if load_ppr_cache else {}
        return self._ppr

    # ---- Популярность item ----
    def ensure_item_pop(self) -> pd.DataFrame:
        if self._item_pop is None:
            train_df: pd.DataFrame = self.ctx["train_df"]
            pop = compute_pop_static(train_df)   # вернёт item_id → cnt
            self._item_pop = pop
        return self._item_pop

    # ---- История пользователя ----
    def ensure_user_history(self, lastK: int = 10) -> dict:
        """
        Возвращает словарь:
          {
            "lastK": Dict[user_id, List[item_id]],
            "u_last_day": Dict[user_id, int],
            "u_i_cnt": Dict[Tuple[u,i], int],
            "u_clicks_7/14/28": Dict[user_id, int],
          }
        """
        if self._user_hist is None:
            train_df: pd.DataFrame = self.ctx["train_df"]
            # lastK
            g = (train_df.sort_values([COL_USER, COL_DATE])
                        .groupby(COL_USER)[COL_ITEM]
                        .apply(list))
            lastK_map = {int(u): [int(x) for x in lst[-lastK:]] for u, lst in g.items()}
            # u_last_day
            ul = train_df.groupby(COL_USER)[COL_DATE].max().astype(int).to_dict()
            # u_i_cnt
            uic = (train_df.groupby([COL_USER, COL_ITEM]).size().astype(int))
            uic_map = {(int(u), int(i)): int(c) for (u, i), c in uic.items()}
            # u_clicks windows
            maxd = int(train_df[COL_DATE].max())
            def cnt_in_win(w):
                dfw = train_df[train_df[COL_DATE] >= maxd - w + 1]
                return dfw.groupby(COL_USER).size().astype(int).to_dict()
            u7, u14, u28 = cnt_in_win(7), cnt_in_win(14), cnt_in_win(28)

            self._user_hist = {
                "lastK": lastK_map,
                "u_last_day": ul,
                "u_i_cnt": uic_map,
                "u_clicks_7": u7,
                "u_clicks_14": u14,
                "u_clicks_28": u28,
            }
        return self._user_hist

    # ---- Марковки / Co-Vis (заглушки; заполни своими генераторами) ----
    def ensure_markov(self) -> dict:
        if self._markov is None:
            self._markov = {}  # {prev_item: {next_item: prob}}
        return self._markov

    def ensure_covis(self) -> dict:
        if self._covis is None:
            self._covis = {}   # {item: {neighbor_item: weight}}
        return self._covis
