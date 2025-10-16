# -*- coding: utf-8 -*-
"""
exp104_covis_2hop.py — Stage 1 / Эксперимент 104: Co-Vis 2-hop

Идея:
- Берём готовых 1-hop соседей (артефакт exp103: neighbors_topN_*.parquet).
- Строим 2-hop: i → mid → dst со скором s2 = gamma * s(i,mid) * s(mid,dst), агрегируем по (i,dst).
- Опционально смешиваем 1-hop и 2-hop: s_total = beta1 * s1 + s2.
- Из комбинированной карты соседей строим per-user кандидатов по последним K семенам.

Артефакты:
- artifacts/exp104_covis_2hop/neighbors2_topN_...parquet                  (2-hop соседи)
- artifacts/candidates/exp104_covis_2hop/val_candidates_covis2_...parquet  (per-user кандидаты)
- metrics/exp104_covis_2hop.csv

Требования к context:
- split, train_df, val_df, val_truth
- НЕ обязательно, но желательно: артефакт exp103 в файловой системе (см. _discover_neighbors1()).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
import glob
import os

from .base_exp import BaseExperiment
from .config import (
    PATHS, COL_USER, COL_ITEM, COL_DATE,
    TOPN_NEIGHBORS_PER_ITEM, K_RECENT_ITEMS_PER_USER, CAND_TOP_M_PER_USER,
    QUICK_MODE, QUICK_USERS, SEED,
)
from .artifacts import ensure_dir, save_df, log_artifact
from .metrics import recall_at_m
from .covis import (
    neighbors_to_map, last_k_items_by_user, candidates_from_covis,
)

# ----------------------------- Конфиг и состояние -----------------------------

@dataclass
class Exp104Config:
    # источники 1-hop
    neighbors1_path: Optional[Path] = None    # если None — попробуем найти в artifacts/exp103_covis_v1/*.parquet

    # построение 2-hop
    gamma: float = 0.5                        # множитель для 2-hop (s2 = gamma * s1 * s_mid_dst)
    branch_topn: int = 50                     # ограничение разветвления: с каждого i берём только top-B mid
    topn_second_order: int = 300              # сколько 2-hop соседей сохранять на item
    drop_self: bool = True                    # выбрасывать dst == i

    # смешивание
    include_first_order: bool = True          # включать 1-hop при формировании итоговых соседей
    beta_first_order: float = 1.0             # вес 1-hop в s_total = beta*s1 + s2
    topn_combined: int = TOPN_NEIGHBORS_PER_ITEM  # финальное topN на item после смешивания

    # генерация кандидатов
    k_recent: int = K_RECENT_ITEMS_PER_USER
    m_per_user: int = CAND_TOP_M_PER_USER
    exclude_seed: bool = True

    # оценка
    eval_M_list: Sequence[int] = (200, 500, 1000)

    # удобства
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED


@dataclass
class Exp104State:
    neighbors1_df: pd.DataFrame
    neighbors2_df: pd.DataFrame
    combined_neighbors_df: pd.DataFrame
    cand_map: Optional[Dict[int, List[int]]
                     ]
    metrics_recall: Optional[pd.DataFrame]
    out_dir: Path
    cand_dir: Path
    neighbors2_path: Path
    combined_path: Path


# ----------------------------- Реализация эксперимента -----------------------------

class Exp104CoVis2Hop(BaseExperiment):
    def __init__(self, cfg: Optional[Exp104Config] = None):
        super().__init__(exp_name="exp104_covis_2hop")
        self.cfg = cfg or Exp104Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.state: Optional[Exp104State] = None

    # ---- fit: грузим 1-hop, строим 2-hop, собираем combined ----
    def fit(self, context: dict) -> Exp104State:
        self.require_context_keys(context, ["train_df", "val_df", "split"])
        split = context["split"]

        nei1 = self._load_neighbors1()
        if nei1 is None or nei1.empty:
            raise FileNotFoundError("Не удалось найти артефакт 1-hop соседей (exp103). Убедись, что exp103 выполнен.")

        # 2-hop
        nei2 = self._build_two_hop(nei1, gamma=self.cfg.gamma,
                                   branch_topn=self.cfg.branch_topn,
                                   topn_second=self.cfg.topn_second_order,
                                   drop_self=self.cfg.drop_self)

        # смешиваем 1-hop и 2-hop
        comb = self._combine_nei(nei1, nei2,
                                 include_first=self.cfg.include_first_order,
                                 beta=self.cfg.beta_first_order,
                                 topn=self.cfg.topn_combined)

        # сохраняем
        neighbors2_path = self.out_dir / f"neighbors2_topN_{self.cfg.topn_second_order}_gamma{str(self.cfg.gamma).replace('.','')}.parquet"
        save_df(neighbors2_path, nei2, index=False)
        log_artifact(neighbors2_path, name=f"{self.exp_name}_neighbors2", type_="dataset")

        combined_path = self.out_dir / f"neighbors12_combined_topN_{self.cfg.topn_combined}_g{str(self.cfg.gamma).replace('.','')}_b{self.cfg.beta_first_order}.parquet"
        save_df(combined_path, comb, index=False)
        log_artifact(combined_path, name=f"{self.exp_name}_neighbors12_combined", type_="dataset")

        self.state = Exp104State(
            neighbors1_df=nei1,
            neighbors2_df=nei2,
            combined_neighbors_df=comb,
            cand_map=None,
            metrics_recall=None,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
            neighbors2_path=neighbors2_path,
            combined_path=combined_path,
        )
        return self.state

    # ---- candidates: используем комбинированных соседей ----
    def candidates(
        self,
        context: dict,
        users: Optional[Sequence[int]] = None,
        M: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])

        train_df: pd.DataFrame = context["train_df"]
        split = context["split"]
        val_truth: Mapping[int, set] = context["val_truth"]

        # список пользователей
        if users is None:
            users = list(val_truth.keys())

        rng = np.random.default_rng(self.cfg.seed)
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        # последние K айтемов
        seeds_full = last_k_items_by_user(train_df, k=int(self.cfg.k_recent), upto_day=int(split.train_end))
        seeds = {int(u): seeds_full.get(int(u), []) for u in users}

        # карта соседей (item -> [(nbr,score), ...]) из комбинированной таблицы
        nei_map = neighbors_to_map(self.state.combined_neighbors_df, keep_score=True)

        cand_map = candidates_from_covis(
            recent_items_by_user=seeds,
            neighbors_df_or_map=nei_map,
            M=int(M or self.cfg.m_per_user),
            exclude_seed=bool(self.cfg.exclude_seed),
        )

        if self.state.cand_map is None:
            self.state.cand_map = {}
        self.state.cand_map.update(cand_map)
        return cand_map

    # ---- evaluate: recall@M ----
    def evaluate(self, context: dict) -> pd.DataFrame:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["val_truth"])

        val_truth: Mapping[int, set] = context["val_truth"]
        if self.state.cand_map is None:
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        rows = []
        for m in self.cfg.eval_M_list:
            r = recall_at_m(val_truth, self.state.cand_map, m=int(m), averaging="micro")
            rows.append({"M": int(m), "recall": float(r)})
        metrics_df = pd.DataFrame(rows)

        self.wandb_log_table("exp104_recall", metrics_df)
        self.state.metrics_recall = metrics_df
        return metrics_df

    # ---- save: метрики + кандидаты ----
    def save(self, context: dict):
        assert self.state is not None, "Run fit() first."
        metrics_path = self.save_metrics_df(
            self.state.metrics_recall if self.state.metrics_recall is not None else pd.DataFrame(),
            filename=f"{self.exp_name}.csv",
            artifact_name=f"{self.exp_name}_metrics",
        )
        if self.state.cand_map is None:
            val_truth: Mapping[int, set] = context["val_truth"]
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        cand_path = self.save_candidates_map(
            self.state.cand_map,
            filename=f"val_candidates_covis2_K{self.cfg.k_recent}_M{self.cfg.m_per_user}_g{str(self.cfg.gamma).replace('.','')}_b{self.cfg.beta_first_order}.parquet",
        )
        return cand_path, metrics_path

    # ============================= ВСПОМОГАТЕЛЬНЫЕ =============================

    def _load_neighbors1(self) -> Optional[pd.DataFrame]:
        """
        Загружает 1-hop соседей из:
          1) cfg.neighbors1_path, если задан и существует
          2) иначе — ищем любой parquet в artifacts/exp103_covis_v1/ с префиксом 'neighbors_topN'
             и берём самый "тяжёлый" (обычно с наибольшим topN/полным набором).
        """
        # 1) задан явный путь
        if self.cfg.neighbors1_path is not None:
            p = Path(self.cfg.neighbors1_path)
            if p.exists():
                return pd.read_parquet(p)
            else:
                print(f"[exp104] Warning: neighbors1_path not found: {p}")

        # 2) автопоиск
        root = PATHS.artifact_dir / "exp103_covis_v1"
        candidates = sorted(glob.glob(str(root / "neighbors_topN_*.parquet")))
        if not candidates:
            return None
        # берём самый большой файл (по размеру), обычно он богаче
        best = max(candidates, key=lambda x: os.path.getsize(x))
        print(f"[exp104] Using neighbors1 from: {best}")
        return pd.read_parquet(best)

    @staticmethod
    def _build_two_hop(
        neighbors1_df: pd.DataFrame,
        gamma: float = 0.5,
        branch_topn: int = 50,
        topn_second: int = 300,
        drop_self: bool = True,
    ) -> pd.DataFrame:
        """
        Строит 2-hop для всех item:
            s2(i->dst) = sum_over_mid{ gamma * s1(i,mid) * s1(mid,dst) }
        Ограничиваем разветвление: на этапе i->mid берём top 'branch_topn'.
        Затем для каждого i оставляем top 'topn_second' по dst.

        Возвращает DataFrame: [item_id, neighbor_id, score] (2-hop компонента).
        """
        if neighbors1_df is None or neighbors1_df.empty:
            return pd.DataFrame(columns=["item_id", "neighbor_id", "score"]).astype({
                "item_id": "int64", "neighbor_id": "int64", "score": "float64"
            })

        # i->mid = A(src, mid, s1)
        A = neighbors1_df.rename(columns={"item_id": "src", "neighbor_id": "mid", "score": "s1"}).copy()
        A = A.sort_values(["src", "s1"], ascending=[True, False])
        A["rank_mid"] = A.groupby("src").cumcount()
        A = A[A["rank_mid"] < int(branch_topn)].drop(columns=["rank_mid"])

        # mid->dst = B(mid, dst, s2_raw)
        B = neighbors1_df.rename(columns={"item_id": "mid", "neighbor_id": "dst", "score": "s2_raw"}).copy()

        # соединяем по mid
        M = A.merge(B, on="mid", how="left")  # columns: src, mid, s1, dst, s2_raw
        M = M.dropna(subset=["dst", "s2_raw"])
        M["dst"] = M["dst"].astype("int64")
        M["score2"] = float(gamma) * M["s1"].astype("float64") * M["s2_raw"].astype("float64")

        if drop_self:
            M = M[M["src"] != M["dst"]]

        # агрегируем по (src, dst)
        agg = (M.groupby(["src", "dst"], as_index=False)["score2"].sum())
        agg = agg.sort_values(["src", "score2"], ascending=[True, False])
        # topN per src
        agg["rank"] = agg.groupby("src").cumcount()
        agg = agg[agg["rank"] < int(topn_second)].drop(columns=["rank"])
        out = agg.rename(columns={"src": "item_id", "dst": "neighbor_id", "score2": "score"})
        out["item_id"] = out["item_id"].astype("int64")
        out["neighbor_id"] = out["neighbor_id"].astype("int64")
        out["score"] = out["score"].astype("float64")
        return out.reset_index(drop=True)

    @staticmethod
    def _combine_nei(
        nei1: pd.DataFrame,
        nei2: pd.DataFrame,
        include_first: bool = True,
        beta: float = 1.0,
        topn: int = TOPN_NEIGHBORS_PER_ITEM,
    ) -> pd.DataFrame:
        """
        Смешивает 1-hop и 2-hop:
            s_total(i, j) = beta * s1(i, j) + s2(i, j)  (если нет компоненты — берём 0)
        Возвращает topN per item.
        """
        # приводим к общей форме
        a = nei2.rename(columns={"score": "s2"})  # 2-hop всегда есть
        if include_first and nei1 is not None and not nei1.empty:
            b = nei1.rename(columns={"score": "s1"})
            # merge outer
            m = pd.merge(a, b, on=["item_id", "neighbor_id"], how="outer")
        else:
            m = a
            m["s1"] = 0.0

        m["s1"] = m["s1"].fillna(0.0).astype("float64")
        m["s2"] = m["s2"].fillna(0.0).astype("float64")
        m["score"] = float(beta) * m["s1"] + m["s2"]

        # topN на айтем
        m = m.sort_values(["item_id", "score"], ascending=[True, False])
        m["rank"] = m.groupby("item_id").cumcount()
        m = m[m["rank"] < int(topn)].drop(columns=["rank", "s1", "s2"])
        m["item_id"] = m["item_id"].astype("int64")
        m["neighbor_id"] = m["neighbor_id"].astype("int64")
        m["score"] = m["score"].astype("float64")
        return m.reset_index(drop=True)