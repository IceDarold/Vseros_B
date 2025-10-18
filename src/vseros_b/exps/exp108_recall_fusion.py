# -*- coding: utf-8 -*-
"""
exp108_recall_fusion.py — Stage 1 / Эксперимент 108: слияние candidate-листов (recall fusion)

Что делает:
- Подтягивает candidate-листы из набора источников (exp101…107), формат parquet: [user_id, items="i1 i2 ..."].
- Превращает ранги в скор (схемы: reciprocal-rank / linear), применяет веса источников, суммирует и дедуплицирует.
- (Опц.) Ограничивает вклад каждого источника per-user (cap_per_source).
- (Опц.) Бэкфилл пустых пользователей глобальным топом (по TRAIN популяронсти).
- Считает recall@M на валидации и ливает метрики/артефакты.

Артефакты:
- artifacts/candidates/exp108_recall_fusion/val_candidates_fused_M<M>_scheme-<sch>_w-<hash>.parquet
- metrics/exp108_recall_fusion.csv
- (опц.) submissions/sub_exp108_fusion_k20.csv (если вызван predict_submission)

Совместимость:
- Требует разово собрать candidate-листы из предыдущих эксп (101–107) на валидации.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Literal, Union
import hashlib
import json

import numpy as np
import pandas as pd

from ..base_exp import BaseExperiment
from ..config import (
    PATHS, COL_USER, COL_ITEM,
    CAND_TOP_M_PER_USER, QUICK_MODE, QUICK_USERS, SEED,
)
from ..artifacts import ensure_dir, save_df, log_artifact
from ..metrics import recall_at_m
from ..pop_decay import compute_pop_static, build_global_top


# ----------------------------- Утилиты загрузки/парсинга -----------------------------

def _read_candidates_parquet(path: Union[str, Path]) -> Dict[int, List[int]]:
    """
    Ожидаемый формат parquet: [user_id, items="i1 i2 ..."].
    Возвращает dict: user_id → [item_id, ...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates parquet not found: {p}")
    df = pd.read_parquet(p)
    if "items" not in df.columns or COL_USER not in df.columns:
        raise ValueError(f"Invalid candidates parquet schema at {p}. Need columns [{COL_USER}, 'items'].")
    out: Dict[int, List[int]] = {}
    for _, r in df[[COL_USER, "items"]].iterrows():
        if isinstance(r["items"], str):
            items = [int(x) for x in r["items"].split()]
        elif isinstance(r["items"], (list, tuple, np.ndarray)):
            items = list(map(int, r["items"]))
        else:
            items = []
        out[int(r[COL_USER])] = items
    return out


def _weights_hash(weights: Mapping[str, float]) -> str:
    s = json.dumps({k: float(v) for k, v in sorted(weights.items())}, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:6]


# ----------------------------- Конфиг и состояние -----------------------------

@dataclass
class SourceSpec:
    name: str
    path: Union[str, Path]       # путь к parquet с колонками [user_id, items="..."]


@dataclass
class Exp108Config:
    sources: Sequence[SourceSpec] = field(default_factory=list)  # список источников
    weights: Mapping[str, float] = field(default_factory=dict)   # веса источников по name (по умолчанию 1.0)

    score_scheme: Literal["rr", "linear"] = "rr"  # как конвертить ранг→скор
    cap_per_source: Optional[int] = None          # макс. кандидатов с одного источника для пользователя (до суммирования)

    # итоговые кандидаты
    m_per_user: int = int(CAND_TOP_M_PER_USER)

    # оценка
    eval_M_list: Sequence[int] = (200, 500, 1000)

    # бэкфилл
    backfill_with_global_top: bool = True
    backfill_K: int = 2000

    # удобства
    quick_mode: bool = QUICK_MODE
    quick_users: Optional[int] = QUICK_USERS
    seed: int = SEED


@dataclass
class Exp108State:
    src_maps: Dict[str, Dict[int, List[int]]]         # name → (user→items)
    fused_map: Optional[Dict[int, List[int]]]
    metrics_recall: Optional[pd.DataFrame]
    out_dir: Path
    cand_dir: Path
    fusion_tag: str                                    # для имени файлов (схема+веса)
    per_source_metrics: Optional[pd.DataFrame]


# ----------------------------- Реализация эксперимента -----------------------------

class Exp108RecallFusion(BaseExperiment):
    def __init__(self, cfg: Optional[Exp108Config] = None):
        super().__init__(exp_name="exp108_recall_fusion")
        self.cfg = cfg or Exp108Config()
        self.out_dir = ensure_dir(PATHS.artifact_dir / self.exp_name)
        self.cand_dir = ensure_dir(PATHS.cand_dir / self.exp_name)
        self.metrics_path = PATHS.metrics_dir / f"{self.exp_name}.csv"
        self.state: Optional[Exp108State] = None

    # ---- fit: читаем источники, возможно посчитаем индивидуальные метрики ----
    def fit(self, context: dict) -> Exp108State:
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])
        val_truth: Mapping[int, set] = context["val_truth"]
        rng = np.random.default_rng(self.cfg.seed)

        # загрузка кандидатов из всех источников
        src_maps: Dict[str, Dict[int, List[int]]] = {}
        for spec in self.cfg.sources:
            try:
                m = _read_candidates_parquet(spec.path)
                src_maps[spec.name] = m
            except Exception as e:
                print(f"[exp108] Warning: failed to read source {spec.name} at {spec.path}: {e}")

        # quick — можем ограничить пользователей для пер-источниковых метрик
        users = list(val_truth.keys())
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        # Метрики per-source (на всякий случай)
        per_src_rows = []
        for name, cmap in src_maps.items():
            # ограничить cmap до users
            cmap_sub = {int(u): cmap.get(int(u), []) for u in users}
            for m in self.cfg.eval_M_list:
                r = recall_at_m({u: val_truth[u] for u in users if u in val_truth}, cmap_sub, m=int(m), averaging="micro")
                per_src_rows.append({"source": name, "M": int(m), "recall": float(r)})

        per_src_df = pd.DataFrame(per_src_rows) if per_src_rows else None
        if per_src_df is not None and not per_src_df.empty:
            self.wandb_log_table("exp108_per_source", per_src_df)

        # тег для файлов — схема и хэш весов
        w_hash = _weights_hash(self.cfg.weights) if self.cfg.weights else "w1"
        fusion_tag = f"{self.cfg.score_scheme}_{w_hash}"

        self.state = Exp108State(
            src_maps=src_maps,
            fused_map=None,
            metrics_recall=None,
            out_dir=self.out_dir,
            cand_dir=self.cand_dir,
            fusion_tag=fusion_tag,
            per_source_metrics=per_src_df,
        )
        return self.state

    # ---- candidates: собственно слияние ----
    def candidates(
        self,
        context: dict,
        users: Optional[Sequence[int]] = None,
        M: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_truth"])

        train_df: pd.DataFrame = context["train_df"]
        val_truth: Mapping[int, set] = context["val_truth"]
        rng = np.random.default_rng(self.cfg.seed)

        # список юзеров
        if users is None:
            users = list(val_truth.keys())

        # quick
        if self.cfg.quick_mode and self.cfg.quick_users and len(users) > self.cfg.quick_users:
            users = list(rng.choice(users, size=self.cfg.quick_users, replace=False))

        # подготовим бэкфилл-глобалтоп при необходимости
        global_top: Optional[List[int]] = None
        if self.cfg.backfill_with_global_top:
            pop = compute_pop_static(train_df)
            global_top = build_global_top(pop, K=max(self.cfg.backfill_K, M or self.cfg.m_per_user))

        # веса источников
        w = {name: float(self.cfg.weights.get(name, 1.0)) for name in self.state.src_maps.keys()}
        scheme = self.cfg.score_scheme
        cap = int(self.cfg.cap_per_source) if self.cfg.cap_per_source else None

        fused: Dict[int, List[int]] = {}
        Mfin = int(M or self.cfg.m_per_user)

        for u in users:
            # собираем из всех источников
            bucket: Dict[int, float] = {}
            any_added = False
            for name, cmap in self.state.src_maps.items():
                items = cmap.get(int(u), [])
                if not items:
                    continue
                any_added = True
                if cap is not None and len(items) > cap:
                    items = items[:cap]
                # начислим скор по схеме
                if scheme == "rr":
                    # 1/r
                    sc = [1.0 / float(r) for r in range(1, len(items) + 1)]
                elif scheme == "linear":
                    L = len(items)
                    sc = [float(L - r + 1) / float(L) for r in range(1, L + 1)]
                else:
                    raise ValueError(f"Unknown score_scheme: {scheme}")
                # с весом источника
                ww = w.get(name, 1.0)
                for it, s in zip(items, sc):
                    bucket[int(it)] = bucket.get(int(it), 0.0) + ww * float(s)

            # если из источников ничего не пришло — подложим глобалтоп
            if (not any_added) and global_top is not None:
                fused[int(u)] = list(map(int, global_top[:Mfin]))
                continue

            if not bucket:
                fused[int(u)] = []
                continue

            # финальная сортировка
            ordered = sorted(bucket.items(), key=lambda kv: (-kv[1], kv[0]))
            fused[int(u)] = [it for it, _ in ordered[:Mfin]]

        # обновим state (важно — не затирать, а дополнять)
        if self.state.fused_map is None:
            self.state.fused_map = {}
        self.state.fused_map.update(fused)
        return fused

    # ---- evaluate: recall@M ----
    def evaluate(self, context: dict) -> pd.DataFrame:
        assert self.state is not None, "Run fit() first."
        self.require_context_keys(context, ["val_truth"])
        val_truth: Mapping[int, set] = context["val_truth"]

        if self.state.fused_map is None:
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        rows = []
        for m in self.cfg.eval_M_list:
            r = recall_at_m(val_truth, self.state.fused_map, m=int(m), averaging="micro")
            rows.append({"M": int(m), "recall": float(r)})
        metrics_df = pd.DataFrame(rows)

        # лог + сохранение
        self.wandb_log_table("exp108_recall_fused", metrics_df)
        self.save_metrics_df(metrics_df, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")

        # сохраним и в state
        self.state.metrics_recall = metrics_df
        return metrics_df

    # ---- save: кандидаты + метрики ----
    def save(self, context: dict):
        assert self.state is not None, "Run fit() first."
        # метрики уже сохранены в evaluate(); повторим на всякий случай
        if self.state.metrics_recall is not None:
            self.save_metrics_df(self.state.metrics_recall, filename=f"{self.exp_name}.csv", artifact_name=f"{self.exp_name}_metrics")

        # кандидаты
        if self.state.fused_map is None:
            val_truth: Mapping[int, set] = context["val_truth"]
            _ = self.candidates(context, users=list(val_truth.keys()), M=self.cfg.m_per_user)

        tag = self.state.fusion_tag
        cand_path = self.save_candidates_map(
            self.state.fused_map,
            filename=f"val_candidates_fused_M{self.cfg.m_per_user}_scheme-{self.cfg.score_scheme}_{tag}.parquet",
        )
        return cand_path, self.metrics_path

    # ---- submission (опционально) ----
    def predict_submission(self, context: dict, sample_df: pd.DataFrame, k_top: int = 20) -> pd.DataFrame:
        """
        Делает сабмит из уже готовой fused_map (на user_id).
        Если каких-то пользователей нет — добивает глобальным топом.
        """
        assert self.state is not None, "Run fit() first."
        train_df: pd.DataFrame = context["train_df"]

        # глобалтоп
        pop = compute_pop_static(train_df)
        global_top = build_global_top(pop, K=max(k_top, 1000))

        # готовим вывод
        tmp = sample_df.copy()
        tmp["rank"] = tmp.groupby(COL_USER).cumcount()

        # соберём массив item_id по пользователю
        out_items: Dict[int, List[int]] = {}
        if self.state.fused_map is None:
            raise RuntimeError("call candidates() before predict_submission().")

        for u in tmp[COL_USER].unique():
            cand = self.state.fused_map.get(int(u), [])
            if not cand:
                cand = global_top
            out_items[int(u)] = list(map(int, cand[:k_top]))

        # проставим по рангу
        def _fill_row(row):
            u = int(row[COL_USER])
            r = int(row["rank"])
            return out_items[u][r]

        tmp[COL_ITEM] = tmp.apply(_fill_row, axis=1)
        tmp = tmp.drop(columns=["rank"]).reset_index(drop=True)

        sub_dir = ensure_dir(PATHS.sub_dir)
        tag = self.state.fusion_tag
        sub_path = sub_dir / f"sub_exp108_fusion_{self.cfg.score_scheme}_{tag}_k{k_top}.csv"
        save_df(sub_path, tmp, index=False)
        log_artifact(sub_path, name=f"{self.exp_name}_submission_{tag}", type_="submission")
        return tmp
