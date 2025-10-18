# -*- coding: utf-8 -*-
"""
exp107_ppr — Personalized PageRank на item-item графе (co-vis по train).

Что делает:
  • строит взвешенный граф соседства айтемов: ребра между айтемами, встречавшимися у одного юзера,
    по умолчанию только между соседними по времени взаимодействиями (pair_mode='consecutive'),
    с экспоненциальным затуханием по разнице дней;
  • нормализует строки (вероятности перехода);
  • для каждого пользователя строит персонализацию v из последних L айтемов (затухание по давности),
    и итеративно считает p = α v + (1-α) Pᵀ p (с отсечкой топ-N на каждом шаге);
  • формирует top-K кандидатов, исключая (по желанию) уже виденные и/или сидаовые айтемы;
  • сохраняет:
      artifacts/models/exp107_ppr/val_scores.parquet   [(user_id,item_id,score)]
      artifacts/models/exp107_ppr/test_scores.parquet  (если есть test users)
      artifacts/models/exp107_ppr/val_metrics.csv
      artifacts/candidates/exp107_ppr/val_candidates_top{K}.parquet  (compact map)
      artifacts/candidates/exp107_ppr/test_candidates_top{K}.parquet
  • predict_submission(sample_df): формирует сабмит из PPR-скоринга.

Примечание:
  • Реализация не зависит от внешнего ppr.py; всё локально и воспроизводимо.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict, Counter
import math
import numpy as np
import pandas as pd

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM, COL_DATE
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.candidates.io import load_compact_map
# compact saver (на случай, если где-то отсутствует)
try:
    from vseros_b.candidates.io import save_compact_map
except Exception:
    def save_compact_map(path: Path, user2items: Dict[int, List[int]]):
        rows = [(int(u), " ".join(map(str, lst))) for u, lst in user2items.items()]
        df = pd.DataFrame(rows, columns=[COL_USER, "items"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


# =========================
# Конфиг
# =========================

@dataclass
class Exp107Config:
    # co-vis построение
    pair_mode: str = "consecutive"   # 'consecutive' | 'all_pairs'
    time_decay_beta: float = 0.25    # затухание по |day_i - day_j|: exp(-beta * Δ)
    min_pair_weight: float = 1e-6    # отбрасываем очень слабые веса (после агрегации)
    min_out_degree: int = 1          # узлы без исходящих удаляем из P

    # PPR
    alpha: float = 0.20              # вероятность рестарта
    max_iters: int = 12              # кол-во итераций
    tol: float = 1e-6                # критерий остановки (L1)
    prune_topn: int = 20000          # редуцируем распределение p до топ-N после каждого шага

    # персонализация
    profile_L: int = 10              # сколько последних айтемов в профиль
    profile_decay_gamma: float = 0.80  # 1.0 — без затухания; <1 — убывание веса по позиции
    seed_exclude: bool = True        # исключать ли сами seed-айтемы из рекомендаций

    # инференс
    exclude_seen: bool = True        # исключать виденные в train
    candidate_cap_per_user: int = 600  # итоговое K кандидатов для артефактов
    k_list: Sequence[int] = (20, 50, 100)
    k_top_submit: int = 20

    # прочее
    seed: int = 42


# =========================
# Построение графа item-item
# =========================

def _build_item_graph(train_df: pd.DataFrame,
                      pair_mode: str = "consecutive",
                      beta: float = 0.25,
                      min_w: float = 1e-6,
                      min_out_degree: int = 1) -> Dict[int, Dict[int, float]]:
    """
    Возвращает словарь переходов P: i -> {j: prob}, где prob — нормированные веса.
    Веса накапливаются по co-vis с эксп. затуханием по разнице дней.
    """
    # аккуратный набор рёбер с весами
    edges: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    # упорядочиваем историю по пользователю/дню
    base = train_df[[COL_USER, COL_ITEM, COL_DATE]].copy()
    base[COL_USER] = base[COL_USER].astype(int)
    base[COL_ITEM] = base[COL_ITEM].astype(int)
    base[COL_DATE] = base[COL_DATE].astype(int)
    base = base.sort_values([COL_USER, COL_DATE])

    if pair_mode == "consecutive":
        for _, g in base.groupby(COL_USER, sort=False):
            it = g[COL_ITEM].values
            dt = g[COL_DATE].values
            # соседние пары
            for k in range(len(it) - 1):
                a, b = int(it[k]), int(it[k+1])
                d = abs(int(dt[k+1]) - int(dt[k]))
                w = math.exp(-beta * d)
                if a != b and w > 0:
                    edges[a][b] += w
                    edges[b][a] += w
    elif pair_mode == "all_pairs":
        # все пары внутри пользователя (квадратично по длине истории; осторожнее)
        for _, g in base.groupby(COL_USER, sort=False):
            it = g[COL_ITEM].values
            dt = g[COL_DATE].values
            n = len(it)
            for k in range(n):
                for m in range(k+1, n):
                    a, b = int(it[k]), int(it[m])
                    if a == b: 
                        continue
                    d = abs(int(dt[m]) - int(dt[k]))
                    w = math.exp(-beta * d)
                    if w > 0:
                        edges[a][b] += w
                        edges[b][a] += w
    else:
        raise ValueError("pair_mode must be in {'consecutive','all_pairs'}")

    # отфильтруем слабые веса и нормализуем по строке
    P: Dict[int, Dict[int, float]] = {}
    for i, nbrs in edges.items():
        # отбрасываем мусор
        nbrs = {j: w for j, w in nbrs.items() if w >= min_w and j != i}
        if not nbrs:
            continue
        s = float(sum(nbrs.values()))
        if s <= 0.0:
            continue
        # нормализуем
        row = {j: float(w / s) for j, w in nbrs.items()}
        # минимальная степень
        if len(row) >= int(min_out_degree):
            P[int(i)] = row

    return P


# =========================
# PPR итерации на словарях
# =========================

def _personalize_from_seeds(seeds: List[int], gamma: float) -> Dict[int, float]:
    """v по seed-позициям с эксп. затуханием по порядку (новее — больше)."""
    if not seeds:
        return {}
    # последний клик — самый весомый
    weights = [gamma ** (len(seeds) - 1 - t) for t in range(len(seeds))]
    s = float(sum(weights))
    if s <= 0.0:
        return {}
    return {int(i): float(w / s) for i, w in zip(seeds, weights)}

def _ppr_iterative(P: Dict[int, Dict[int, float]],
                   v: Dict[int, float],
                   alpha: float,
                   max_iters: int,
                   tol: float,
                   prune_topn: int) -> Dict[int, float]:
    """
    Итерации p = α v + (1-α) Pᵀ p; P задан списками смежности (строки — исходящие).
    Представление p — как dict item->prob; на каждом шаге укорачиваем до top-N.
    """
    if not v:
        return {}

    p = v.copy()  # старт с персонализации
    for _ in range(int(max_iters)):
        new_p: Dict[int, float] = {}
        # базовая часть α v
        a_part = float(alpha)
        for i, w in v.items():
            new_p[i] = new_p.get(i, 0.0) + a_part * float(w)
        # перенос по графу
        carry = float(1.0 - alpha)
        for i, pi in p.items():
            row = P.get(int(i))
            if not row:
                continue
            # распределить pi по соседям
            mul = carry * float(pi)
            for j, pij in row.items():
                new_p[j] = new_p.get(j, 0.0) + mul * float(pij)

        # L1-схождение (приблизительно, на top-n)
        if prune_topn and len(new_p) > prune_topn:
            # оставим только top-n
            items, vals = zip(*new_p.items())
            vals = np.asarray(vals, dtype=np.float64)
            if prune_topn < len(vals):
                idx = np.argpartition(-vals, prune_topn - 1)[:prune_topn]
                new_p = {int(items[k]): float(vals[k]) for k in idx.tolist()}
            else:
                new_p = dict(new_p)

        # нормируем (стабилизация сумм)
        s = float(sum(new_p.values()))
        if s > 0:
            for k in list(new_p.keys()):
                new_p[k] = new_p[k] / s

        # критерий остановки
        # посчитаем L1 на пересечении ключей (дешёвая аппрокс.)
        diff = 0.0
        if len(p) and len(new_p):
            inter = set(p.keys()) | set(new_p.keys())
            for k in inter:
                diff += abs(float(new_p.get(k, 0.0)) - float(p.get(k, 0.0)))
        if diff <= float(tol):
            p = new_p
            break

        p = new_p

    return p


# =========================
# Профили пользователей
# =========================

def _users_last_items(train_df: pd.DataFrame, L: int) -> Dict[int, List[int]]:
    base = train_df.sort_values([COL_USER, COL_DATE])
    out: Dict[int, List[int]] = {}
    for u, g in base.groupby(COL_USER, sort=False):
        out[int(u)] = list(map(int, g[COL_ITEM].tail(int(L)).tolist()))
    return out

def _users_seen_map(train_df: pd.DataFrame) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for u, g in train_df.groupby(COL_USER, sort=False):
        out[int(u)] = set(map(int, g[COL_ITEM].unique()))
    return out


# =========================
# Эксперимент
# =========================

class Exp107PPR(BaseExperiment):
    exp_name = "exp107_ppr"

    def __init__(self, cfg: Optional[Exp107Config] = None):
        self.cfg = cfg or Exp107Config()
        self.P_: Optional[Dict[int, Dict[int, float]]] = None
        self.last_items_: Optional[Dict[int, List[int]]] = None
        self.seen_map_: Optional[Dict[int, set]] = None

    # ---------- helpers ----------

    def _ensure_graph_and_profiles(self, ctx: dict):
        if self.P_ is None:
            train_df: pd.DataFrame = ctx["train_df"]
            self.P_ = _build_item_graph(
                train_df=train_df,
                pair_mode=self.cfg.pair_mode,
                beta=float(self.cfg.time_decay_beta),
                min_w=float(self.cfg.min_pair_weight),
                min_out_degree=int(self.cfg.min_out_degree),
            )
        if self.last_items_ is None:
            self.last_items_ = _users_last_items(ctx["train_df"], L=int(self.cfg.profile_L))
        if self.seen_map_ is None and self.cfg.exclude_seen:
            self.seen_map_ = _users_seen_map(ctx["train_df"])

    def _users_for_val(self, ctx: dict) -> List[int]:
        if "val_truth" in ctx and isinstance(ctx["val_truth"], dict):
            return list(map(int, ctx["val_truth"].keys()))
        if "val_users" in ctx:
            return list(map(int, ctx["val_users"]))
        # запасной вариант: пользователи последних 7 дней train
        base = ctx["train_df"]
        mx = int(base[COL_DATE].max())
        lo = mx - 6
        return list(map(int, base[base[COL_DATE] >= lo][COL_USER].unique()))

    def _users_for_test(self, ctx: dict) -> List[int]:
        if "test_users" in ctx:
            return list(map(int, ctx["test_users"]))
        if "sample_df" in ctx and isinstance(ctx["sample_df"], pd.DataFrame):
            return list(map(int, ctx["sample_df"][COL_USER].unique()))
        return []

    def _score_users(self, users: List[int], topk: int) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """
        Считает PPR-скоры и top-K кандидатов для списка users.
        Возвращает (df_scores [u,i,score], u2items map).
        """
        if not users:
            return pd.DataFrame(columns=[COL_USER, COL_ITEM, "score"]), {}

        P = self.P_ or {}
        last_map = self.last_items_ or {}
        seen_map = self.seen_map_ or defaultdict(set)

        rows = []
        u2items: Dict[int, List[int]] = {}

        for u in users:
            seeds = list(map(int, last_map.get(int(u), [])[-int(self.cfg.profile_L):]))
            if not seeds:
                # у юзера нет истории — пропускаем
                continue
            v = _personalize_from_seeds(seeds, gamma=float(self.cfg.profile_decay_gamma))
            if not v:
                continue

            p = _ppr_iterative(
                P=P,
                v=v,
                alpha=float(self.cfg.alpha),
                max_iters=int(self.cfg.max_iters),
                tol=float(self.cfg.tol),
                prune_topn=int(self.cfg.prune_topn),
            )
            if not p:
                continue

            # исключим seen и seed-айтемы (по настройке)
            forbid = set()
            if self.cfg.exclude_seen:
                forbid |= set(map(int, seen_map.get(int(u), set())))
            if self.cfg.seed_exclude:
                forbid |= set(seeds)

            # топ-K
            items, scores = zip(*p.items())
            items = np.asarray(items, dtype=np.int64)
            scores = np.asarray(scores, dtype=np.float32)
            # отфильтруем запрещённые
            if len(forbid) > 0:
                mask = np.array([int(it) not in forbid for it in items], dtype=bool)
                items = items[mask]
                scores = scores[mask]
            if len(items) == 0:
                continue

            kk = min(int(topk), len(items))
            idx = np.argpartition(-scores, kk - 1)[:kk]
            idx = idx[np.argsort(-scores[idx], kind="mergesort")]
            top_items = items[idx].tolist()
            top_scores = scores[idx].tolist()

            u2items[int(u)] = list(map(int, top_items))
            rows.extend([(int(u), int(it), float(sc)) for it, sc in zip(top_items, top_scores)])

        df = pd.DataFrame(rows, columns=[COL_USER, COL_ITEM, "score"])
        return df, u2items

    # ---------- API ----------

    def fit(self, ctx: dict):
        self.ctx = ctx
        self._ensure_graph_and_profiles(ctx)

        users = self._users_for_val(ctx)
        if not users:
            raise RuntimeError("[exp107_ppr] cannot resolve validation users")

        # скоринг
        df_val, map_val = self._score_users(users, topk=int(self.cfg.candidate_cap_per_user))

        # метрики
        truth: Dict[int, set] = ctx["val_truth"]
        pred_map = {u: items for u, items in map_val.items() if items}
        rows = []
        for k in self.cfg.k_list:
            rows.append({
                "k": k,
                "HR@k": recall_at_m(pred_map, truth, m=k),
                "NDCG@k": ndcg_at_k(pre
