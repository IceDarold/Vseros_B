# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_exp import BaseExperiment
from .config import PATHS
from .artifacts import ensure_dir, save_df, save_and_log_df, save_json, log_table_df
from .metrics import (
    hr_at_k_userwise,              # аккуратный HR@K по пользователям
    ndcg_at_k_userwise,            # аккуратный NDCG@K по пользователям
    coverage_at_k,                 # item-coverage@K (по val_item_cnt)
)

# W&B — опционально
try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None  # type: ignore
    _WANDB = False


# -----------------------------------------------------------------------------
# Конфиг/стейт
# -----------------------------------------------------------------------------

@dataclass
class Exp101Config:
    # сетка эксп. затухания по времени (чем больше, тем сильнее приоритет последним дням)
    decay_grid: Tuple[float, ...] = (0.00, 0.02, 0.06, 0.12, 0.20)
    # сколько кандидатов класть на пользователя (для вал/test артефактов и сабмита)
    per_user_K: int = 600
    # сколько верхних айтемов держать глобально (чтобы не таскать всю голову)
    global_top_cap: int = 200_000
    # фильтр по минимальному числу встреч в train (0 = без фильтра)
    min_train_freq: int = 0
    # артефактные имена и поведение
    name: str = "exp101_pop_decay"
    # логгировать train/val головы топов как таблицы
    log_top_previews: bool = True


@dataclass
class Exp101State:
    best_decay: float = 0.0
    best_spearman: float = 0.0
    train_item_scores: Optional[pd.Series] = None       # index=item_id, value=score
    global_ranking: Optional[pd.Index] = None           # отсортированный список item_id
    # кэши для сабмита/кандидатов
    top_items_cut: Optional[pd.Index] = None


# -----------------------------------------------------------------------------
# Вспомогательные
# -----------------------------------------------------------------------------

def _compute_recency_weighted_pop(
    df: pd.DataFrame, end_day: int, decay: float
) -> pd.Series:
    """
    df: long (user_id, item_id, date), ТОЛЬКО train-дни
    вес = exp(-decay * (end_day - date))
    возвращает: Series[item_id] = score
    """
    if len(df) == 0:
        return pd.Series(dtype="float32")

    d = df[["item_id", "date"]].copy()
    # чтобы избежать больших экспонент, можно центрировать на end_day
    w = np.exp(-float(decay) * (float(end_day) - d["date"].astype("float32")))
    d["w"] = w.astype("float32")
    pop = d.groupby("item_id", sort=False)["w"].sum()
    pop = pop.astype("float32")
    return pop


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    """
    Спирмен по пересечению индексов, без NaN.
    """
    idx = a.index.intersection(b.index)
    if len(idx) < 10:
        return 0.0
    a_rank = a.loc[idx].rank(ascending=False)
    b_rank = b.loc[idx].rank(ascending=False)
    a_mean = a_rank.mean()
    b_mean = b_rank.mean()
    num = ((a_rank - a_mean) * (b_rank - b_mean)).sum()
    den = math.sqrt(((a_rank - a_mean) ** 2).sum()) * math.sqrt(((b_rank - b_mean) ** 2).sum())
    if den <= 0:
        return 0.0
    return float(num / den)


def _series_head_table(s: pd.Series, n: int = 20) -> pd.DataFrame:
    """Удобный превью топа как таблицы для W&B."""
    if s is None or len(s) == 0:
        return pd.DataFrame({"item_id": [], "score": []})
    head = s.sort_values(ascending=False).head(n)
    return pd.DataFrame({"item_id": head.index.astype(np.int64), "score": head.values.astype(np.float32)})


def _users_from_days(df: pd.DataFrame, days: List[int]) -> np.ndarray:
    sub = df[df["date"].isin(days)]
    return sub["user_id"].drop_duplicates().astype(np.int64).values


# -----------------------------------------------------------------------------
# Эксперимент
# -----------------------------------------------------------------------------

class Exp101PopDecay(BaseExperiment):
    """
    Простая глобальная популярность с экспоненциальным затуханием по времени.
    - выбор decay по максимальной корреляции с вал.частотами айтемов (стабильный критерий);
    - оценка на вал: HR/NDCG@K по пользователям (фиксированный K=20/50/100 из metrics);
    - формирование кандидатов и сабмита (лонг-формат).
    """

    def __init__(self, cfg: Optional[Exp101Config] = None):
        super().__init__(exp_name=(cfg.name if cfg else "exp101_pop_decay"))
        self.cfg: Exp101Config = cfg or Exp101Config()
        self.state: Exp101State = Exp101State()

    # ---- fit: обучаемся на train, подбираем decay, собираем ранжирование ----
    def fit(self, context: Dict) -> Exp101State:
        t0 = time.time()
        self.require_context_keys(context, ["train_df", "val_df", "split", "val_item_cnt"])

        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame = context["val_df"]
        split: SimpleNamespace = context["split"]
        val_item_cnt: pd.Series = context["val_item_cnt"]

        # фильтр по train-частоте (если включён)
        if self.cfg.min_train_freq > 0:
            vc = train_df["item_id"].value_counts()
            keep_items = set(vc[vc >= int(self.cfg.min_train_freq)].index.astype(np.int64))
            train_df = train_df[train_df["item_id"].isin(keep_items)]
            if self.verbose:
                print(f"[{self.exp_name}] after freq>={self.cfg.min_train_freq}: train rows={len(train_df):,}, uniq items={train_df['item_id'].nunique():,}")

        # подберём decay по корреляции c вал. частотами айтемов
        best_decay, best_r = 0.0, -1.0
        best_scores = None

        for d in self.cfg.decay_grid:
            scores = _compute_recency_weighted_pop(train_df, end_day=int(split.train_end), decay=float(d))
            # нормализация (по желанию можно не делать)
            scores = scores / max(1e-9, float(scores.max()))
            r = _safe_spearman(scores, val_item_cnt)
            if self.verbose:
                print(f"[{self.exp_name}] decay={d:.3f} spearman(train_pop vs val_cnt)={r:.5f}   (items={len(scores):,})")
            if r > best_r:
                best_r = r
                best_decay = d
                best_scores = scores

        if best_scores is None:
            # на всякий случай fallback
            best_scores = train_df["item_id"].value_counts().astype("float32")
            best_scores = best_scores / max(1.0, float(best_scores.max()))
            best_decay = 0.0
            best_r = _safe_spearman(best_scores, val_item_cnt)

        # сортировка головы
        order = best_scores.sort_values(ascending=False)
        if self.cfg.global_top_cap and self.cfg.global_top_cap > 0:
            order = order.head(int(self.cfg.global_top_cap))
        top_items = order.index

        # сохраняем в state
        self.state.best_decay = float(best_decay)
        self.state.best_spearman = float(best_r)
        self.state.train_item_scores = order.astype("float32")
        self.state.global_ranking = pd.Index(order.index.astype(np.int64))
        self.state.top_items_cut = self.state.global_ranking

        # W&B превью
        if _WANDB and wandb.run is not None and self.cfg.log_top_previews:
            log_table_df(f"{self.exp_name}/top_train_head", _series_head_table(order, 25))
            try:
                wandb.summary[f"{self.exp_name}/best_variant"] = f"decay_{best_decay:.2f}"
                wandb.summary[f"{self.exp_name}/spearman_train_val"] = float(best_r)
            except Exception:
                pass

        if self.verbose:
            print(f"[{self.exp_name}] best decay={best_decay:.3f}, spearman={best_r:.5f}, kept items={len(order):,} (cap={self.cfg.global_top_cap})")

        if _WANDB and wandb.run is not None:
            wandb.log({f"{self.exp_name}/fit_payload": 1})

        if self.verbose:
            print(f"[{self.exp_name}] fit() done in {time.time()-t0:.1f}s")
        return self.state

    # ---- evaluate: считаем HR/NDCG/coverage на вал ----
    def evaluate(self, context: Dict) -> pd.DataFrame:
        t0 = time.time()
        self.require_context_keys(context, ["val_df", "split", "val_item_cnt"])
        assert self.state.global_ranking is not None, "Run fit() first."

        val_df: pd.DataFrame = context["val_df"]
        split: SimpleNamespace = context["split"]
        val_item_cnt: pd.Series = context["val_item_cnt"]

        K_list = (20, 50, 100)

        # единый топ для всех пользователей
        gtop = list(self.state.global_ranking)
        # HR/NDCG @ K — userwise
        hr_rows = []
        ndcg_rows = []
        for K in K_list:
            hr = hr_at_k_userwise(val_df, gtop[:K])
            nd = ndcg_at_k_userwise(val_df, gtop[:K])
            hr_rows.append({"k": K, "HR@k": hr})
            ndcg_rows.append({"k": K, "NDCG@k": nd})

        # coverage@K — item coverage на вал-юниверсе
        cov_rows = []
        order_series = self.state.train_item_scores  # Series[item_id]=score
        for K in K_list:
            cov = coverage_at_k(order_series, val_item_cnt, ks=(K,))
            cov_rows.append({"k": K, "COV@k": float(cov.iloc[0]["coverage"]) if hasattr(cov, "iloc") else float(cov)})

        # соберём таблицу метрик
        # (соединим по k)
        m = pd.DataFrame({"k": K_list})
        m = m.merge(pd.DataFrame(hr_rows), on="k", how="left")
        m = m.merge(pd.DataFrame(ndcg_rows), on="k", how="left")
        m = m.merge(pd.DataFrame(cov_rows), on="k", how="left")

        # лог в csv и W&B
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.cfg.name)
        save_df(out_dir / "val_metrics.csv", m, index=False)
        if _WANDB and wandb.run is not None:
            log_table_df(f"{self.exp_name}/val_metrics", m)

        if self.verbose:
            print(f"[{self.exp_name}] evaluate() →\n{m}")
            print(f"[{self.exp_name}] evaluate() done in {time.time()-t0:.1f}s")

        return m

    # ---- save: положим "модель" (глобальный топ) + кандидатов val/test ----
    def save(self, context: Dict) -> Tuple[Optional[str], Optional[str]]:
        t0 = time.time()
        self.require_context_keys(context, ["train_df", "val_df", "split", "sample_df"])
        assert self.state.top_items_cut is not None, "Run fit() first."

        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame = context["val_df"]
        split: SimpleNamespace = context["split"]
        sample_df: pd.DataFrame = context["sample_df"]

        # --- 1) сохраним «модель» (глобальные веса айтемов) ---
        model_dir = ensure_dir(PATHS.artifact_dir / "models" / self.cfg.name)
        scores = self.state.train_item_scores.rename("score").reset_index().rename(columns={"index": "item_id"})
        save_df(model_dir / "item_scores.parquet", scores, index=False)
        meta = {
            "best_decay": self.state.best_decay,
            "best_spearman": self.state.best_spearman,
            "global_top_cap": self.cfg.global_top_cap,
            "min_train_freq": self.cfg.min_train_freq,
        }
        save_json(model_dir / "model_meta.json", meta)

        # лог как артефакт (W&B — опционально)
        if _WANDB and wandb.run is not None:
            try:
                save_and_log_df(scores, model_dir / "item_scores_for_wandb.csv",
                                artifact_name=f"{self.cfg.name}_item_scores",
                                artifact_type="model-scores",
                                table_key=f"{self.exp_name}/item_scores_preview")
            except Exception:
                pass

        # --- 2) кандидаты для валидации и теста ---
        cand_dir = ensure_dir(PATHS.cand_dir / self.cfg.name)
        K = int(self.cfg.per_user_K)
        gtop = list(self.state.top_items_cut)

        # val users
        val_users = _users_from_days(val_df, list(split.val_days))
        # test users — из sample
        test_users = sample_df["user_id"].drop_duplicates().astype(np.int64).values

        def _mk_cands(users: np.ndarray) -> pd.DataFrame:
            # одинаковый топ на всех — быстро через repeat без explode
            n = len(users)
            k = min(K, len(gtop))
            items = np.array(gtop[:k], dtype=np.int64)
            # сформируем каркас
            u_col = np.repeat(users, k)
            i_col = np.tile(items, n)
            r_col = np.tile(np.arange(1, k + 1, dtype=np.int32), n)
            s_col = (1.0 / r_col).astype("float32")  # простая убывающая «оценка»
            out = pd.DataFrame({
                "user_id": u_col,
                "item_id": i_col,
                "rank": r_col,
                "score": s_col,
                "src": self.cfg.name,
            })
            return out

        val_cands = _mk_cands(val_users)
        test_cands = _mk_cands(test_users)

        p_val = save_df(cand_dir / "val_candidates.parquet", val_cands, index=False)
        p_test = save_df(cand_dir / "test_candidates.parquet", test_cands, index=False)

        if self.verbose:
            print(f"[{self.exp_name}] saved candidates: val={val_cands.shape}, test={test_cands.shape}")
            print(f"[{self.exp_name}] save() done in {time.time()-t0:.1f}s")

        return str(p_val), str(p_test)

    # ---- predict_submission: возвращаем ЛОНГ-формат под sample_df ----
    def predict_submission(self, context: Dict, sample_df: pd.DataFrame, k_top: int = 20) -> pd.DataFrame:
        """
        Ожидаем, что sample_df имеет колонки ['user_id','item_id'] и каждые K строк на пользователя.
        Возвращаем копию sample_df с заполненным 'item_id' (лонг-формат).
        """
        assert self.state.top_items_cut is not None, "Run fit() first."
        t0 = time.time()

        K = int(k_top)
        gtop = list(self.state.top_items_cut[:K])

        # быстрый assign: сформируем map user->topK и вливаем батчем
        users = sample_df["user_id"].values
        uniq_users, inv = np.unique(users, return_inverse=True)
        topK = np.array(gtop, dtype=np.int64)
        # ожидаем, что в sample по каждому юзеру ровно K строк (валидация структуры — опционально)
        # соберём матрицу (U,K) → развёрнем по inv
        # если в sample иногда !=K, безопаснее просто «циклом»; но это медленнее
        filled = sample_df.copy()
        # создадим шаблон (для каждого uniq_user — свой список topK)
        pool = np.vstack([topK for _ in range(len(uniq_users))])
        # сколько строк на пользователя в sample
        # аккуратный способ — отсортированный sample: будем пробегать и повторять
        # но здесь предположим ровно K на пользователя:
        try:
            # проверка (не падаем, просто предупреждаем)
            counts = pd.Series(users).value_counts().values
            if not np.all(counts == K):
                # fallback — медленнее, но корректно
                grp = filled.groupby("user_id", sort=False)
                rows = []
                for uid, g in grp:
                    need = len(g)
                    take = min(K, len(topK))
                    rep = np.resize(topK[:take], need)
                    out = g.copy()
                    out.loc[:, "item_id"] = rep
                    rows.append(out)
                filled = pd.concat(rows, axis=0, ignore_index=True)
                if self.verbose:
                    print(f"[{self.exp_name}] sample has irregular K per user → used safe fallback path.")
                return filled
        except Exception:
            pass

        # быстрый путь: восстановим порядок sample
        # индексы внутри каждого пользователя должны идти блоками длины K
        # сформируем вектор item_id согласно исходному порядку
        # inv даёт индекс uniq_user для каждой строки sample
        filled["item_id"] = pool[inv, np.arange(len(inv)) % K]
        if self.verbose:
            print(f"[{self.exp_name}] predict_submission() done in {time.time()-t0:.1f}s  (rows={len(filled):,})")
        return filled
