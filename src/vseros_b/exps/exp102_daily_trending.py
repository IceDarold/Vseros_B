# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base_exp import BaseExperiment
from ..config import PATHS
from ..artifacts import ensure_dir, save_df, save_json, save_and_log_df, log_table_df
from ..metrics import hr_at_k_userwise, ndcg_at_k_userwise, coverage_at_k
from .. import trending as tr

# W&B — опционально
try:
    import wandb
    _WANDB = True
except Exception:
    wandb = None  # type: ignore
    _WANDB = False


# =============================================================================
# Конфиг / стейт
# =============================================================================

@dataclass
class Exp102Config:
    """
    Daily-trending кандидаты без утечек.
      - mode='frozen_train': один глобальный топ по окну [train_end - window_days, train_end]
      - mode='day_aware': топ на каждый день d∈[val_start,val_end] по окну [d - window_days, d-1]
    """
    name: str = "exp102_daily_trending"
    mode: str = "frozen_train"     # 'frozen_train' | 'day_aware'
    window_days: int = 3
    topk_per_day: int = 1000       # глубина day-топа (для day_aware/оценки)
    per_user_M: int = 600          # сколько кандидатов на пользователя (верхний M)
    min_count: int = 1             # порог частоты для попадания в day-топ
    log_top_previews: bool = True  # логировать головы топов в W&B


@dataclass
class Exp102State:
    mode: str = "frozen_train"
    # frozen
    global_top: Optional[pd.Index] = None          # pd.Index[item_id] (отсортированный)
    global_scores: Optional[pd.Series] = None      # Series[item_id]=score
    # day-aware
    day_top: Optional[Dict[int, List[int]]] = None # day -> list[item_id]


# =============================================================================
# Утилиты локальные
# =============================================================================

def _users_from_days(df: pd.DataFrame, days: List[int]) -> np.ndarray:
    sub = df[df["date"].isin(days)]
    return sub["user_id"].drop_duplicates().astype(np.int64).values

def _mk_uniform_candidates(users: np.ndarray, top_items: List[int], K: int, src: str) -> pd.DataFrame:
    """
    Единый топ для всех пользователей.
    """
    n = len(users)
    k = min(K, len(top_items))
    items = np.array(top_items[:k], dtype=np.int64)
    u_col = np.repeat(users, k)
    i_col = np.tile(items, n)
    r_col = np.tile(np.arange(1, k + 1, dtype=np.int32), n)
    s_col = (1.0 / r_col).astype("float32")
    return pd.DataFrame({
        "user_id": u_col,
        "item_id": i_col,
        "rank": r_col,
        "score": s_col,
        "src": src,
    })

def _safe_table_top(items: List[int], n: int = 20) -> pd.DataFrame:
    n = min(n, len(items))
    return pd.DataFrame({"rank": np.arange(1, n + 1, dtype=np.int32),
                         "item_id": np.array(items[:n], dtype=np.int64)})


# =============================================================================
# Эксперимент
# =============================================================================

class Exp102DailyTrending(BaseExperiment):
    """
    Трендинг-кандидаты без утечек. Два режима:
      - frozen_train: один глобальный топ по последнему train-окну
      - day_aware: топы на каждый вал-день по скользящему окну
    """

    def __init__(self, cfg: Optional[Exp102Config] = None):
        super().__init__(exp_name=(cfg.name if cfg else "exp102_daily_trending"))
        self.cfg = cfg or Exp102Config()
        self.state = Exp102State(mode=self.cfg.mode)

    # -------------------------------------------------------------------------
    # FIT
    # -------------------------------------------------------------------------
    def fit(self, context: Dict) -> Exp102State:
        t0 = time.time()
        self.require_context_keys(context, ["train_df", "val_df", "split"])
        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame = context["val_df"]
        split: SimpleNamespace = context["split"]

        if self.cfg.mode == "frozen_train":
            # один глобальный топ из последнего окна train
            gtop_list, g_scores = tr.trending_global_from_last_train_window(
                train_df=train_df,
                train_end=int(split.train_end),
                window_days=int(self.cfg.window_days),
                topk=int(max(self.cfg.topk_per_day, self.cfg.per_user_M)),
                min_count=int(self.cfg.min_count),
            )
            self.state.global_top = pd.Index(np.array(gtop_list, dtype=np.int64))
            self.state.global_scores = g_scores

            if self.verbose:
                print(f"[{self.exp_name}] frozen_train: global_top size={len(self.state.global_top)}  window={self.cfg.window_days}d")

            if _WANDB and wandb.run is not None and self.cfg.log_top_previews:
                log_table_df(f"{self.exp_name}/global_top_head", _safe_table_top(gtop_list, 30))

        elif self.cfg.mode == "day_aware":
            # day→top (валидационные дни)
            day_top = tr.build_val_day_toplists(
                train_df=train_df,
                val_start=int(split.val_start),
                val_end=int(split.val_end),
                window_days=int(self.cfg.window_days),
                topk_per_day=int(self.cfg.topk_per_day),
                min_count=int(self.cfg.min_count),
            )
            self.state.day_top = day_top

            if self.verbose:
                days = list(day_top.keys())
                print(f"[{self.exp_name}] day_aware: built day_top for days {days[:3]}.. total={len(days)}")
                for d in days[:2]:
                    print(f"  - day={d} top_head={day_top[d][:10]}")

            if _WANDB and wandb.run is not None and self.cfg.log_top_previews:
                # залогируем по 2 дня превью, чтобы не взрывать UI
                days = list(day_top.keys())
                for d in days[:2]:
                    log_table_df(f"{self.exp_name}/day{d}_top_head", _safe_table_top(day_top[d], 30))
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")

        if _WANDB and wandb.run is not None:
            try:
                wandb.summary[f"{self.exp_name}/mode"] = self.cfg.mode
                wandb.summary[f"{self.exp_name}/window_days"] = int(self.cfg.window_days)
            except Exception:
                pass

        if self.verbose:
            print(f"[{self.exp_name}] fit() done in {time.time()-t0:.1f}s")
        return self.state

    # -------------------------------------------------------------------------
    # EVALUATE
    # -------------------------------------------------------------------------
    def evaluate(self, context: Dict) -> pd.DataFrame:
        t0 = time.time()
        self.require_context_keys(context, ["val_df", "split", "val_item_cnt"])
        val_df: pd.DataFrame = context["val_df"]
        split: SimpleNamespace = context["split"]
        val_item_cnt: pd.Series = context["val_item_cnt"]

        K_list = (20, 50, 100)
        rows = []

        if self.cfg.mode == "frozen_train":
            assert self.state.global_top is not None, "Run fit() first."
            gtop = list(self.state.global_top)

            # HR/NDCG @ K — единый топ
            for K in K_list:
                hr = hr_at_k_userwise(val_df, gtop[:K])
                nd = ndcg_at_k_userwise(val_df, gtop[:K])
                rows.append({"k": K, "HR@k": hr, "NDCG@k": nd})

            # coverage@K
            if self.state.global_scores is not None:
                cov_df = coverage_at_k(self.state.global_scores, val_item_cnt, ks=K_list)
                # cov_df: columns ['k','coverage']
                if isinstance(cov_df, pd.DataFrame):
                    for _, r in cov_df.iterrows():
                        # дополним запись для нужного k
                        for j in range(len(rows)):
                            if rows[j]["k"] == int(r["k"]):
                                rows[j]["COV@k"] = float(r["coverage"])

        else:  # day_aware
            assert self.state.day_top is not None, "Run fit() first."
            # считаем помодно: для каждого дня своя рекомендация
            # HR/NDCG считаем по пользователям, присутствующим в конкретный день
            day_groups = dict(tuple(val_df.groupby("date", sort=False)))
            agg = {K: {"hr_sum": 0.0, "ndcg_sum": 0.0, "cnt": 0} for K in K_list}

            for d, df_d in day_groups.items():
                top_d = self.state.day_top.get(int(d))
                if not top_d:
                    continue
                for K in K_list:
                    hr = hr_at_k_userwise(df_d, top_d[:K])
                    nd = ndcg_at_k_userwise(df_d, top_d[:K])
                    n_users = df_d["user_id"].nunique()
                    agg[K]["hr_sum"] += hr * n_users
                    agg[K]["ndcg_sum"] += nd * n_users
                    agg[K]["cnt"] += n_users

            for K in K_list:
                cnt = max(1, agg[K]["cnt"])
                rows.append({
                    "k": K,
                    "HR@k": agg[K]["hr_sum"] / cnt,
                    "NDCG@k": agg[K]["ndcg_sum"] / cnt,
                })

            # coverage@K — приблизим глобальным трендингом последнего train-окна,
            # чтобы иметь сопоставимую метрику
            gtop_list, g_scores = tr.trending_global_from_last_train_window(
                train_df=context["train_df"],
                train_end=int(split.train_end),
                window_days=int(self.cfg.window_days),
                topk=int(max(self.cfg.topk_per_day, self.cfg.per_user_M)),
                min_count=int(self.cfg.min_count),
            )
            cov_df = coverage_at_k(g_scores, val_item_cnt, ks=K_list)
            if isinstance(cov_df, pd.DataFrame):
                for _, r in cov_df.iterrows():
                    for j in range(len(rows)):
                        if rows[j]["k"] == int(r["k"]):
                            rows[j]["COV@k"] = float(r["coverage"])

        out = pd.DataFrame(rows)

        # лог/сохранение
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.cfg.name)
        save_df(out_dir / "val_metrics.csv", out, index=False)
        if _WANDB and wandb.run is not None:
            log_table_df(f"{self.exp_name}/val_metrics", out)

        if self.verbose:
            print(f"[{self.exp_name}] evaluate() →\n{out}")
            print(f"[{self.exp_name}] evaluate() done in {time.time()-t0:.1f}s")

        return out

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    def save(self, context: Dict) -> Tuple[Optional[str], Optional[str]]:
        t0 = time.time()
        self.require_context_keys(context, ["train_df", "val_df", "split", "sample_df"])
        train_df: pd.DataFrame = context["train_df"]
        val_df: pd.DataFrame = context["val_df"]
        split: SimpleNamespace = context["split"]
        sample_df: pd.DataFrame = context["sample_df"]

        cand_dir = ensure_dir(PATHS.cand_dir / self.cfg.name)
        K = int(self.cfg.per_user_M)

        # --- VAL candidates ---
        if self.cfg.mode == "frozen_train":
            assert self.state.global_top is not None
            users_val = _users_from_days(val_df, list(split.val_days))
            val_cands = _mk_uniform_candidates(users_val, list(self.state.global_top), K, self.cfg.name)
        else:
            assert self.state.day_top is not None
            # склеим по дням
            parts = []
            for d in range(int(split.val_start), int(split.val_end) + 1):
                users_d = val_df.loc[val_df["date"] == int(d), "user_id"].drop_duplicates().astype(np.int64).values
                top_d = self.state.day_top.get(int(d), [])
                if len(users_d) == 0 or len(top_d) == 0:
                    continue
                df_d = _mk_uniform_candidates(users_d, top_d, K, self.cfg.name)
                df_d["date"] = int(d)
                parts.append(df_d)
            val_cands = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(
                columns=["user_id", "item_id", "rank", "score", "src", "date"]
            )

        p_val = save_df(cand_dir / "val_candidates.parquet", val_cands, index=False)

        # --- TEST candidates ---
        # Для теста используем глобальный трендинг по последнему train-окну (без утечки).
        gtop_list, _g_scores = tr.trending_global_from_last_train_window(
            train_df=train_df,
            train_end=int(split.train_end),
            window_days=int(self.cfg.window_days),
            topk=int(max(self.cfg.topk_per_day, self.cfg.per_user_M)),
            min_count=int(self.cfg.min_count),
        )
        users_test = sample_df["user_id"].drop_duplicates().astype(np.int64).values
        test_cands = _mk_uniform_candidates(users_test, gtop_list, K, self.cfg.name)
        p_test = save_df(cand_dir / "test_candidates.parquet", test_cands, index=False)

        # --- модельные метаданные (для воспроизводимости) ---
        model_dir = ensure_dir(PATHS.artifact_dir / "models" / self.cfg.name)
        meta = {
            "mode": self.cfg.mode,
            "window_days": int(self.cfg.window_days),
            "topk_per_day": int(self.cfg.topk_per_day),
            "per_user_M": int(self.cfg.per_user_M),
            "min_count": int(self.cfg.min_count),
        }
        save_json(model_dir / "model_meta.json", meta)

        if self.verbose:
            print(f"[{self.exp_name}] saved candidates: val={val_cands.shape}, test={test_cands.shape}")
            print(f"[{self.exp_name}] save() done in {time.time()-t0:.1f}s")

        # лёгкие превью в W&B
        if _WANDB and wandb.run is not None:
            try:
                log_table_df(f"{self.exp_name}/val_cands_head", val_cands.head(40))
                log_table_df(f"{self.exp_name}/test_cands_head", test_cands.head(40))
            except Exception:
                pass

        return str(p_val), str(p_test)

    # -------------------------------------------------------------------------
    # SUBMISSION (только для frozen_train)
    # -------------------------------------------------------------------------
    def predict_submission(self, context: Dict, sample_df: pd.DataFrame, k_top: int = 20) -> pd.DataFrame:
        """
        Для сабмита используем global_top (без утечки). ЛОНГ-формат под sample_df.
        """
        if self.cfg.mode != "frozen_train":
            raise NotImplementedError("predict_submission is supported only for mode='frozen_train'.")

        assert self.state.global_top is not None, "Run fit() first."
        t0 = time.time()

        K = int(k_top)
        gtop = list(self.state.global_top[:K])

        users = sample_df["user_id"].values
        uniq_users, inv = np.unique(users, return_inverse=True)
        topK = np.array(gtop, dtype=np.int64)

        filled = sample_df.copy()

        # проверим равномерность K строк на пользователя
        try:
            counts = pd.Series(users).value_counts().values
            if not np.all(counts == K):
                # безопасный путь — корректно, хоть и медленнее
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

        pool = np.vstack([topK for _ in range(len(uniq_users))])
        filled["item_id"] = pool[inv, np.arange(len(inv)) % K]

        if self.verbose:
            print(f"[{self.exp_name}] predict_submission() done in {time.time()-t0:.1f}s  (rows={len(filled):,})")
        return filled
