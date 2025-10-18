# -*- coding: utf-8 -*-
"""
exp303_hard_rules — правил-бэйзд реранкер поверх скоров базовой модели.

Что делаем на каждом пуле кандидатов (val/test):
  1) забираем исходные скоры chosen source_exp (из артефактов или пересчитываем для 201/202/203);
  2) нормализуем скоры per-user (rank/minmax/zscore);
  3) считаем rule-сигналы:
      • recency(item): exp(-alpha * (last_train_day - last_day(item)))
      • pop(item): популярность за последние W дней (лог-нормализация)
      • affinity(user,item): схожесть item c недавним профилем пользователя (i2v или co-vis)
  4) blended = w_base*s0 + w_rec*rec + w_pop*pop + w_aff*affinity  (и, опц., - div_lambda*max_sim_to_selected)
  5) сортируем; логируем метрики; сохраняем артефакты; делаем сабмит.

Зависимости: pandas, numpy (+ optional: lightgbm/catboost/xgboost для пересчёта базовых скоров).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import glob
import os
import math
from collections import defaultdict

import numpy as np
import pandas as pd

# optional model libs for back-compute
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostRanker, Pool as CatPool
except Exception:
    CatBoostRanker, CatPool = None, None
try:
    from xgboost import XGBRanker
except Exception:
    XGBRanker = None

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM, COL_DATE
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.resources import FeatureResources
from vseros_b.features.schema import FEATURES_VERSION, load_schema, apply_norm, coerce_dtypes
from vseros_b.features.builders import _cand_map_to_df as cand_map_to_df
from vseros_b.candidates.io import load_compact_map


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_FEATURES: Sequence[str] = (
    "sources_basic", "pop_basic", "calendar_basic", "user_basic",
    "i2v_sim", "covis_local", "markov_basic", "regularizers_basic",
    "interact_strong", "diag_basic",
)

@dataclass
class Exp303Config:
    # откуда берём базовые скоры
    source_exp: str = "exp205_ranker_stacking"  # 'exp201_ranker_lgbm'|'exp202_ranker_catboost'|'exp203_ranker_xgb'|'exp205_ranker_stacking'|'exp301_blend_linear'
    features_for_source: Sequence[str] = DEFAULT_FEATURES
    out_tag_source: str = FEATURES_VERSION

    # нормализация исходного скора per-user
    norm_mode: str = "rank"  # 'rank'|'minmax_user'|'zscore_user'|'none'

    # rule-веса
    w_base: float = 1.0
    w_recency: float = 0.15
    w_pop: float = 0.10
    w_affinity: float = 0.25

    # recency/pop параметры
    recency_alpha: float = 0.35     # чем больше — тем резче убывание по «возрасту»
    pop_window_days: int = 7        # горизонт окна популярности

    # профиль пользователя
    user_profile_L: int = 5         # сколько последних user-айтемов брать в профиль
    sim_backend: str = "auto"       # 'auto'|'i2v'|'covis' (для affinity и diversity)

    # лёгкая диверсификация (мягкий штраф на схожесть к уже выбранным)
    div_lambda: float = 0.0         # 0 — выключено; 0.1..0.2 — лёгкий эффект

    # ускоритель
    candidate_cap_per_user: int = 200

    # метрики/сабмит
    k_list: Sequence[int] = (20, 50, 100)
    k_top_submit: int = 20

    seed: int = 42


# -----------------------------------------------------------------------------
# Helpers: load candidates & base scores
# -----------------------------------------------------------------------------

def _pick_latest_cand(exp_dir: Path, mode: str) -> Optional[Path]:
    if not exp_dir.exists(): return None
    patt = "val_candidates*.parquet" if mode == "val" else "test_candidates*.parquet"
    files = glob.glob(str(exp_dir / patt))
    if not files: return None
    files = sorted(files, key=lambda p: os.path.getsize(p))
    return Path(files[-1])

def _load_fusion_map_from_ctx_or_artifacts(ctx: dict, mode: str = "val") -> Dict[int, List[int]]:
    cand_map = ctx.get("fusion_map_" + mode) or ctx.get("cand_map_" + mode) or ctx.get("fusion_map")
    if cand_map:
        return cand_map
    base = PATHS.cand_dir / "exp108_recall_fusion"
    p = _pick_latest_cand(base, mode=mode)
    if not p:
        raise FileNotFoundError(f"[exp303] no fusion candidates for mode={mode} in {base}")
    return load_compact_map(p)

def _artifact_model_dir(exp_name: str) -> Path:
    return PATHS.artifact_dir / "models" / exp_name

def _load_scores_from_artifact(exp_name: str, mode: str) -> Optional[pd.DataFrame]:
    mdir = _artifact_model_dir(exp_name)
    p = mdir / ("val_scores.parquet" if mode == "val" else "test_scores.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        cols = set(df.columns)
        if "score" in cols:
            return df[[COL_USER, COL_ITEM, "score"]].copy()
        if "blended" in cols:
            return df[[COL_USER, COL_ITEM]].assign(score=df["blended"].astype("float32"))
    return None

# пересчёт базовых скоров для 201/202/203 при отсутствии артефактов
def _build_features(ctx: dict, user2items: Dict[int, List[int]], feature_names: Sequence[str]) -> pd.DataFrame:
    base = cand_map_to_df(user2items)
    autodiscover()
    res = FeatureResources(ctx=ctx)
    reg = select(list(feature_names))
    feats_df = base.copy()
    for _, (spec, builder) in reg.items():
        part = builder(feats_df[[COL_USER, COL_ITEM]].copy(), ctx, res)
        feats_df = feats_df.merge(part, on=[COL_USER, COL_ITEM], how="left", copy=False)
    for c in feats_df.columns:
        if c in (COL_USER, COL_ITEM): continue
        if pd.api.types.is_float_dtype(feats_df[c]): feats_df[c] = feats_df[c].fillna(0.0).astype("float32")
        else: feats_df[c] = feats_df[c].fillna(0)
    return feats_df

def _load_schema_from_model_dir(model_dir: Path) -> Dict[str, dict]:
    sp = model_dir / "feature_schema.json"
    if not sp.exists():
        feat_dir = PATHS.artifact_dir / "features" / FEATURES_VERSION
        sp = feat_dir / "feature_schema.json"
    return load_schema(sp) if sp.exists() else {}

def _predict_exp201(ctx, mode, cand_map, feats):
    if lgb is None: return None
    mdir = _artifact_model_dir("exp201_ranker_lgbm")
    mp = mdir / "model.txt"
    if not mp.exists(): return None
    booster = lgb.Booster(model_file=str(mp))
    feat_names = booster.feature_name()
    stats = _load_schema_from_model_dir(mdir)
    df = _build_features(ctx, cand_map, feats)
    df = coerce_dtypes(df, [c for c in df.columns if c not in (COL_USER, COL_ITEM)])
    df = apply_norm(df, stats)
    X = pd.DataFrame(index=df.index)
    for f in feat_names:
        X[f] = df[f] if f in df.columns else 0.0
    s = booster.predict(X, raw_score=False)
    return df[[COL_USER, COL_ITEM]].assign(score=s.astype("float32"))

def _predict_exp202(ctx, mode, cand_map, feats):
    if CatBoostRanker is None: return None
    mdir = _artifact_model_dir("exp202_ranker_catboost")
    mp = mdir / "model.cbm"
    if not mp.exists(): return None
    model = CatBoostRanker(); model.load_model(str(mp))
    feat_names = list(getattr(model, "feature_names_", []))
    stats = _load_schema_from_model_dir(mdir)
    df = _build_features(ctx, cand_map, feats)
    df = coerce_dtypes(df, [c for c in df.columns if c not in (COL_USER, COL_ITEM)])
    df = apply_norm(df, stats)
    if feat_names and all(f in df.columns for f in feat_names):
        X = df[feat_names]
    else:
        feat_cols = [c for c in df.columns if c not in (COL_USER, COL_ITEM)]
        X = df[sorted(feat_cols)]
    pool = CatPool(data=X)
    s = model.predict(pool)
    return df[[COL_USER, COL_ITEM]].assign(score=np.asarray(s, dtype="float32"))

def _predict_exp203(ctx, mode, cand_map, feats):
    if XGBRanker is None: return None
    mdir = _artifact_model_dir("exp203_ranker_xgb")
    mp = mdir / "model.json"
    if not mp.exists(): return None
    model = XGBRanker(); model.load_model(str(mp))
    feat_names = list(model.get_booster().feature_names) if model.get_booster() is not None else None
    stats = _load_schema_from_model_dir(mdir)
    df = _build_features(ctx, cand_map, feats)
    df = coerce_dtypes(df, [c for c in df.columns if c not in (COL_USER, COL_ITEM)])
    df = apply_norm(df, stats)
    if feat_names and all(f in df.columns for f in feat_names):
        X = df[feat_names]
    else:
        feat_cols = [c for c in df.columns if c not in (COL_USER, COL_ITEM)]
        X = df[sorted(feat_cols)]
    s = model.predict(X)
    return df[[COL_USER, COL_ITEM]].assign(score=np.asarray(s, dtype="float32"))

def _get_source_scores(ctx: dict, exp: str, mode: str, cand_map: Dict[int, List[int]], feats: Sequence[str]) -> pd.DataFrame:
    # 1) готовые артефакты?
    df = _load_scores_from_artifact(exp, mode)
    if df is not None:
        return df
    # 2) пересчёт (201/202/203)
    if exp == "exp201_ranker_lgbm": 
        df = _predict_exp201(ctx, mode, cand_map, feats)
    elif exp == "exp202_ranker_catboost":
        df = _predict_exp202(ctx, mode, cand_map, feats)
    elif exp == "exp203_ranker_xgb":
        df = _predict_exp203(ctx, mode, cand_map, feats)
    elif exp in ("exp205_ranker_stacking", "exp301_blend_linear"):
        raise RuntimeError(f"[exp303] {exp} must provide saved val_scores/test_scores parquet.")
    else:
        raise ValueError(f"[exp303] unknown source_exp: {exp}")
    if df is None or df.empty:
        raise RuntimeError(f"[exp303] cannot obtain scores for {exp} ({mode})")
    return df


# -----------------------------------------------------------------------------
# Rule signals: recency, popularity, affinity
# -----------------------------------------------------------------------------

def _normalize_userwise(df: pd.DataFrame, mode: str, score_col: str = "score") -> pd.Series:
    s = df[score_col].astype("float32").values
    if mode == "none":
        return df[score_col].astype("float32")
    if mode == "rank":
        out = np.empty_like(s, dtype=np.float32)
        for _, g in df.groupby(COL_USER, sort=False):
            idx = g.index.values
            vals = g[score_col].values
            order = np.argsort(-vals, kind="mergesort")
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(order))
            out[idx] = 1.0 if len(order) == 1 else (1.0 - ranks.astype(np.float32) / (len(order) - 1))
        return pd.Series(out, index=df.index, dtype="float32")
    if mode == "minmax_user":
        out = np.empty_like(s, dtype=np.float32)
        for _, g in df.groupby(COL_USER, sort=False):
            idx = g.index.values
            v = g[score_col].values.astype(np.float32)
            vmin, vmax = float(np.min(v)), float(np.max(v))
            out[idx] = 0.5 if abs(vmax - vmin) < 1e-12 else (v - vmin) / (vmax - vmin)
        return pd.Series(out, index=df.index, dtype="float32")
    if mode == "zscore_user":
        out = np.empty_like(s, dtype=np.float32)
        for _, g in df.groupby(COL_USER, sort=False):
            idx = g.index.values
            v = g[score_col].values.astype(np.float32)
            m, sd = float(np.mean(v)), float(np.std(v) + 1e-6)
            z = (v - m) / sd
            out[idx] = 1.0 / (1.0 + np.exp(-z))
        return pd.Series(out, index=df.index, dtype="float32")
    return df[score_col].astype("float32")

def _build_global_recency_pop(train_df: pd.DataFrame, pop_window_days: int, recency_alpha: float) -> Tuple[Dict[int, float], Dict[int, float], int]:
    # последний день в train
    max_day = int(train_df[COL_DATE].max())
    # last_day_by_item
    last_day = train_df.groupby(COL_ITEM, sort=False)[COL_DATE].max().astype("int32")
    last_day = {int(i): int(d) for i, d in last_day.items()}
    # recency score
    rec = {}
    for i, d in last_day.items():
        age = max(0, max_day - d)
        rec[i] = float(math.exp(-recency_alpha * age))
    # popularity in window
    lo = max_day - int(pop_window_days) + 1
    win = train_df[train_df[COL_DATE] >= lo]
    cnt = win.groupby(COL_ITEM, sort=False)[COL_USER].count().astype("int64")
    if len(cnt) == 0:
        pop = defaultdict(float)
    else:
        cmax = float(cnt.max())
        # лог-норм: log1p(c)/log1p(cmax)
        denom = math.log1p(cmax) if cmax > 0 else 1.0
        pop = {int(i): float(math.log1p(int(c)) / denom) for i, c in cnt.items()}
    return rec, pop, max_day

# --- Affinity (user,item) через i2v/CoVis ---

class _I2V:
    def __init__(self, res: FeatureResources):
        self.ok = False
        self.vecs = None
        try:
            d = res.ensure_item2vec()
            if isinstance(d, dict):
                if "item2vec" in d and isinstance(d["item2vec"], dict):
                    self.vecs = {int(k): self._norm(np.asarray(v, dtype=np.float32)) for k, v in d["item2vec"].items()}
                    self.ok = True
                elif "emb" in d and "id2idx" in d:
                    emb = np.asarray(d["emb"], dtype=np.float32)
                    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                    id2idx = {int(k): int(v) for k, v in d["id2idx"].items()}
                    self.vecs = (emb, id2idx)
                    self.ok = True
        except Exception:
            self.ok = False

    @staticmethod
    def _norm(v): 
        n = float(np.linalg.norm(v) + 1e-9); return v / n

    def vec(self, item: int) -> Optional[np.ndarray]:
        if self.vecs is None: return None
        if isinstance(self.vecs, dict):
            return self.vecs.get(int(item))
        emb, id2idx = self.vecs
        idx = id2idx.get(int(item))
        return emb[idx] if idx is not None else None

    def sim(self, a: int, b: int) -> float:
        va, vb = self.vec(a), self.vec(b)
        if va is None or vb is None: return 0.0
        s = float(np.dot(va, vb))
        return 0.5 * (s + 1.0)  # -> [0,1]

class _CoVis:
    def __init__(self, res: FeatureResources):
        self.ok = False
        self.nb: Dict[int, Dict[int, float]] = {}
        try:
            g = res.ensure_covis()
            if isinstance(g, dict):
                # нормируем веса по максимуму для узла
                for k, m in g.items():
                    k = int(k)
                    if not m: 
                        self.nb[k] = {}
                        continue
                    mx = max(float(w) for w in m.values())
                    self.nb[k] = {int(n): (float(w) / mx if mx > 0 else 0.0) for n, w in m.items()}
                self.ok = True
        except Exception:
            self.ok = False

    def sim(self, a: int, b: int) -> float:
        return float(self.nb.get(int(a), {}).get(int(b), 0.0))

def _build_user_profiles(train_df: pd.DataFrame, users: Sequence[int], L: int, i2v: Optional[_I2V], covis: Optional[_CoVis]):
    """
    Возвращает:
      profiles_i2v: dict u -> np.vector (нормированный) или None
      profiles_last_items: dict u -> List[item_id] (для covis affinity)
    """
    # последние L кликов пользователя в train
    profiles_last_items: Dict[int, List[int]] = {}
    for u, g in train_df[train_df[COL_USER].isin(users)].sort_values([COL_USER, COL_DATE]).groupby(COL_USER, sort=False):
        items = list(map(int, g[COL_ITEM].tail(int(L)).tolist()))
        profiles_last_items[int(u)] = items

    profiles_i2v: Dict[int, Optional[np.ndarray]] = {}
    if i2v is not None and i2v.ok:
        for u, items in profiles_last_items.items():
            vecs = [i2v.vec(i) for i in items]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                m = np.mean(np.stack(vecs, axis=0), axis=0)
                n = float(np.linalg.norm(m) + 1e-9)
                profiles_i2v[u] = (m / n).astype(np.float32)
            else:
                profiles_i2v[u] = None
    else:
        profiles_i2v = {int(u): None for u in profiles_last_items.keys()}

    return profiles_i2v, profiles_last_items

def _affinity_scores_for_group(u: int,
                               items: np.ndarray,
                               i2v: Optional[_I2V],
                               covis: Optional[_CoVis],
                               prof_i2v: Optional[np.ndarray],
                               prof_last_items: List[int]) -> np.ndarray:
    if prof_i2v is not None and i2v is not None and i2v.ok:
        # косинус item vs профиль → [0,1]
        out = np.zeros(len(items), dtype=np.float32)
        for t, it in enumerate(items):
            v = i2v.vec(int(it))
            if v is None:
                out[t] = 0.0
            else:
                s = float(np.dot(v, prof_i2v))
                out[t] = 0.5 * (s + 1.0)
        return out
    # CoVis: средняя близость к последним L айтемам
    if covis is not None and covis.ok and prof_last_items:
        out = np.zeros(len(items), dtype=np.float32)
        lst = list(map(int, prof_last_items))
        for t, it in enumerate(items):
            sims = [covis.sim(int(it), j) for j in lst]
            out[t] = float(np.mean(sims)) if sims else 0.0
        return out
    return np.zeros(len(items), dtype=np.float32)

# (опц.) маленькая диверсификация (штраф к схожести к уже выбранным)
def _diversity_penalty_for_group(items: np.ndarray,
                                 base_scores: np.ndarray,
                                 i2v: Optional[_I2V],
                                 covis: Optional[_CoVis],
                                 lamb: float) -> np.ndarray:
    if lamb <= 0.0:
        return base_scores
    out = np.zeros_like(base_scores, dtype=np.float32)
    # greedy: выбираем лучший, потом уменьшаем схожих
    remaining = list(range(len(items)))
    selected: List[int] = []
    scores = base_scores.astype(float).copy()
    while remaining:
        # best j
        j = int(max(remaining, key=lambda t: scores[t]))
        out[j] = scores[j]
        selected.append(j)
        remaining.remove(j)
        # штрафуем оставшихся по max-sim к выбранному (простая версия)
        for t in remaining:
            it, isel = int(items[t]), int(items[j])
            sim = 0.0
            if i2v is not None and i2v.ok:
                vi, vj = i2v.vec(it), i2v.vec(isel)
                if vi is not None and vj is not None:
                    sim = 0.5 * (float(np.dot(vi, vj)) + 1.0)
            elif covis is not None and covis.ok:
                sim = covis.sim(it, isel)
            scores[t] = float(scores[t] - lamb * sim)
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------

class Exp303HardRules(BaseExperiment):
    exp_name = "exp303_hard_rules"

    def __init__(self, cfg: Optional[Exp303Config] = None):
        self.cfg = cfg or Exp303Config()
        # кэш/стейт
        self._rec_map: Optional[Dict[int, float]] = None
        self._pop_map: Optional[Dict[int, float]] = None
        self._max_day: Optional[int] = None
        self._i2v: Optional[_I2V] = None
        self._covis: Optional[_CoVis] = None
        self._prof_i2v: Optional[Dict[int, Optional[np.ndarray]]] = None
        self._prof_last: Optional[Dict[int, List[int]]] = None

    # ------- core helpers -------

    def _get_source_scores(self, ctx: dict, mode: str) -> pd.DataFrame:
        fusion_map = _load_fusion_map_from_ctx_or_artifacts(ctx, mode=mode)
        df = _get_source_scores(ctx, self.cfg.source_exp, mode, fusion_map, self.cfg.features_for_source)
        cap = int(self.cfg.candidate_cap_per_user or 0)
        if cap > 0:
            df = (df.sort_values([COL_USER, "score"], ascending=[True, False])
                    .groupby(COL_USER, sort=False).head(cap).reset_index(drop=True))
        return df

    def _ensure_global_maps(self, ctx: dict):
        if self._rec_map is not None and self._pop_map is not None:
            return
        train_df: pd.DataFrame = ctx["train_df"]
        rec, pop, max_day = _build_global_recency_pop(
            train_df=train_df,
            pop_window_days=self.cfg.pop_window_days,
            recency_alpha=self.cfg.recency_alpha,
        )
        self._rec_map, self._pop_map, self._max_day = rec, pop, max_day

    def _ensure_profiles(self, ctx: dict, users: Sequence[int]):
        if self._prof_i2v is not None and self._prof_last is not None:
            return
        res = FeatureResources(ctx=ctx)
        self._i2v = _I2V(res)
        if not self._i2v.ok:
            self._i2v = None
        self._covis = _CoVis(res)
        if not self._covis.ok:
            self._covis = None
        train_df: pd.DataFrame = ctx["train_df"]
        self._prof_i2v, self._prof_last = _build_user_profiles(
            train_df=train_df,
            users=list(map(int, users)),
            L=int(self.cfg.user_profile_L),
            i2v=self._i2v,
            covis=self._covis,
        )

    # ------- pipeline -------

    def _rerank(self, df_scores: pd.DataFrame, ctx: dict) -> pd.DataFrame:
        self._ensure_global_maps(ctx)
        users = df_scores[COL_USER].unique()
        self._ensure_profiles(ctx, users)

        # нормализуем базовый скор per-user
        df = df_scores.copy()
        df["s_base"] = _normalize_userwise(df, mode=self.cfg.norm_mode, score_col="score").astype("float32")

        # добавим recency/pop
        rec_map = self._rec_map or {}
        pop_map = self._pop_map or {}
        df["s_rec"] = df[COL_ITEM].map(lambda i: rec_map.get(int(i), 0.0)).astype("float32")
        df["s_pop"] = df[COL_ITEM].map(lambda i: pop_map.get(int(i), 0.0)).astype("float32")

        # посчитаем affinity per-user
        out_parts = []
        for u, g in df.groupby(COL_USER, sort=False):
            items = g[COL_ITEM].values.astype("int64")
            # профиль
            prof_v = None
            if self._prof_i2v is not None:
                prof_v = self._prof_i2v.get(int(u))
            last_items = []
            if self._prof_last is not None:
                last_items = self._prof_last.get(int(u), [])
            s_aff = _affinity_scores_for_group(
                u=int(u),
                items=items,
                i2v=self._i2v,
                covis=self._covis,
                prof_i2v=prof_v,
                prof_last_items=last_items,
            )
            gg = g.copy()
            gg["s_aff"] = s_aff
            out_parts.append(gg)
        df = pd.concat(out_parts, axis=0).reset_index(drop=True)

        # итоговый rule-скор
        w = self.cfg
        df["rule_score"] = (
            w.w_base * df["s_base"] +
            w.w_recency * df["s_rec"] +
            w.w_pop * df["s_pop"] +
            w.w_affinity * df["s_aff"]
        ).astype("float32")

        # лёгкая диверсификация (штраф на схожесть к уже выбранным)
        if self.cfg.div_lambda and self.cfg.div_lambda > 0.0:
            parts = []
            for u, g in df.groupby(COL_USER, sort=False):
                items = g[COL_ITEM].values.astype("int64")
                scores = g["rule_score"].values.astype("float32")
                adj = _diversity_penalty_for_group(items, scores, self._i2v, self._covis, lamb=float(self.cfg.div_lambda))
                gg = g[[COL_USER, COL_ITEM]].copy()
                gg["rule_score"] = adj.astype("float32")
                parts.append(gg)
            df = pd.concat(parts, axis=0).reset_index(drop=True)

        return df[[COL_USER, COL_ITEM, "rule_score"]]

    # ------- API -------

    def fit(self, ctx: dict):
        self.ctx = ctx
        cfg = self.cfg

        # 1) вал-скоры базовой модели
        src_val = self._get_source_scores(ctx, mode="val")

        # 2) rule-реранк
        val_rr = self._rerank(src_val, ctx)

        # 3) метрики до/после
        truth: Dict[int, set] = ctx["val_truth"]
        base_map = (
            src_val.sort_values([COL_USER, "score"], ascending=[True, False])
                   .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        rr_map = (
            val_rr.sort_values([COL_USER, "rule_score"], ascending=[True, False])
                  .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )

        rows = []
        for k in cfg.k_list:
            rows.append({
                "k": k,
                "HR_base": recall_at_m(base_map, truth, m=k),
                "NDCG_base": ndcg_at_k(base_map, truth, k=k),
                "HR_rules": recall_at_m(rr_map, truth, m=k),
                "NDCG_rules": ndcg_at_k(rr_map, truth, k=k),
            })

        # 4) сохранения + W&B
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_scores.parquet", val_rr.rename(columns={"rule_score": "score"}))
        save_df(out_dir / "val_metrics.csv", pd.DataFrame(rows), index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            for r in rows:
                for key in ("HR_base", "NDCG_base", "HR_rules", "NDCG_rules"):
                    wb.summary[f"{key}@{r['k']}"] = r[key]

        return {"metrics": rows}

    def evaluate(self, ctx: dict):
        self.ctx = ctx
        out_dir = PATHS.artifact_dir / "models" / self.exp_name
        p = out_dir / "val_metrics.csv"
        if p.exists():
            return {"metrics_csv": str(p)}
        return {}

    def save(self, ctx: dict):
        # всё уже сохранено в fit
        return {"saved": True}

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: Optional[int] = None) -> pd.DataFrame:
        if sample_df is None or sample_df.empty:
            raise ValueError("[exp303] sample_df is required")

        self.ctx = ctx
        cfg = self.cfg
        K = int(k_top or cfg.k_top_submit)

        # 1) test-скоры источника
        src_test = self._get_source_scores(ctx, mode="test")

        # 2) rule-реранк
        test_rr = self._rerank(src_test, ctx)

        # 3) top-K → сабмит
        topk = (
            test_rr.sort_values([COL_USER, "rule_score"], ascending=[True, False])
                   .groupby(COL_USER)[COL_ITEM].apply(lambda s: " ".join(map(str, s.head(K).tolist())))
                   .reset_index().rename(columns={COL_ITEM: "items"})
        )

        sub = sample_df.copy()
        if "items" not in sub.columns:
            sub["items"] = ""
        sub = sub.drop(columns=["items"], errors="ignore").merge(topk, on=COL_USER, how="left")
        sub["items"] = sub["items"].fillna("")

        sub_dir = ensure_dir(PATHS.submission_dir)
        out_csv = sub_dir / f"sub_{self.exp_name}_k{K}.csv"
        sub.to_csv(out_csv, index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            try: wb.save(str(out_csv))
            except Exception: pass

        return sub
