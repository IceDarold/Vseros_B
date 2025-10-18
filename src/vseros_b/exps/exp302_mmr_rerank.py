# -*- coding: utf-8 -*-
"""
exp302_mmr_rerank — диверсифицирующий MMR-реранкер поверх существующих скорингов.

Что делает:
  • грузит скоры источника (exp205/exp201/exp202/exp203, либо val_scores артефакт);
  • строит индекс схожести айтемов (backend: 'auto'|'i2v'|'lgcn'|'covis'|'overlap');
  • для каждого пользователя применяет MMR: argmax λ*rel - (1-λ)*max_sim_to_selected;
  • сохраняет вал-скоры с колонкой mmr_score, метрики и сабмит.

Рекомендации:
  • λ ~ 0.6–0.8 (больше — ближе к исходному ранжированию).
  • backend='auto' сначала ищет i2v, потом lgcn, потом covis, иначе — overlap (user co-occur).

Зависимости: базовые — pandas/numpy. Для i2v/lgcn загрузка через FeatureResources если доступны.
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

from vseros_b.base_exp import BaseExperiment
from vseros_b.config import PATHS, COL_USER, COL_ITEM, COL_DATE
from vseros_b.artifacts import ensure_dir, save_df
from vseros_b.metrics import recall_at_m, ndcg_at_k
from vseros_b.features.resources import FeatureResources
from vseros_b.features.registry import autodiscover, select
from vseros_b.features.schema import FEATURES_VERSION, load_schema, apply_norm, coerce_dtypes
from vseros_b.features.builders import _cand_map_to_df as cand_map_to_df, build_matrix # noop
from vseros_b.candidates.io import load_compact_map

# опциональные модели — чтобы уметь пересчитать скоры, если нет val_scores
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


# -----------------------------------------------------------------------------
# Конфиг
# -----------------------------------------------------------------------------

@dataclass
class Exp302Config:
    # откуда берём исходные скоры
    source_exp: str = "exp205_ranker_stacking"   # 'exp201_ranker_lgbm' | 'exp202_ranker_catboost' | 'exp203_ranker_xgb' | 'exp205_ranker_stacking' | 'exp301_blend_linear'
    features_for_source: Sequence[str] = (
        "sources_basic", "pop_basic", "calendar_basic", "user_basic",
        "i2v_sim", "covis_local", "markov_basic", "regularizers_basic",
        "interact_strong", "diag_basic",
    )
    out_tag_source: str = FEATURES_VERSION

    # MMR
    mmr_lambda: float = 0.7             # баланс релевантность/диверсификация
    k_top_submit: int = 20
    candidate_cap_per_user: int = 200   # можно ограничить кол-во кандидатов перед MMR (ускоряет)

    # Backend схожести айтемов
    sim_backend: str = "auto"           # 'auto'|'i2v'|'lgcn'|'covis'|'overlap'
    # для 'overlap' (user co-occur)
    overlap_min_users: int = 3          # игнорить айтемы с очень малым числом пользователей в train

    # нормировка исходных скоров per-user перед MMR
    norm_mode: str = "rank"             # 'none'|'rank'|'minmax_user'|'zscore_user'

    # метрики
    k_list: Sequence[int] = (20, 50, 100)

    seed: int = 42


# -----------------------------------------------------------------------------
# Загрузка/пересчёт исходных скоров
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
        raise FileNotFoundError(f"[exp302] no fusion candidates for mode={mode} in {base}")
    return load_compact_map(p)

def _artifact_model_dir(exp_name: str) -> Path:
    return PATHS.artifact_dir / "models" / exp_name

def _load_scores_from_artifact(exp_name: str, mode: str) -> Optional[pd.DataFrame]:
    # Сначала ищем явные score-таблицы
    mdir = _artifact_model_dir(exp_name)
    p1 = mdir / ("val_scores.parquet" if mode == "val" else "test_scores.parquet")
    if p1.exists():
        df = pd.read_parquet(p1)
        if "score" in df.columns:
            return df[[COL_USER, COL_ITEM, "score"]].copy()
        if "blended" in df.columns:  # для blend
            return df[[COL_USER, COL_ITEM]].assign(score=df["blended"].astype("float32"))
    # Для стэкинга бывает только val_scores — для test пересчёт не делаем здесь.
    return None

# пересчёт для базовых ранкеров, если нет val_scores/test_scores
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
    # 2) пересчёт (только базовые деревяшки)
    if exp == "exp201_ranker_lgbm": 
        df = _predict_exp201(ctx, mode, cand_map, feats)
    elif exp == "exp202_ranker_catboost":
        df = _predict_exp202(ctx, mode, cand_map, feats)
    elif exp == "exp203_ranker_xgb":
        df = _predict_exp203(ctx, mode, cand_map, feats)
    elif exp in ("exp205_ranker_stacking", "exp301_blend_linear"):
        raise RuntimeError(f"[exp302] {exp} must provide saved val_scores/test_scores parquet.")
    else:
        raise ValueError(f"[exp302] unknown source_exp: {exp}")
    if df is None or df.empty:
        raise RuntimeError(f"[exp302] cannot obtain scores for {exp} ({mode})")
    return df


# -----------------------------------------------------------------------------
# Индексы схожести айтемов
# -----------------------------------------------------------------------------

class _SimIndex:
    def sim(self, i: int, j: int) -> float:
        return 0.0

class _I2VSim(_SimIndex):
    def __init__(self, res: FeatureResources):
        ok = False
        self.vecs = None
        try:
            d = res.ensure_item2vec()  # ожидается {'item2vec': {item_id: np.array(...)} } либо {'emb':..., 'map':...}
            if isinstance(d, dict):
                if "item2vec" in d and isinstance(d["item2vec"], dict):
                    self.vecs = {int(k): self._norm(np.asarray(v, dtype=np.float32)) for k, v in d["item2vec"].items()}
                    ok = True
                elif "emb" in d and "id2idx" in d:
                    emb = np.asarray(d["emb"], dtype=np.float32)
                    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                    id2idx = {int(k): int(v) for k, v in d["id2idx"].items()}
                    self.vecs = (emb, id2idx)
                    ok = True
        except Exception:
            ok = False
        self.ok = ok

    @staticmethod
    def _norm(v): 
        n = float(np.linalg.norm(v) + 1e-9); return v / n

    def _vec(self, item: int):
        if self.vecs is None: return None
        if isinstance(self.vecs, dict):
            return self.vecs.get(int(item))
        emb, id2idx = self.vecs
        idx = id2idx.get(int(item))
        return emb[idx] if idx is not None else None

    def sim(self, i: int, j: int) -> float:
        vi, vj = self._vec(i), self._vec(j)
        if vi is None or vj is None: return 0.0
        s = float(np.dot(vi, vj))
        # map [-1,1] -> [0,1] мягко
        return 0.5 * (s + 1.0)

class _LGCNSim(_SimIndex):
    def __init__(self, res: FeatureResources):
        ok = False
        self.vecs = None
        try:
            d = res.ensure_lgcn()  # ожидается {'item_emb': np.ndarray, 'id2idx': dict}
            if isinstance(d, dict) and "item_emb" in d and "id2idx" in d:
                emb = np.asarray(d["item_emb"], dtype=np.float32)
                emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                id2idx = {int(k): int(v) for k, v in d["id2idx"].items()}
                self.vecs = (emb, id2idx)
                ok = True
        except Exception:
            ok = False
        self.ok = ok

    def sim(self, i: int, j: int) -> float:
        if not self.ok or self.vecs is None: return 0.0
        emb, id2idx = self.vecs
        ii, jj = id2idx.get(int(i)), id2idx.get(int(j))
        if ii is None or jj is None: return 0.0
        s = float(np.dot(emb[ii], emb[jj]))
        return 0.5 * (s + 1.0)

class _CoVisSim(_SimIndex):
    def __init__(self, res: FeatureResources):
        ok = False
        self.nb = None
        try:
            g = res.ensure_covis()  # ожидается dict {item: {nbr: weight}}
            if isinstance(g, dict): 
                self.nb = {int(k): {int(n): float(w) for n, w in v.items()} for k, v in g.items()}
                # нормируем в [0,1] по каждому узлу
                for k, m in self.nb.items():
                    if not m: continue
                    mx = max(m.values())
                    if mx > 0:
                        for n in list(m.keys()):
                            m[n] = m[n] / mx
                ok = True
        except Exception:
            ok = False
        self.ok = ok

    def sim(self, i: int, j: int) -> float:
        if not self.ok or self.nb is None: return 0.0
        return float(self.nb.get(int(i), {}).get(int(j), 0.0))

class _OverlapSim(_SimIndex):
    """
    Сходство по пересечению множеств пользователей в train.
    sim(i,j) = |U_i ∩ U_j| / sqrt(|U_i|*|U_j|)  (cosine на бинарной инцидентности)
    """
    def __init__(self, train_df: pd.DataFrame, needed_items: set[int], min_users: int = 3):
        # соберём U(i) только для нужных айтемов
        filt = train_df[train_df[COL_ITEM].isin(list(needed_items))][[COL_USER, COL_ITEM]]
        # item -> set(users)
        self.u_sets: Dict[int, set] = {}
        for it, g in filt.groupby(COL_ITEM, sort=False):
            us = set(map(int, g[COL_USER].unique()))
            if len(us) >= max(1, int(min_users)):
                self.u_sets[int(it)] = us
        self.cache: Dict[Tuple[int, int], float] = {}

    def sim(self, i: int, j: int) -> float:
        a, b = int(i), int(j)
        if a == b: return 1.0
        key = (a, b) if a <= b else (b, a)
        if key in self.cache: return self.cache[key]
        Ai = self.u_sets.get(a); Aj = self.u_sets.get(b)
        if Ai is None or Aj is None:
            self.cache[key] = 0.0; return 0.0
        inter = len(Ai & Aj)
        denom = math.sqrt(len(Ai) * len(Aj)) + 1e-9
        s = float(inter / denom)
        # клиппинг в [0,1]
        s = max(0.0, min(1.0, s))
        self.cache[key] = s
        return s


def _make_sim_index(ctx: dict, backend: str, items_needed: set[int]) -> _SimIndex:
    backend = (backend or "auto").lower()
    res = FeatureResources(ctx=ctx)

    if backend in ("i2v", "auto"):
        try:
            idx = _I2VSim(res)
            if getattr(idx, "ok", False):
                return idx
        except Exception:
            pass

    if backend in ("lgcn", "auto"):
        try:
            idx = _LGCNSim(res)
            if getattr(idx, "ok", False):
                return idx
        except Exception:
            pass

    if backend in ("covis", "auto"):
        try:
            idx = _CoVisSim(res)
            if getattr(idx, "ok", False):
                return idx
        except Exception:
            pass

    # fallback — user overlap на TRAIN
    train_df: pd.DataFrame = ctx["train_df"]
    return _OverlapSim(train_df=train_df, needed_items=items_needed, min_users=int(ctx.get("features_cfg", {}).get("feat_mmr", {}).get("overlap_min_users", 3)))


# -----------------------------------------------------------------------------
# Нормализация per-user
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


# -----------------------------------------------------------------------------
# MMR
# -----------------------------------------------------------------------------

def _mmr_for_user(items: np.ndarray, rels: np.ndarray, top_k: int, sim_index: _SimIndex, lamb: float) -> np.ndarray:
    """
    Greedy MMR: возвращает порядок индексов (из локального массива) для top_k.
    items: массив item_id (int)
    rels:  массив релевантностей (float) (нормированные исходные скоры)
    """
    n = len(items)
    if n <= top_k:
        return np.arange(n, dtype=np.int32)

    selected = []
    remaining = list(range(n))
    # стартуем с лучшего по релевантности
    first = int(np.argmax(rels))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < top_k and remaining:
        best_j = None
        best_val = -1e18
        # подготовим max similarity к уже выбранным
        for j in remaining:
            it_j = int(items[j])
            # max sim to any selected
            max_sim = 0.0
            for s in selected:
                it_s = int(items[s])
                sim = sim_index.sim(it_j, it_s)
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= 1.0:
                        break
            val = lamb * float(rels[j]) - (1.0 - lamb) * max_sim
            if val > best_val:
                best_val = val
                best_j = j
        selected.append(best_j)
        remaining.remove(best_j)

    # дописываем оставшиеся в порядке релевантности
    if len(selected) < n:
        rest = [j for j in remaining]
        rest_sorted = sorted(rest, key=lambda j: float(rels[j]), reverse=True)
        selected.extend(rest_sorted)
    return np.asarray(selected, dtype=np.int32)


# -----------------------------------------------------------------------------
# Эксперимент
# -----------------------------------------------------------------------------

class Exp302MMRRerank(BaseExperiment):
    exp_name = "exp302_mmr_rerank"

    def __init__(self, cfg: Optional[Exp302Config] = None):
        self.cfg = cfg or Exp302Config()
        self.weights: Optional[dict] = None  # резерв под будущее
        self.stats: Optional[dict] = None

    # ------------- ядро -------------

    def _get_source_scores(self, ctx: dict, mode: str) -> pd.DataFrame:
        fusion_map = _load_fusion_map_from_ctx_or_artifacts(ctx, mode=mode)
        df = _get_source_scores(ctx, self.cfg.source_exp, mode, fusion_map, self.cfg.features_for_source)
        # (опц.) cap кандидатов по юзеру — ускоряет MMR
        cap = int(self.cfg.candidate_cap_per_user or 0)
        if cap > 0:
            df = (df.sort_values([COL_USER, "score"], ascending=[True, False])
                    .groupby(COL_USER).head(cap).reset_index(drop=True))
        return df

    def _apply_mmr(self, df_scores: pd.DataFrame, ctx: dict, mode: str) -> pd.DataFrame:
        # нормируем пер юзер (по конфигу)
        df = df_scores.copy()
        df["rel"] = _normalize_userwise(df, mode=self.cfg.norm_mode, score_col="score").astype("float32")

        # подготовим индекс схожести для айтемов, встречающихся в пуле
        needed_items = set(map(int, df[COL_ITEM].unique()))
        sim_idx = _make_sim_index(ctx, self.cfg.sim_backend, items_needed=needed_items)

        # прогон по пользователям
        out_parts = []
        for u, g in df.groupby(COL_USER, sort=False):
            items = g[COL_ITEM].values.astype("int64")
            rels  = g["rel"].values.astype("float32")
            order = _mmr_for_user(items, rels, top_k=max(self.cfg.k_list), sim_index=sim_idx, lamb=float(self.cfg.mmr_lambda))
            gg = g.iloc[order].copy()
            gg["mmr_score"] = np.linspace(1.0, 0.0, num=len(gg), dtype=np.float32)  # псевдо-скор по порядку
            out_parts.append(gg[[COL_USER, COL_ITEM, "mmr_score"]])
        mmr_df = pd.concat(out_parts, axis=0).reset_index(drop=True)
        return mmr_df

    # ------------- API -------------

    def fit(self, ctx: dict):
        self.ctx = ctx
        cfg = self.cfg

        # 1) вал-скоры источника
        src_val = self._get_source_scores(ctx, mode="val")

        # 2) применим MMR
        mmr_val = self._apply_mmr(src_val, ctx, mode="val")

        # 3) метрики до/после
        truth: Dict[int, set] = ctx["val_truth"]

        # до
        base_map = (
            src_val.sort_values([COL_USER, "score"], ascending=[True, False])
                   .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )
        # после
        mmr_map = (
            mmr_val.sort_values([COL_USER, "mmr_score"], ascending=[True, False])
                   .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )

        rows = []
        for k in cfg.k_list:
            rows.append({
                "k": k,
                "HR_base": recall_at_m(base_map, truth, m=k),
                "NDCG_base": ndcg_at_k(base_map, truth, k=k),
                "HR_mmr": recall_at_m(mmr_map, truth, m=k),
                "NDCG_mmr": ndcg_at_k(mmr_map, truth, k=k),
            })

        # 4) лог и сохранения
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_scores.parquet", mmr_val.rename(columns={"mmr_score": "score"}))
        save_df(out_dir / "val_metrics.csv", pd.DataFrame(rows), index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            for r in rows:
                for key in ("HR_base", "NDCG_base", "HR_mmr", "NDCG_mmr"):
                    wb.summary[f"{key}@{r['k']}"] = r[key]

        return {"metrics": rows}

    def evaluate(self, ctx: dict):
        # метрики уже посчитали в fit()
        self.ctx = ctx
        out_dir = PATHS.artifact_dir / "models" / self.exp_name
        p = out_dir / "val_metrics.csv"
        if p.exists():
            return {"metrics_csv": str(p)}
        return {}

    def save(self, ctx: dict):
        # всё сохранили в fit
        return {"saved": True}

    def predict_submission(self, ctx: dict, sample_df: pd.DataFrame, k_top: Optional[int] = None) -> pd.DataFrame:
        if sample_df is None or sample_df.empty:
            raise ValueError("[exp302] sample_df is required")

        self.ctx = ctx
        cfg = self.cfg
        K = int(k_top or cfg.k_top_submit)

        # 1) test-скоры источника
        src_test = self._get_source_scores(ctx, mode="test")

        # 2) MMR
        mmr_test = self._apply_mmr(src_test, ctx, mode="test")

        # 3) top-K → сабмит
        topk = (
            mmr_test.sort_values([COL_USER, "mmr_score"], ascending=[True, False])
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
