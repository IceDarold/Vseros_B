# -*- coding: utf-8 -*-
"""
exp304_fallbacks — безопасный донабор top-K, когда базовая модель вернула мало кандидатов.

Пайплайн:
  1) Берём исходный скор-лист (source_exp: exp201/202/203 умеем пересчитать; exp205/exp301 — из артефактов val/test_scores).
  2) Для валидации считаем метрики «до».
  3) Строим fallback-пулы (по train):
       - per_user_covis: соседи co-vis для последних L айтемов пользователя.
       - per_user_i2v: (если есть эмбеддинги) соседи в i2v среди top-M глобально популярных.
       - global_trending: топ по последним W_trend дням.
       - global_pop: топ по последним W_pop дням.
     Исключаем уже отобранные, и (опционально) уже виденные юзером в train (exclude_seen=True).
  4) Донабираем до K в указанном порядке (topup_order).
  5) Сохраняем val_scores/val_metrics и делаем сабмит.

Замечание:
  • Рэнкинг базовой модели не «портим» — fallback только добавляет недостающее.
  • Для exp205/exp301 нужен сохранённый val_scores/test_scores.parquet.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import glob
import os
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# optional базовые модели для пересчёта скоров (exp201/202/203)
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


# ==============================
# Config
# ==============================

DEFAULT_FEATURES: Sequence[str] = (
    "sources_basic", "pop_basic", "calendar_basic", "user_basic",
    "i2v_sim", "covis_local", "markov_basic", "regularizers_basic",
    "interact_strong", "diag_basic",
)

@dataclass
class Exp304Config:
    # базовый источник скоров
    source_exp: str = "exp205_ranker_stacking"  # 'exp201_ranker_lgbm'|'exp202_ranker_catboost'|'exp203_ranker_xgb'|'exp205_ranker_stacking'|'exp301_blend_linear'
    features_for_source: Sequence[str] = DEFAULT_FEATURES
    out_tag_source: str = FEATURES_VERSION

    # топап порядок
    topup_order: Sequence[str] = ("per_user_covis", "per_user_i2v", "global_trending", "global_pop")

    # лимиты
    k_top_submit: int = 20
    candidate_cap_per_user: int = 300     # ограничиваем размер пула перед fallback (ускоряет)

    # окна/параметры глобальных сигналов (по train)
    trend_window_days: int = 3
    pop_window_days: int = 7

    # профиль пользователя
    user_profile_L: int = 5               # последние L интеракций из train
    i2v_pop_universe: int = 1000          # среди каких global-pop искать i2v соседей
    i2v_neighbors_per_item: int = 50      # сколько соседей брать от каждого последнего айтема

    # исключения
    exclude_seen: bool = True             # не рекомендовать то, что юзер уже видел в train

    # метрики
    k_list: Sequence[int] = (20, 50, 100)

    seed: int = 42


# ==============================
# Вспомогалки: кандидаты и скоры
# ==============================

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
        raise FileNotFoundError(f"[exp304] no fusion candidates for mode={mode} in {base}")
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
        if "blended" in cols:  # для blend
            return df[[COL_USER, COL_ITEM]].assign(score=df["blended"].astype("float32"))
    return None

# если артефакта нет — пересчёт для 201/202/203
def _build_features(ctx: dict, user2items: Dict[int, List[int]], feature_names: Sequence[str]) -> pd.DataFrame:
    base = cand_map_to_df(user2items)
    autodiscover()
    from vseros_b.features.resources import FeatureResources
    res = FeatureResources(ctx=ctx)
    from vseros_b.features.registry import select
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
    # 1) пробуем готовые артефакты
    df = _load_scores_from_artifact(exp, mode)
    if df is not None:
        return df
    # 2) пересчёт (только базовые деревья)
    if exp == "exp201_ranker_lgbm":
        df = _predict_exp201(ctx, mode, cand_map, feats)
    elif exp == "exp202_ranker_catboost":
        df = _predict_exp202(ctx, mode, cand_map, feats)
    elif exp == "exp203_ranker_xgb":
        df = _predict_exp203(ctx, mode, cand_map, feats)
    elif exp in ("exp205_ranker_stacking", "exp301_blend_linear"):
        raise RuntimeError(f"[exp304] {exp} must provide saved val/test score parquet in artifacts.")
    else:
        raise ValueError(f"[exp304] unknown source_exp: {exp}")
    if df is None or df.empty:
        raise RuntimeError(f"[exp304] cannot obtain scores for {exp} ({mode})")
    return df


# ==============================
# Глобальные списки (по train)
# ==============================

def _global_top_by_window(train_df: pd.DataFrame, window_days: int) -> List[int]:
    if len(train_df) == 0:
        return []
    max_day = int(train_df[COL_DATE].max())
    lo = max_day - int(window_days) + 1
    df = train_df[train_df[COL_DATE] >= lo]
    cnt = df.groupby(COL_ITEM, sort=False)[COL_USER].count().astype("int64")
    order = cnt.sort_values(ascending=False).index.values.tolist()
    return list(map(int, order))

def _users_seen_map(train_df: pd.DataFrame) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for u, g in train_df.groupby(COL_USER, sort=False):
        out[int(u)] = set(map(int, g[COL_ITEM].unique()))
    return out

def _users_last_items(train_df: pd.DataFrame, L: int) -> Dict[int, List[int]]:
    # последние L айтемов по пользователю
    base = train_df.sort_values([COL_USER, COL_DATE])
    out: Dict[int, List[int]] = {}
    for u, g in base.groupby(COL_USER, sort=False):
        out[int(u)] = list(map(int, g[COL_ITEM].tail(int(L)).tolist()))
    return out


# ==============================
# CoVis / i2v ресурсы
# ==============================

class _CoVisIndex:
    def __init__(self, res: FeatureResources):
        self.ok = False
        self.nb: Dict[int, Dict[int, float]] = {}
        try:
            raw = res.ensure_covis()
            if isinstance(raw, dict):
                # нормируем веса 0..1 по максимуму узла
                for k, m in raw.items():
                    k = int(k)
                    if not m: 
                        self.nb[k] = {}
                        continue
                    mx = max(float(w) for w in m.values())
                    self.nb[k] = {int(n): (float(w)/mx if mx > 0 else 0.0) for n, w in m.items()}
                self.ok = True
        except Exception:
            self.ok = False

    def neighbors(self, item: int, topk: int) -> List[int]:
        if not self.ok: return []
        m = self.nb.get(int(item), {})
        if not m: return []
        # уже отсортировано после нормировки не гарантировано — отсортируем
        return [j for j, _ in sorted(m.items(), key=lambda t: t[1], reverse=True)[:topk]]

class _I2VIndex:
    """
    Лёгкий i2v: используем эмбеддинги, но ищем похожие только среди заданного 'universe' (обычно топ-поп).
    """
    def __init__(self, res: FeatureResources):
        self.ok = False
        self.emb = None
        self.id2idx = None
        try:
            d = res.ensure_item2vec()
            if isinstance(d, dict):
                if "emb" in d and "id2idx" in d:
                    emb = np.asarray(d["emb"], dtype=np.float32)
                    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                    self.emb = emb
                    self.id2idx = {int(k): int(v) for k, v in d["id2idx"].items()}
                    self.ok = True
                elif "item2vec" in d and isinstance(d["item2vec"], dict):
                    # соберём массив
                    keys = sorted([int(k) for k in d["item2vec"].keys()])
                    mat = np.stack([np.asarray(d["item2vec"][str(k)] if str(k) in d["item2vec"] else d["item2vec"][k], dtype=np.float32) for k in keys], axis=0)
                    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
                    self.emb = mat
                    self.id2idx = {int(k): i for i, k in enumerate(keys)}
                    self.ok = True
        except Exception:
            self.ok = False

    def vec(self, item: int) -> Optional[np.ndarray]:
        if not self.ok: return None
        idx = self.id2idx.get(int(item)) if self.id2idx is not None else None
        if idx is None: return None
        return self.emb[idx]

    def top_similar_in_universe(self, item: int, universe: List[int], topk: int) -> List[int]:
        v = self.vec(item)
        if (v is None) or (not universe): 
            return []
        # косинус ~ скалярное произведение (u уже нормирован)
        sims = []
        for j in universe:
            w = self.vec(j)
            if w is None: 
                sims.append(-2.0)
            else:
                sims.append(float(np.dot(v, w)))
        order = np.argsort(-np.asarray(sims))[:topk]
        # маппим в [0, len-1] — уже сделали
        return [int(universe[idx]) for idx in order.tolist() if sims[idx] > -1.0]


# ==============================
# Fallback сборка
# ==============================

def _build_fallback_pool_for_user(
    u: int,
    have: List[int],
    need: int,
    order: Sequence[str],
    last_items: List[int],
    seen_items: set,
    covis_idx: Optional[_CoVisIndex],
    i2v_idx: Optional[_I2VIndex],
    i2v_universe: List[int],
    trend_list: List[int],
    pop_list: List[int],
) -> List[int]:
    """Возвращает список айтемов fallback (без повторов), длиной <= need."""
    out: List[int] = []
    have_set = set(map(int, have))
    seen = set(map(int, seen_items or set()))
    def push(cands: List[int]):
        nonlocal out, have_set, seen
        for it in cands:
            ii = int(it)
            if ii in have_set: 
                continue
            if ii in out:
                continue
            if ii in seen:
                continue
            out.append(ii)
            if len(out) >= need:
                return True
        return False

    for src in order:
        if len(out) >= need:
            break
        if src == "per_user_covis" and covis_idx is not None and covis_idx.ok and last_items:
            buf = []
            for li in last_items:
                buf.extend(covis_idx.neighbors(int(li), topk=50))
            # частотная агрегация + порядок по частоте
            if buf:
                counts = Counter(map(int, buf))
                cands = [it for it, _ in counts.most_common()]
                if push(cands):
                    break

        elif src == "per_user_i2v" and i2v_idx is not None and i2v_idx.ok and last_items and i2v_universe:
            buf = []
            # из каждого последнего — top-N внутри поп-универсума
            for li in last_items:
                topn = i2v_idx.top_similar_in_universe(int(li), i2v_universe, topk=50)
                buf.extend(topn)
            if buf:
                counts = Counter(map(int, buf))
                cands = [it for it, _ in counts.most_common()]
                if push(cands):
                    break

        elif src == "global_trending" and trend_list:
            if push(trend_list):
                break

        elif src == "global_pop" and pop_list:
            if push(pop_list):
                break

        else:
            # неизвестный источник — пропускаем
            continue

    return out[:need]


# ==============================
# Эксперимент
# ==============================

class Exp304Fallbacks(BaseExperiment):
    exp_name = "exp304_fallbacks"

    def __init__(self, cfg: Optional[Exp304Config] = None):
        self.cfg = cfg or Exp304Config()
        # кэш
        self._trend_list: Optional[List[int]] = None
        self._pop_list: Optional[List[int]] = None
        self._seen_map: Optional[Dict[int, set]] = None
        self._last_items: Optional[Dict[int, List[int]]] = None
        self._covis: Optional[_CoVisIndex] = None
        self._i2v: Optional[_I2VIndex] = None
        self._i2v_universe: Optional[List[int]] = None

    # ------------- helpers -------------

    def _get_source_scores(self, ctx: dict, mode: str) -> pd.DataFrame:
        fusion_map = _load_fusion_map_from_ctx_or_artifacts(ctx, mode=mode)
        df = _get_source_scores(ctx, self.cfg.source_exp, mode, fusion_map, self.cfg.features_for_source)
        cap = int(self.cfg.candidate_cap_per_user or 0)
        if cap > 0:
            df = (df.sort_values([COL_USER, "score"], ascending=[True, False])
                    .groupby(COL_USER, sort=False).head(cap).reset_index(drop=True))
        return df

    def _prepare_train_side_structs(self, ctx: dict):
        if self._trend_list is not None:
            return
        train_df: pd.DataFrame = ctx["train_df"]
        # глобальные списки
        self._trend_list = _global_top_by_window(train_df, self.cfg.trend_window_days)
        self._pop_list = _global_top_by_window(train_df, self.cfg.pop_window_days)
        # seen и профили
        self._seen_map = _users_seen_map(train_df) if self.cfg.exclude_seen else defaultdict(set)
        self._last_items = _users_last_items(train_df, L=int(self.cfg.user_profile_L))
        # индексы
        res = FeatureResources(ctx=ctx)
        self._covis = _CoVisIndex(res)
        self._i2v = _I2VIndex(res)
        # i2v вселенная — ограничим top-M по глобальной популярности
        M = int(self.cfg.i2v_pop_universe)
        base_uni = self._pop_list or []
        self._i2v_universe = base_uni[:M] if M > 0 else base_uni

    def _topup_users(self, base_df: pd.DataFrame, K: int) -> Tuple[pd.DataFrame, dict]:
        """
        На вход: base_df [user_id, item_id, score]; 
        Возврат: финальный DF [user_id, item_id, score, src], и счётчики по источникам.
        """
        cfg = self.cfg
        self._prepare_train_side_structs(self.ctx)
        trend = self._trend_list or []
        pop = self._pop_list or []
        seen_map = self._seen_map or defaultdict(set)
        last_map = self._last_items or {}
        covis_idx = self._covis
        i2v_idx = self._i2v
        i2v_uni = self._i2v_universe or []

        out_parts = []
        stats = Counter()

        # нормализуем базовые скоры per-user → чтобы у fallback были «меньше»
        base_sorted = (base_df.sort_values([COL_USER, "score"], ascending=[True, False])
                            .groupby(COL_USER, sort=False)
                            .apply(lambda g: g.reset_index(drop=True)).reset_index(drop=True))

        # базовый блок + донабор
        for u, g in base_sorted.groupby(COL_USER, sort=False):
            u = int(u)
            items = list(map(int, g[COL_ITEM].tolist()))
            # уже выбранные
            picked = list(items[:K])
            if len(picked) < K:
                need = K - len(picked)
                fb = _build_fallback_pool_for_user(
                    u=u,
                    have=picked,
                    need=need,
                    order=cfg.topup_order,
                    last_items=last_map.get(u, []),
                    seen_items=seen_map.get(u, set()),
                    covis_idx=covis_idx,
                    i2v_idx=i2v_idx,
                    i2v_universe=i2v_uni,
                    trend_list=trend,
                    pop_list=pop,
                )
                # учтём статистику источников (грубо: по порядку topup_order)
                if fb:
                    # пометим одним тегом — откуда «первый» пришёл (для грубой статистики)
                    for it in fb:
                        # определяем источник для каждой позиции
                        src_tag = None
                        for src in cfg.topup_order:
                            if src == "per_user_covis" and covis_idx is not None and it in (covis_idx.nb.get(it, {}) or {}).keys():
                                src_tag = "per_user_covis"; break
                        if src_tag is None and i2v_idx is not None and i2v_idx.ok:
                            src_tag = "per_user_i2v"
                        if src_tag is None and it in trend:
                            src_tag = "global_trending"
                        if src_tag is None:
                            src_tag = "global_pop" if it in pop else "fallback"
                        stats[src_tag] += 1
                    picked.extend(fb)

            # соберём результирующую таблицу с «псевдо-скором» по позиции
            take = picked[:K]
            # отдаём монотонно убывающий скор — чтобы порядок зафиксировать
            scores = np.linspace(1.0, 0.0, num=len(take), dtype=np.float32)
            part = pd.DataFrame({COL_USER: u, COL_ITEM: take, "score": scores})
            out_parts.append(part)

        out_df = pd.concat(out_parts, axis=0).reset_index(drop=True)
        return out_df, dict(stats)

    # ------------- API -------------

    def fit(self, ctx: dict):
        self.ctx = ctx
        cfg = self.cfg

        # 1) вал-скоры базового источника
        src_val = self._get_source_scores(ctx, mode="val")

        # 2) метрики «до»
        truth: Dict[int, set] = ctx["val_truth"]
        base_map = (
            src_val.sort_values([COL_USER, "score"], ascending=[True, False])
                   .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )

        base_rows = []
        for k in cfg.k_list:
            base_rows.append({
                "k": k,
                "HR_base": recall_at_m(base_map, truth, m=k),
                "NDCG_base": ndcg_at_k(base_map, truth, k=k),
            })

        # 3) Фоллбек донабор
        final_val, usage = self._topup_users(src_val, K=int(cfg.k_top_submit))

        # 4) метрики «после»
        final_map = (
            final_val.sort_values([COL_USER, "score"], ascending=[True, False])
                     .groupby(COL_USER)[COL_ITEM].apply(list).to_dict()
        )

        rows = []
        for k in cfg.k_list:
            rows.append({
                "k": k,
                "HR_base": next((r["HR_base"] for r in base_rows if r["k"] == k), np.nan),
                "NDCG_base": next((r["NDCG_base"] for r in base_rows if r["k"] == k), np.nan),
                "HR_final": recall_at_m(final_map, truth, m=k),
                "NDCG_final": ndcg_at_k(final_map, truth, k=k),
            })

        # 5) сохранение
        out_dir = ensure_dir(PATHS.artifact_dir / "models" / self.exp_name)
        save_df(out_dir / "val_scores.parquet", final_val)
        save_df(out_dir / "val_metrics.csv", pd.DataFrame(rows), index=False)
        (out_dir / "usage.json").write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")

        # W&B
        wb = ctx.get("wandb_run")
        if wb is not None:
            wb.config.update({"exp": self.exp_name, **asdict(cfg)}, allow_val_change=True)
            for r in rows:
                wb.summary[f"HR_base@{r['k']}"] = r["HR_base"]
                wb.summary[f"NDCG_base@{r['k']}"] = r["NDCG_base"]
                wb.summary[f"HR_final@{r['k']}"] = r["HR_final"]
                wb.summary[f"NDCG_final@{r['k']}"] = r["NDCG_final"]
            # usage
            for k, v in usage.items():
                wb.summary[f"fallback_used_{k}"] = int(v)

        return {"metrics": rows, "usage": usage}

    def evaluate(self, ctx: dict):
        # метрики уже посчитали в fit
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
            raise ValueError("[exp304] sample_df is required")

        self.ctx = ctx
        K = int(k_top or self.cfg.k_top_submit)

        # 1) test-скоры источника
        src_test = self._get_source_scores(ctx, mode="test")

        # 2) донабор
        final_test, usage = self._topup_users(src_test, K=K)

        # 3) сабмит
        topk = (
            final_test.sort_values([COL_USER, "score"], ascending=[True, False])
                      .groupby(COL_USER)[COL_ITEM]
                      .apply(lambda s: " ".join(map(str, s.head(K).tolist())))
                      .reset_index().rename(columns={COL_ITEM: "items"})
        )

        sub = sample_df.copy()
        if "items" not in sub.columns:
            sub["items"] = ""
        sub = sub.drop(columns=["items"], errors="ignore").merge(topk, on="user_id", how="left")
        sub["items"] = sub["items"].fillna("")

        sub_dir = ensure_dir(PATHS.submission_dir)
        out_csv = sub_dir / f"sub_{self.exp_name}_k{K}.csv"
        sub.to_csv(out_csv, index=False)

        wb = ctx.get("wandb_run")
        if wb is not None:
            try:
                wb.save(str(out_csv))
                wb.summary["fallback_test_used_total"] = int(sum(usage.values()))
            except Exception:
                pass

        return sub
