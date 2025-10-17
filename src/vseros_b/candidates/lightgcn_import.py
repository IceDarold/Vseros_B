# -*- coding: utf-8 -*-
"""
lightgcn_import.py — импорт эмбеддингов LightGCN и ранжирование пользователей.

Что умеет:
- загрузка user_emb.npy / item_emb.npy + user_map.parquet / item_map.parquet (локально или из W&B)
- преобразование user_id → u_idx, item_id → i_idx (и обратно)
- ранжирование per-user по dot-product (батчами), опционально на ограниченном пуле item индексов
- фильтрация уже виденных в train айтемов (exclude_seen)
- конвертация выдачи из i_idx в item_id

Опционально: FAISS ускоряет topM без пула (IndexFlatIP). Для варианта с пулом используем батчевый матмул.

Зависимости: numpy, pandas (обяз.), (опц.) wandb, faiss.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# мягкие импорты
try:
    import wandb  # type: ignore
    _WANDB = True
except Exception:
    wandb = None  # type: ignore
    _WANDB = False

try:
    import faiss  # type: ignore
    _FAISS = True
except Exception:
    faiss = None  # type: ignore
    _FAISS = False

from ..config import COL_USER, COL_ITEM, PATHS


# ============================== модели данных ==============================

@dataclass
class LGCNEmbeddings:
    user_emb: np.ndarray     # [U, D], float32
    item_emb: np.ndarray     # [I, D], float32
    user_map: pd.DataFrame   # columns: [user_id, u_idx]
    item_map: pd.DataFrame   # columns: [item_id, i_idx]

    def dims(self) -> Tuple[int, int, int]:
        U = int(self.user_emb.shape[0])
        I = int(self.item_emb.shape[0])
        D = int(self.user_emb.shape[1])
        return U, I, D


@dataclass
class ImportConfig:
    # локальные пути (если заданы — используем их)
    user_emb_path: Optional[Union[str, Path]] = None
    item_emb_path: Optional[Union[str, Path]] = None
    user_map_path: Optional[Union[str, Path]] = None
    item_map_path: Optional[Union[str, Path]] = None

    # или W&B artifact (project/name:version). Например: "tshopping_stage1/lightgcn_emb:latest"
    wandb_artifact: Optional[str] = None          # набор файлов внутри (user_emb.npy, item_emb.npy, user_map.parquet, item_map.parquet)
    wandb_project: Optional[str] = None           # если артефакт указан без project (не обязательно)
    wandb_alias_or_version: Optional[str] = None  # если нужно переопределить версию

    # ранжирование
    batch_users: int = 2048       # размер батча по пользователям
    pool_batch_items: int = 8192  # батч по айтемам при расчёте dot с пулом
    use_faiss: bool = True        # ускорение только для кейса "без пула"
    oversample_factor: float = 1.3  # запрашивать у FAISS чуть больше кандидатов и выкидывать seen


# ============================== загрузка эмбеддингов ==============================

def _ensure_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)

def load_local(cfg: ImportConfig) -> LGCNEmbeddings:
    """
    Загрузить эмбеддинги и маппинги из локальных файлов.
    """
    u_emb_p = _ensure_path(cfg.user_emb_path or (PATHS.artifact_dir / "lightgcn" / "user_emb.npy"))
    i_emb_p = _ensure_path(cfg.item_emb_path or (PATHS.artifact_dir / "lightgcn" / "item_emb.npy"))
    u_map_p = _ensure_path(cfg.user_map_path or (PATHS.artifact_dir / "lightgcn" / "user_map.parquet"))
    i_map_p = _ensure_path(cfg.item_map_path or (PATHS.artifact_dir / "lightgcn" / "item_map.parquet"))

    if not u_emb_p.exists() or not i_emb_p.exists() or not u_map_p.exists() or not i_map_p.exists():
        raise FileNotFoundError(f"Не найдены файлы эмбеддингов/маппингов: {u_emb_p}, {i_emb_p}, {u_map_p}, {i_map_p}")

    user_emb = np.load(u_emb_p).astype(np.float32, copy=False)
    item_emb = np.load(i_emb_p).astype(np.float32, copy=False)
    user_map = pd.read_parquet(u_map_p)
    item_map = pd.read_parquet(i_map_p)

    # приведение схемы колонок
    if "u_idx" not in user_map.columns:
        # допускаем альтернативное имя
        ucol = [c for c in user_map.columns if c != COL_USER][0]
        user_map = user_map.rename(columns={ucol: "u_idx"})
    if "i_idx" not in item_map.columns:
        icol = [c for c in item_map.columns if c != COL_ITEM][0]
        item_map = item_map.rename(columns={icol: "i_idx"})

    return LGCNEmbeddings(user_emb=user_emb, item_emb=item_emb, user_map=user_map, item_map=item_map)


def load_from_wandb(cfg: ImportConfig) -> LGCNEmbeddings:
    """
    Скачать из W&B artifact папку с файлами эмбеддингов и маппингов.
    Ожидаемые имена внутри артефакта:
        - user_emb.npy
        - item_emb.npy
        - user_map.parquet
        - item_map.parquet
    """
    if not _WANDB:
        raise ImportError("wandb не установлен — не могу скачать артефакт.")

    if cfg.wandb_artifact is None:
        raise ValueError("Укажи cfg.wandb_artifact (например, 'tshopping_stage1/lightgcn_emb:latest').")

    # login мягко
    try:
        wandb.login(anonymous="allow")
    except Exception:
        pass

    at = wandb.use_artifact(cfg.wandb_artifact)
    dirpath = Path(at.download())
    # имена файлов (стандартные)
    user_emb = np.load(dirpath / "user_emb.npy").astype(np.float32, copy=False)
    item_emb = np.load(dirpath / "item_emb.npy").astype(np.float32, copy=False)
    user_map = pd.read_parquet(dirpath / "user_map.parquet")
    item_map = pd.read_parquet(dirpath / "item_map.parquet")

    if "u_idx" not in user_map.columns:
        ucol = [c for c in user_map.columns if c != COL_USER][0]
        user_map = user_map.rename(columns={ucol: "u_idx"})
    if "i_idx" not in item_map.columns:
        icol = [c for c in item_map.columns if c != COL_ITEM][0]
        item_map = item_map.rename(columns={icol: "i_idx"})

    return LGCNEmbeddings(user_emb=user_emb, item_emb=item_emb, user_map=user_map, item_map=item_map)


# ============================== маппинги и пулы ==============================

def users_to_indices(user_ids: Sequence[int], user_map: pd.DataFrame) -> np.ndarray:
    """
    Преобразует массив user_id → u_idx; незнакомые пользователи получают -1.
    """
    df = pd.DataFrame({COL_USER: pd.Series(user_ids, dtype="int64")})
    out = df.merge(user_map, on=COL_USER, how="left")["u_idx"].fillna(-1).astype("int64").to_numpy()
    return out

def items_to_indices(item_ids: Sequence[int], item_map: pd.DataFrame) -> np.ndarray:
    df = pd.DataFrame({COL_ITEM: pd.Series(item_ids, dtype="int64")})
    out = df.merge(item_map, on=COL_ITEM, how="left")["i_idx"].fillna(-1).astype("int64").to_numpy()
    return out

def indices_to_items(i_idx_list: Sequence[int], item_map: pd.DataFrame) -> List[int]:
    # item_map имеет i_idx уникален → делаем быстрый обратный словарь 1 раз
    if "i_idx" not in item_map.columns:
        raise ValueError("item_map must contain 'i_idx'.")
    inv = item_map.set_index("i_idx")[COL_ITEM].to_dict()
    return [int(inv.get(int(i), -1)) for i in i_idx_list]

def build_seen_map(train_df: pd.DataFrame, user_map: pd.DataFrame, item_map: pd.DataFrame) -> Dict[int, set]:
    """
    Собирает карту: u_idx → set(i_idx) по TRAIN (для фильтрации seen).
    Неизвестные user/item отброшены.
    """
    df = (train_df
          .merge(user_map, on=COL_USER, how="inner")
          .merge(item_map, on=COL_ITEM, how="inner"))[["u_idx", "i_idx"]]
    seen = {}
    for u, g in df.groupby("u_idx", sort=False):
        seen[int(u)] = set(int(x) for x in g["i_idx"].values)
    return seen


# ============================== ранжирование ==============================

def _rank_batched_dot(
    user_emb: np.ndarray,              # [U, D]
    item_emb: np.ndarray,              # [I, D]
    u_idx_batch: np.ndarray,           # [B]
    M: int,
    pool_idx: Optional[np.ndarray] = None,    # [P] индексы айтемов; если None — по всем айтемам
    exclude_seen: Optional[Dict[int, set]] = None,
) -> Dict[int, List[int]]:
    """
    Батчевое ранжирование по dot-product. Если задан pool_idx — считаем только по пулу.
    exclude_seen: карта u_idx → set(i_idx) (выбросить просмотренные).
    Возвращает: u_idx → список top-M i_idx.
    """
    D = user_emb.shape[1]
    out: Dict[int, List[int]] = {}

    # матрицы для ускорения
    Y = item_emb if pool_idx is None else item_emb[pool_idx]  # [I or P, D]
    idx_map = (np.arange(Y.shape[0], dtype=np.int64) if pool_idx is None else pool_idx)  # позиция → i_idx

    # векторизованный батч
    # разделяем по разумным чанкам пользователей, чтобы не выбивать память
    BATCH = max(1, min(4096, len(u_idx_batch)))
    for start in range(0, len(u_idx_batch), BATCH):
        end = min(start + BATCH, len(u_idx_batch))
        u_slice = u_idx_batch[start:end]
        valid_mask = (u_slice >= 0)
        if not valid_mask.any():
            continue
        u_valid = u_slice[valid_mask]
        Umat = user_emb[u_valid]  # [Bv, D]

        # Scores = U * Y^T  => shape [Bv, P]
        scores = Umat @ Y.T

        # исключаем seen (ставим -inf)
        if exclude_seen is not None and len(exclude_seen) > 0:
            for r, u in enumerate(u_valid):
                seen_u = exclude_seen.get(int(u))
                if not seen_u:
                    continue
                if pool_idx is None:
                    # seen индексы по полному I
                    # ставим -inf у позиций этих i_idx
                    scores[r, list(seen_u)] = -np.inf
                else:
                    # seen заданы в глобальных i_idx → найдём пересечение с pool_idx
                    # быстрый способ: маска через поиск
                    # построим локальный словарь i_idx→position для пула один раз?
                    # ради простоты — сделаем векторное сравнение (P может быть до 2000-5000 — ок).
                    mask = np.isin(pool_idx, np.fromiter(seen_u, dtype=np.int64))
                    scores[r, mask] = -np.inf

        # top-M
        topK = min(M, scores.shape[1])
        idx_part = np.argpartition(-scores, kth=topK-1, axis=1)[:, :topK]
        part_scores = np.take_along_axis(scores, idx_part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top_local = np.take_along_axis(idx_part, order, axis=1)  # позиции в Y

        # конвертируем в i_idx
        for r, u in enumerate(u_valid):
            loc = top_local[r]
            items_idx = idx_map[loc]
            out[int(u)] = items_idx.astype(np.int64).tolist()

    return out


def _rank_faiss_all_items(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    u_idx_batch: np.ndarray,
    M: int,
    exclude_seen: Optional[Dict[int, set]] = None,
    oversample_factor: float = 1.3,
) -> Dict[int, List[int]]:
    """
    Быстрое ранжирование по всем айтемам через FAISS (IndexFlatIP).
    Для exclude_seen делаем oversample и затем фильтруем.
    """
    out: Dict[int, List[int]] = {}
    if not _FAISS:
        # fallback на батчевый dot
        return _rank_batched_dot(user_emb, item_emb, u_idx_batch, M, pool_idx=None, exclude_seen=exclude_seen)

    d = item_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_emb.astype(np.float32, copy=False))

    K = int(max(M + 50, int(M * oversample_factor)))
    Q = user_emb[u_idx_batch]  # [B, D]
    D, I = index.search(Q, K)  # [B, K]

    for r, u in enumerate(u_idx_batch):
        cand = I[r].tolist()
        if exclude_seen is not None:
            seen = exclude_seen.get(int(u), set())
            if seen:
                cand = [c for c in cand if c not in seen]
        out[int(u)] = cand[:M]
    return out


def rank_users(
    emb: LGCNEmbeddings,
    users: Sequence[int],
    M: int,
    pool_items: Optional[Sequence[int]] = None,   # item_ids (сырые) или i_idx? → см. use_indices_for_pool
    exclude_seen_map: Optional[Dict[int, set]] = None,  # карта u_idx→set(i_idx)
    use_indices_for_pool: bool = False,          # True если pool_items — это уже i_idx; False если item_id
    cfg: Optional[ImportConfig] = None,
) -> Dict[int, List[int]]:
    """
    Ранжирует пользователей `users` (в user_id!) и возвращает карту:
        user_id → [item_id1, item_id2, ...] длиной M.
    Если user не встречался в TRAIN → выдаём пустой список.

    Аргументы:
    - pool_items: ограниченный пул (желательно). По умолчанию ранжируем по всем айтемам (дорого).
      Можно передать как список i_idx (use_indices_for_pool=True) или item_id (False, тогда сделаем маппинг).
    - exclude_seen_map: ожидается на индексах (u_idx → set(i_idx)); собери её через build_seen_map(train_df, ...).
    - cfg: ImportConfig с батч-размерами/FAISS флагом.
    """
    cfg = cfg or ImportConfig()

    # user_id → u_idx
    u_idx = users_to_indices(users, emb.user_map)  # [-1 если неизвестный]
    known_mask = (u_idx >= 0)
    if not known_mask.any():
        return {int(u): [] for u in users}

    # пул айтемов на индексах
    if pool_items is None:
        pool_idx = None
    else:
        if use_indices_for_pool:
            pool_idx = np.asarray(pool_items, dtype=np.int64)
        else:
            pool_idx = items_to_indices(pool_items, emb.item_map)
            pool_idx = pool_idx[pool_idx >= 0]
            if len(pool_idx) == 0:
                pool_idx = None

    out_idx: Dict[int, List[int]] = {}

    # ранжируем батчами
    B = int(cfg.batch_users)
    known_users = np.asarray(users, dtype=np.int64)[known_mask]
    known_u_idx = u_idx[known_mask]

    for start in range(0, len(known_u_idx), B):
        end = min(start + B, len(known_u_idx))
        u_batch = known_u_idx[start:end]

        if pool_idx is None:
            # ранжировать по всем айтемам
            if cfg.use_faiss and _FAISS:
                part = _rank_faiss_all_items(
                    emb.user_emb, emb.item_emb, u_batch, M,
                    exclude_seen=exclude_seen_map, oversample_factor=cfg.oversample_factor
                )
            else:
                part = _rank_batched_dot(
                    emb.user_emb, emb.item_emb, u_batch, M,
                    pool_idx=None, exclude_seen=exclude_seen_map
                )
        else:
            # ранжировать только по пулу
            part = _rank_batched_dot(
                emb.user_emb, emb.item_emb, u_batch, M,
                pool_idx=pool_idx, exclude_seen=exclude_seen_map
            )

        out_idx.update(part)

    # конвертируем в выдачу на user_id с item_id
    idx2id = emb.item_map.set_index("i_idx")[COL_ITEM].to_dict()
    result: Dict[int, List[int]] = {}
    for u_id, u_i in zip(known_users, known_u_idx):
        items_i = out_idx.get(int(u_i), [])
        result[int(u_id)] = [int(idx2id.get(int(k), -1)) for k in items_i]

    # неизвестные пользователи → пустые списки
    for u in users:
        if int(u) not in result:
            result[int(u)] = []

    return result


# ============================== вспомогательные I/O ==============================

def save_candidates_map_item_id(cand_map: Mapping[int, Sequence[int]], path: Union[str, Path]) -> Path:
    """
    Сохраняет карту user_id → [item_id1 item_id2 ...] в parquet:
      columns: [user_id, items]  (items — строка "i1 i2 ...")
    """
    df = pd.DataFrame({
        COL_USER: [int(u) for u in cand_map.keys()],
        "items": [" ".join(map(str, cand_map[u])) for u in cand_map.keys()]
    })
    path = _ensure_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ============================== “сборка” из разных источников ==============================

def load_embeddings(cfg: ImportConfig) -> LGCNEmbeddings:
    """
    Удобная обёртка: если заданы локальные пути — грузим локально, иначе тянем из W&B.
    """
    if cfg.user_emb_path or cfg.item_emb_path or cfg.user_map_path or cfg.item_map_path:
        return load_local(cfg)
    if cfg.wandb_artifact:
        return load_from_wandb(cfg)
    # дефолт: пытаемся локально по стандартным путям
    return load_local(cfg)
