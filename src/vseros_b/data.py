# -*- coding: utf-8 -*-
"""
Загрузка данных, дедуп, time-split, построение кэшей-инвариантов
для экспериментов Stage 1 (работаем только с train).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# scipy.sparse нужен лишь для PPR/LightGCN графа; если не установлен — просто пропустим сборку COO
try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover
    sp = None

from .config import (
    COL_USER, COL_ITEM, COL_DATE,
    VAL_DAYS, DECAY_REF, LAMBDA_LIST,
    SplitInfo, PATHS,
)

# ------------- базовые загрузчики -------------

def _infer_format(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in [".parquet", ".pq", ".pqt"]:
        return "parquet"
    if suf in [".csv", ".txt"]:
        return "csv"
    raise ValueError(f"Неизвестное расширение файла: {path.suffix}")

def load_interactions(path: Path, file_format: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает датасет и возвращает DF строго с колонками [user_id, item_id, date].
    Типы приводим к int64 (date — int32, если влезает).
    """
    fmt = (file_format or _infer_format(path)).lower()
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    need = {COL_USER, COL_ITEM, COL_DATE}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Нет колонок: {missing}. Есть: {list(df.columns)}")

    df = df[[COL_USER, COL_ITEM, COL_DATE]].copy()
    # аккуратно приводим типы
    for c in [COL_USER, COL_ITEM, COL_DATE]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # date компактнее
    if df[COL_DATE].max() <= np.iinfo(np.int32).max:
        df[COL_DATE] = df[COL_DATE].astype("int32")
    else:
        df[COL_DATE] = df[COL_DATE].astype("int64")

    df[COL_USER] = df[COL_USER].astype("int64")
    df[COL_ITEM] = df[COL_ITEM].astype("int64")
    return df

def dedup_user_date_item(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дедуп по ключу (user_id, date, item_id) — как в EDA.
    """
    return df.drop_duplicates([COL_USER, COL_DATE, COL_ITEM]).reset_index(drop=True)

# ------------- time split -------------

def make_time_split(df: pd.DataFrame, val_days: int = VAL_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Делит на train/val по последним val_days (валидируемся на диапазоне [T-val_days+1 .. T]).
    Возвращает (train_df, val_df, SplitInfo).
    """
    date_min = int(df[COL_DATE].min())
    date_max = int(df[COL_DATE].max())
    val_start = date_max - (val_days - 1)
    val_end   = date_max
    train_start = date_min
    train_end   = val_start - 1

    if train_end < train_start:
        raise ValueError("Слишком короткий временной диапазон для заданного VAL_DAYS.")

    train_df = df[(df[COL_DATE] >= train_start) & (df[COL_DATE] <= train_end)].copy()
    val_df   = df[(df[COL_DATE] >= val_start)   & (df[COL_DATE] <= val_end)].copy()

    split = SplitInfo(
        date_min=date_min, date_max=date_max,
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
    )
    return train_df, val_df, split

# ------------- кэши-инварианты -------------

def build_basket_items(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Корзины для co-vis: уникальные (user_id, date, item_id) из TRAIN.
    """
    basket = train_df[[COL_USER, COL_DATE, COL_ITEM]].drop_duplicates()
    return basket.reset_index(drop=True)

def compute_item_support(basket_items: pd.DataFrame) -> pd.DataFrame:
    """
    Поддержка айтема = в скольких корзинах (user,date) он появился.
    """
    support = (
        basket_items.groupby(COL_ITEM)[COL_DATE]
        .count()
        .rename("support")
        .astype("int64")
        .to_frame()
        .reset_index()
    )
    return support

def build_val_truth(val_df: pd.DataFrame) -> Dict[int, set]:
    """
    Истина на вал-окне: user -> множество релевантных item_id.
    """
    return (
        val_df.groupby(COL_USER)[COL_ITEM]
        .apply(lambda s: set(map(int, s.values)))
        .to_dict()
    )

def build_val_item_counts(val_df: pd.DataFrame) -> pd.Series:
    """
    Сколько вал-интеракций приходится на каждый item.
    """
    return (
        val_df.groupby(COL_ITEM)[COL_DATE]
        .count()
        .rename("val_cnt")
        .astype("int64")
    )

def build_bipartite_graph(train_df: pd.DataFrame):
    """
    Бипартитный граф user–item (TRAIN).
    Возвращает tuple: (A_ui: coo_matrix, user_index: dict, item_index: dict)
    Если SciPy недоступен — вернёт (None, user_index, item_index).
    """
    # сжимаем id до 0..U-1 и 0..I-1
    u_ids = pd.Index(train_df[COL_USER].unique(), dtype="int64")
    i_ids = pd.Index(train_df[COL_ITEM].unique(), dtype="int64")
    u_map = {int(u): idx for idx, u in enumerate(u_ids)}
    i_map = {int(i): idx for idx, i in enumerate(i_ids)}

    rows = train_df[COL_USER].map(u_map).to_numpy()
    cols = train_df[COL_ITEM].map(i_map).to_numpy()

    # веса рёбер = 1 (можно позже внести time-decay)
    data = np.ones_like(rows, dtype=np.float32)

    if sp is None:
        # SciPy нет — вернём заглушку
        return None, u_map, i_map

    A = sp.coo_matrix((data, (rows, cols)), shape=(len(u_map), len(i_map)))
    return A, u_map, i_map

# ------------- пайплайн «загрузить и подготовить» -------------

def load_and_prepare(
    data_path: Path = PATHS.train_path,
    file_format: Optional[str] = None,
    ensure_dirs: bool = True,
):
    """
    Утилита «сделай всё и верни пакетом»:
      - загрузить → дедуп
      - time-split
      - кэши: basket_items, item_support, val_truth, val_item_cnt, bipartite graph
    Возвращает dict с ключами:
      df, train_df, val_df, split, basket_items, item_support, val_truth, val_item_cnt, graph (и мапы)
    """
    if ensure_dirs:
        PATHS.ensure()

    df = load_interactions(Path(data_path), file_format=file_format)
    df = dedup_user_date_item(df)

    train_df, val_df, split = make_time_split(df, val_days=VAL_DAYS)

    # кэши
    basket_items = build_basket_items(train_df)
    item_support = compute_item_support(basket_items)
    val_truth    = build_val_truth(val_df)
    val_item_cnt = build_val_item_counts(val_df)
    graph, user_index, item_index = build_bipartite_graph(train_df)

    # небольшой принт-резюме
    print({
        "rows_total": len(df),
        "rows_train": len(train_df),
        "rows_val":   len(val_df),
        "users": int(df[COL_USER].nunique()),
        "items": int(df[COL_ITEM].nunique()),
        "date_min": split.date_min,
        "date_max": split.date_max,
        "val_range": [split.val_start, split.val_end],
    })

    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "split": split,
        "basket_items": basket_items,
        "item_support": item_support,
        "val_truth": val_truth,
        "val_item_cnt": val_item_cnt,
        "graph": graph,
        "user_index": user_index,
        "item_index": item_index,
    }
