# -*- coding: utf-8 -*-
"""
Глобальный конфиг для Stage 1 (recall).
Держим все пути и дефолтные гиперы тут, чтобы не размазывать по ноутам.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ==== колонки (у нас строго так) ====
COL_USER = "user_id"
COL_ITEM = "item_id"
COL_DATE = "date"

# ==== базовые пути (переопределяются через переменные окружения) ====

def _path_from_env(var: str, default: Path) -> Path:
    value = os.getenv(var)
    if value:
        return Path(value).expanduser()
    return default

PROJECT_ROOT = _path_from_env("VSEROS_B_PROJECT_ROOT", Path(__file__).resolve().parents[2])
DATA_DIR = _path_from_env("VSEROS_B_DATA_DIR", PROJECT_ROOT / "data")
ARTIFACT_DIR = _path_from_env("VSEROS_B_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
SUB_DIR = _path_from_env("VSEROS_B_SUBMISSIONS_DIR", PROJECT_ROOT / "submissions")

INTERM_DIR = _path_from_env("VSEROS_B_INTERMEDIATE_DIR", ARTIFACT_DIR / "intermediates")  # кеши (basket_items, item_support, графы, ...)
CAND_DIR = _path_from_env("VSEROS_B_CANDIDATES_DIR", ARTIFACT_DIR / "candidates")        # пер-юзер кандидаты (parquet)
METRICS_DIR = _path_from_env("VSEROS_B_METRICS_DIR", ARTIFACT_DIR / "metrics")           # метрики (json/csv)

TRAIN_PATH = _path_from_env("VSEROS_B_TRAIN_PATH", DATA_DIR / "train_data.pq")
SAMPLE_PATH = _path_from_env("VSEROS_B_SAMPLE_PATH", DATA_DIR / "sample_submission.csv")

# ==== валидация ====
VAL_DAYS = 7  # последние 7 дней → вал

# ==== гиперпараметры Stage 1 ====
# exp101 — λ-свип для time-decay
LAMBDA_LIST: List[float] = [0.02, 0.05, 0.08, 0.12]
DECAY_REF: Optional[int] = None  # если None → используем конец train (T-7)

# exp102 — окна для daily trending
TRENDING_WINDOWS: List[int] = [3, 5, 7]

# exp103/104 — co-vis
COVIS_MIN_PAIR_COUNT = 2
COVIS_SCORE_TYPE = "cosine"  # 'cosine' | 'jaccard' | 'lift'
TOPN_NEIGHBORS_PER_ITEM = 300
DECAY_LAMBDA_COVIS = 0.05

# кандидаты на пользователя
K_RECENT_ITEMS_PER_USER = 5
CAND_TOP_M_PER_USER     = 1000

# item2vec-lite
I2V_DIM = 128
I2V_NS  = 10
I2V_TOPN = 200

# PPR
PPR_ALPHA = 0.15
PPR_ITERS = 20

# ==== управление запуском ====
SEED = 42
QUICK_MODE = False         # если True — ограничим пользователей/дни в тяжёлых шагах
QUICK_USERS: Optional[int] = 100_000  # лимит юзеров в quick-режиме (None = без лимита)
FORCE_REBUILD = False      # пересчитывать кеши/артефакты, даже если уже есть

# ==== Weights & Biases (W&B) ====
WANDB_PROJECT = "tshopping"
WANDB_GROUP   = "stage1_recall"

@dataclass
class SplitInfo:
    date_min: int
    date_max: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int

@dataclass
class Paths:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    train_path: Path = TRAIN_PATH
    sample_path: Path = SAMPLE_PATH
    artifact_dir: Path = ARTIFACT_DIR
    interm_dir: Path = INTERM_DIR
    cand_dir: Path = CAND_DIR
    metrics_dir: Path = METRICS_DIR
    sub_dir: Path = SUB_DIR

    def ensure(self) -> None:
        for attr in ("artifact_dir", "interm_dir", "cand_dir", "metrics_dir", "sub_dir"):
            getattr(self, attr).mkdir(parents=True, exist_ok=True)

# Удобный единый объект путей (можно импортить и сразу ensure())
PATHS = Paths()
