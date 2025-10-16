# -*- coding: utf-8 -*-
"""
Глобальный конфиг для Stage 1 (recall).
Держим все пути и дефолтные гиперы тут, чтобы не размазывать по ноутам.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ==== колонки (у нас строго так) ====
COL_USER = "user_id"
COL_ITEM = "item_id"
COL_DATE = "date"

# ==== базовые пути (под твой Google Drive) ====
# Если хочешь переопределять через переменные окружения — можно так:
# os.getenv("TSHOP_DATA_PATH", "/content/.../train_data.pq")
TRAIN_PATH = Path("/content/drive/MyDrive/ML/Всерос по ИИ 2025/Основной этап/B/train_data.pq")
SAMPLE_PATH = Path("/content/drive/MyDrive/ML/Всерос по ИИ 2025/Основной этап/B/sample_submission.csv")

# Куда складываем артефакты и сабмиты
ARTIFACT_DIR = Path("/content/drive/MyDrive/ML/Всерос по ИИ 2025/Основной этап/B/artifacts")
SUB_DIR      = Path("/content/drive/MyDrive/ML/Всерос по ИИ 2025/Основной этап/B/submissions")

# Подпапки внутри артефактов
INTERM_DIR   = ARTIFACT_DIR / "intermediates"   # кеши (basket_items, item_support, графы, ...)
CAND_DIR     = ARTIFACT_DIR / "candidates"      # пер-юзер кандидаты (parquet)
METRICS_DIR  = ARTIFACT_DIR / "metrics"         # метрики (json/csv)

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
    train_path: Path = TRAIN_PATH
    sample_path: Path = SAMPLE_PATH
    artifact_dir: Path = ARTIFACT_DIR
    interm_dir: Path = INTERM_DIR
    cand_dir: Path = CAND_DIR
    metrics_dir: Path = METRICS_DIR
    sub_dir: Path = SUB_DIR

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.interm_dir.mkdir(parents=True, exist_ok=True)
        self.cand_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.sub_dir.mkdir(parents=True, exist_ok=True)

# Удобный единый объект путей (можно импортить и сразу ensure())
PATHS = Paths()
