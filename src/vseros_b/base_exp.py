# -*- coding: utf-8 -*-
"""
base_exp.py — базовый класс для экспериментов Stage 1.

Единый интерфейс:
    - fit(context)                   → посчитать всё из TRAIN и подготовить состояние
    - candidates(context, ...)       → вернуть per-user кандидатов (если это источник)
    - predict_submission(context, sample_df) → сделать сабмит (если у эксперимента это уместно)
    - evaluate(context)              → вернуть таблицу/словарь метрик
    - save(context)                  → сохранить артефакты (метрики/кандидаты/прочее)

Плюс:
    - out_dir / cand_dir / metrics_dir, auto-ensure
    - save_metrics_df(), save_candidates_map() и лёгкие W&B обёртки
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd

from .config import (
    PATHS, COL_USER, COL_ITEM,
    SEED, WANDB_PROJECT, WANDB_GROUP,
)
from .artifacts import ensure_dir, save_df, save_json, log_artifact, log_table_df

# W&B — опционально (модуль работает и без активного wandb.run)
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


@dataclass
class ExpIO:
    """Контейнер путей артефактов для конкретного эксперимента."""
    exp_name: str
    out_dir: Path
    cand_dir: Path
    metrics_dir: Path

    @classmethod
    def make(cls, exp_name: str) -> "ExpIO":
        out_dir = ensure_dir(PATHS.artifact_dir / exp_name)
        cand_dir = ensure_dir(PATHS.cand_dir / exp_name)
        metrics_dir = ensure_dir(PATHS.metrics_dir)
        return cls(exp_name=exp_name, out_dir=out_dir, cand_dir=cand_dir, metrics_dir=metrics_dir)


class BaseExperiment(ABC):
    """
    Базовый класс: реализует общие хелперы, но не знает про конкретную логику.
    Наследники обязаны определить: fit / evaluate; опционально: candidates / predict_submission / save.
    """

    def __init__(self, exp_name: str, seed: int = SEED):
        self.exp_name = exp_name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.io = ExpIO.make(exp_name)
        self.verbose: bool = False
        self._state: Dict[str, Any] = {}  # гибкий карман для состояний наследника

    # ---------- обязательные методы, которые должны реализовать наследники ----------

    @abstractmethod
    def fit(self, context: Dict[str, Any]) -> Any:
        """
        Построить все необходимые артефакты из TRAIN и поместить их в self._state.
        context: словарь с объектами из data.load_and_prepare (train_df, val_df, split, caches...).
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Посчитать и вернуть метрики (таблица/словарь).
        Должно использовать только self._state и ВАЛ (для оценки).
        """
        raise NotImplementedError

    # ---------- опциональные методы ----------

    def candidates(
        self,
        context: Dict[str, Any],
        users: Optional[Sequence[int]] = None,
        M: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        """
        Вернуть per-user кандидатов (если эксперимент является источником кандидатов).
        По умолчанию NotImplemented (выкинем если вызвали).
        """
        raise NotImplementedError(f"{self.exp_name} does not implement candidates()")

    def predict_submission(
        self,
        context: Dict[str, Any],
        sample_df: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Вернуть сабмит (если эксперимент умеет генерировать глобальный список).
        По умолчанию NotImplemented (выкинем если вызвали).
        """
        raise NotImplementedError(f"{self.exp_name} does not implement predict_submission()")

    def save(self, context: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Сохранить ключевые артефакты (метрики/кандидаты/модели). Вернуть пути или (None, None).
        Наследник может переопределить. Базовая версия ничего не делает.
        """
        return None, None

    # ---------- общие хелперы для наследников ----------

    def save_metrics_df(self, df: pd.DataFrame, filename: Optional[str] = None, artifact_name: Optional[str] = None) -> Path:
        """
        Сохранить таблицу метрик в metrics_dir и залогировать как W&B Artifact (если возможно).
        """
        filename = filename or f"{self.exp_name}.csv"
        path = self.io.metrics_dir / filename
        save_df(path, df, index=False)
        if artifact_name:
            try:
                log_artifact(path, name=artifact_name, type_="dataset")
            except Exception:
                pass
        return path

    def save_candidates_map(self, cand_map: Mapping[int, Sequence[int]], filename: str) -> Path:
        """
        Сохранить per-user кандидатов в виде (user_id, items) с items="i1 i2 ...".
        """
        df = self._cand_map_to_df(cand_map)
        path = self.io.cand_dir / filename
        save_df(path, df, index=False)
        try:
            log_artifact(path, name=f"{self.exp_name}_candidates", type_="dataset")
        except Exception:
            pass
        return path

    def wandb_log_table(self, key: str, df: pd.DataFrame) -> None:
        """Мягкое логирование таблицы в W&B (если ран активен)."""
        if _WANDB_AVAILABLE and wandb.run is not None and df is not None and not df.empty:
            try:
                log_table_df(key, df)
            except Exception:
                pass

    def wandb_summary(self, **kwargs: Any) -> None:
        """Обновить summary текущего W&B рана (мягко)."""
        if _WANDB_AVAILABLE and wandb.run is not None:
            for k, v in kwargs.items():
                try:
                    wandb.summary[k] = v
                except Exception:
                    pass

    # ---------- утилиты ----------

    @staticmethod
    def _cand_map_to_df(cand_map: Mapping[int, Sequence[int]]) -> pd.DataFrame:
        """
        Преобразует {user: [i1, i2, ...]} → DataFrame(user_id, items="i1 i2 ...")
        """
        return pd.DataFrame({
            COL_USER: [int(u) for u in cand_map.keys()],
            "items": [" ".join(map(str, cand_map[u])) for u in cand_map.keys()]
        })

    @staticmethod
    def ensure_run_dirs(exp_name: str) -> ExpIO:
        """Убедиться, что каталоги существуют, и вернуть ExpIO."""
        return ExpIO.make(exp_name)

    # ---------- доступ к состоянию (необязательно, но удобно) ----------

    def get_state(self, key: Optional[str] = None, default: Any = None) -> Any:
        """Вернуть self._state или по ключу."""
        if key is None:
            return self._state
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Положить значение в self._state."""
        self._state[key] = value

    # ---------- подсказка по контексту ----------

    @staticmethod
    def require_context_keys(context: Dict[str, Any], required: Sequence[str]) -> None:
        """
        Санити-чек: проверить, что в context есть все ожидаемые ключи.
        Поднимет ValueError с явным списком недостающих.
        """
        missing = [k for k in required if k not in context]
        if missing:
            raise ValueError(f"Context is missing keys: {missing}")
