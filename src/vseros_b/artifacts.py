# -*- coding: utf-8 -*-
"""
Утилиты для сохранения/загрузки артефактов и мягкой интеграции с W&B.
Никакой привязки к Google Drive: пути задаются извне (или берутся из config.PATHS).
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union, Dict, Any
import json
import pandas as pd

# W&B — опционален: если недоступен или не залогинен, просто пропускаем логирование
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    wandb = None
    _WANDB_AVAILABLE = False

from .config import PATHS


# ----------------------------- файловые операции -----------------------------

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_df(path: Union[str, Path], df: pd.DataFrame, index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    suf = path.suffix.lower()
    if suf in [".parquet", ".pq", ".pqt"]:
        df.to_parquet(path, index=index)
    elif suf in [".csv", ".txt"]:
        df.to_csv(path, index=index)
    else:
        # по умолчанию csv
        df.to_csv(path.with_suffix(".csv"), index=index)
        path = path.with_suffix(".csv")
    return path


def load_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    suf = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(path)
    if suf in [".parquet", ".pq", ".pqt"]:
        return pd.read_parquet(path)
    if suf in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Неизвестное расширение файла: {suf}")


def save_json(path: Union[str, Path], obj: Dict[str, Any], indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def already_exists(path: Union[str, Path]) -> bool:
    return Path(path).exists()


# ----------------------------- W&B артефакты -----------------------------

def _wandb_is_ready() -> bool:
    if not _WANDB_AVAILABLE:
        return False
    try:
        # если run не инициализирован — логировать нельзя
        _ = wandb.run is not None
        return wandb.run is not None
    except Exception:  # pragma: no cover
        return False


def log_artifact(paths: Union[str, Path, Sequence[Union[str, Path]]],
                 name: str,
                 type_: str = "dataset",
                 metadata: Optional[Dict[str, Any]] = None,
                 aliases: Optional[List[str]] = None) -> Optional["wandb.Artifact"]:
    """
    Логирует один или набор файлов как W&B Artifact.
    Возвращает объект Artifact или None, если W&B недоступен.
    """
    if not _wandb_is_ready():
        return None

    if isinstance(paths, (str, Path)):
        file_list = [Path(paths)]
    else:
        file_list = [Path(p) for p in paths]

    for p in file_list:
        if not p.exists():
            raise FileNotFoundError(f"Artifact file not found: {p}")

    art = wandb.Artifact(name=name, type=type_, metadata=metadata or {})
    for p in file_list:
        art.add_file(str(p))
    wandb.log_artifact(art, aliases=aliases or [])
    return art


def log_table_df(key: str, df: pd.DataFrame) -> None:
    """
    Логирует pandas DataFrame как W&B Table (если доступен).
    """
    if not _wandb_is_ready():
        return
    try:
        wandb.log({key: wandb.Table(dataframe=df)})
    except Exception:
        # мягко игнорируем любые проблемы логирования
        pass


# ----------------------------- удобные "сценарии" -----------------------------

def save_and_log_df(df: pd.DataFrame,
                    local_path: Union[str, Path],
                    artifact_name: Optional[str] = None,
                    artifact_type: str = "dataset",
                    table_key: Optional[str] = None) -> Path:
    """
    Сохраняет df на диск и, если возможно, логирует как Artifact + Table.
    """
    path = save_df(local_path, df)
    if artifact_name:
        try:
            log_artifact(path, name=artifact_name, type_=artifact_type)
        except Exception:
            pass
    if table_key:
        log_table_df(table_key, df)
    return path


# ----------------------------- базовые каталоги проекта -----------------------------

def ensure_project_dirs() -> None:
    """
    Гарантирует наличие стандартных папок из config.PATHS.
    Полезно вызывать в начале оркестратора (и это не зависит от Google Drive).
    """
    PATHS.ensure()
