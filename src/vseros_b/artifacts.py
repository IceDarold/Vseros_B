# -*- coding: utf-8 -*-
"""
artifacts.py — утилиты для сохранения/загрузки артефактов и мягкой интеграции с W&B.
- Без привязки к Google Drive: пути задаются извне (или берутся из config.PATHS).
- W&B — опционален: если недоступен/не залогинен, функции просто молча пропускают логирование.

Важно: добавлена санитация имён артефактов под требования W&B:
  допускаются только [A-Za-z0-9._-]
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union, Dict, Any, Mapping

import json
import pandas as pd

# W&B — опционален
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:  # pragma: no cover
    wandb = None  # type: ignore
    _WANDB_AVAILABLE = False

from .config import PATHS


# =============================================================================
# Логгер
# =============================================================================

LOGGER = logging.getLogger("vseros_b.artifacts")


# =============================================================================
# Файловые операции
# =============================================================================

def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_df(path: Union[str, Path], df: pd.DataFrame, index: bool = False) -> Path:
    """
    Сохраняет DataFrame по расширению:
      .parquet/.pq/.pqt → Parquet
      .csv/.txt         → CSV
      иное              → принудительно .csv
    """
    path = Path(path)
    ensure_dir(path.parent)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq", ".pqt"):
        df.to_parquet(path, index=index)
    elif suf in (".csv", ".txt"):
        df.to_csv(path, index=index)
    else:
        # по умолчанию csv
        new_path = path.with_suffix(".csv")
        df.to_csv(new_path, index=index)
        path = new_path
    return path


def load_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq", ".pqt"):
        return pd.read_parquet(path)
    if suf in (".csv", ".txt"):
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
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def already_exists(path: Union[str, Path]) -> bool:
    return Path(path).exists()


# =============================================================================
# W&B helpers
# =============================================================================

def _wandb_offline() -> bool:
    return os.getenv("WANDB_MODE", "").lower() == "offline"


def _wandb_is_ready() -> bool:
    if not _WANDB_AVAILABLE:
        return False
    if _wandb_offline():
        return False
    try:
        return wandb.run is not None  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return False


# --- Санитация имён артефактов/алиасов под правила W&B ---
# Разрешены только: латиница, цифры, точка, дефис, подчёркивание
_NAME_BAD = re.compile(r"[^A-Za-z0-9._-]+")

def _sanitize_artifact_name(name: str) -> str:
    s = _NAME_BAD.sub("-", str(name))
    s = s.strip("._-")
    return s or "artifact"

def _sanitize_alias(a: str) -> str:
    s = _NAME_BAD.sub("-", str(a))
    s = s.strip("._-")
    return s or "latest"


def log_artifact(paths: Union[str, Path, Sequence[Union[str, Path]]],
                 name: str,
                 type_: str = "dataset",
                 metadata: Optional[Dict[str, Any]] = None,
                 aliases: Optional[List[str]] = None) -> Optional["wandb.Artifact"]:
    """
    Логирует один или набор файлов как W&B Artifact (если W&B доступен).
    Возвращает объект Artifact или None, если W&B недоступен.

    Параметры:
      - paths: путь или список путей к файлам
      - name:  ЧЕЛОВЕЧЕСКОЕ имя артефакта — будет очищено под правила W&B
      - type_: тип артефакта ('dataset', 'model', 'submission', ...)
      - metadata: произвольный словарь метаданных
      - aliases: список алиасов (тоже будут очищены)
    """
    if not _wandb_is_ready():
        return None

    # нормализуем список файлов
    if isinstance(paths, (str, Path)):
        file_list = [Path(paths)]
    else:
        file_list = [Path(p) for p in paths]

    for p in file_list:
        if not p.exists():
            raise FileNotFoundError(f"Artifact file not found: {p}")

    safe_name = _sanitize_artifact_name(name)
    safe_aliases = [ _sanitize_alias(a) for a in (aliases or []) ]

    try:
        art = wandb.Artifact(name=safe_name, type=type_, metadata=metadata or {})  # type: ignore[attr-defined]
        for p in file_list:
            art.add_file(str(p))
        wandb.log_artifact(art, aliases=safe_aliases)  # type: ignore[attr-defined]
        LOGGER.info("Logged artifact '%s' with files: %s", safe_name, [str(p) for p in file_list])
        return art
    except Exception:
        # мягко игнорируем любые проблемы логирования
        LOGGER.warning("Failed to log W&B artifact '%s'", safe_name, exc_info=True)
        return None


def log_table_df(key: str, df: pd.DataFrame) -> None:
    """
    Логирует pandas DataFrame как W&B Table (если доступен).
    """
    if not _wandb_is_ready():
        return
    try:
        wandb.log({key: wandb.Table(dataframe=df)})  # type: ignore[attr-defined]
        LOGGER.debug("Logged W&B table '%s' (%d rows)", key, len(df))
    except Exception:
        # мягко игнорируем любые проблемы логирования
        LOGGER.warning("Failed to log W&B table '%s'", key, exc_info=True)
        pass


def save_and_log_df(df: pd.DataFrame,
                    local_path: Union[str, Path],
                    artifact_name: Optional[str] = None,
                    artifact_type: str = "dataset",
                    table_key: Optional[str] = None,
                    aliases: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Path:
    """
    Сохраняет df на диск и, если возможно, логирует как Artifact + Table.

    - artifact_name будет очищен под правила W&B.
    - table_key — ключ для логирования превью-таблицы.
    """
    path = save_df(local_path, df)
    if artifact_name:
        try:
            log_artifact(path, name=artifact_name, type_=artifact_type,
                         metadata=metadata, aliases=aliases)
        except Exception:
            pass
    if table_key:
        log_table_df(table_key, df)

    return path


def log_metrics(metrics: Mapping[str, Any], prefix: str = "", step: Optional[int] = None) -> None:
    """
    Логирует словарь метрик в W&B (если ран активен).
    """
    if not _wandb_is_ready():
        return
    try:
        payload = {f"{prefix}{k}": v for k, v in metrics.items()}
        wandb.log(payload, step=step)  # type: ignore[attr-defined]
        LOGGER.debug("Logged metrics to W&B: %s", payload)
    except Exception:
        LOGGER.warning("Failed to log metrics %s", metrics, exc_info=True)


def download_artifact(artifact_ref: str, target_dir: Union[str, Path]) -> Optional[Path]:
    """
    Скачивает W&B Artifact по полному ref (entity/project/name:alias) в target_dir.
    Возвращает путь к локальной папке или None, если W&B недоступен.
    """
    if not _WANDB_AVAILABLE or _wandb_offline():
        LOGGER.debug("Skipping artifact download for '%s' (W&B unavailable/offline)", artifact_ref)
        return None
    try:
        api = wandb.Api()  # type: ignore[attr-defined]
        art = api.artifact(artifact_ref)  # type: ignore[attr-defined]
        target_dir = ensure_dir(target_dir)
        local_path = Path(art.download(root=str(target_dir)))  # type: ignore[attr-defined]
        LOGGER.info("Downloaded artifact '%s' → %s", artifact_ref, local_path)
        return local_path
    except Exception:
        LOGGER.warning("Failed to download artifact '%s'", artifact_ref, exc_info=True)
        return None


# =============================================================================
# Базовые каталоги проекта
# =============================================================================

def ensure_project_dirs() -> None:
    """
    Гарантирует наличие стандартных папок из config.PATHS.
    Полезно вызывать в начале оркестратора/ноутбука.
    """
    PATHS.ensure()
