"""Lightweight logging helpers used across scripts and experiments."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def log_timing(logger: logging.Logger, message: str, *, level: int = logging.INFO) -> Iterator[None]:
    start = time.perf_counter()
    logger.log(level, "%s - start", message)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s - done in %.2fs", message, elapsed)
