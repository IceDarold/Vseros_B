"""Vseros_B experimentation toolkit."""

from __future__ import annotations

from pathlib import Path

from .base_exp import BaseExperiment
from . import registry

_VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"
try:
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()
except FileNotFoundError:  # pragma: no cover - package may be used without VERSION file
    __version__ = "0.0.0"

__all__ = ["__version__", "BaseExperiment", "registry"]
