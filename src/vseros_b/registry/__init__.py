"""Experiment registry for dynamic lookup and orchestration."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Type

from ..base_exp import BaseExperiment

_REGISTRY: Dict[str, Type[BaseExperiment]] = {}
_LOADED_DEFAULTS = False


def register(name: str, cls: Type[BaseExperiment]) -> None:
    if not issubclass(cls, BaseExperiment):
        raise TypeError(f"Cannot register {cls!r}: expected subclass of BaseExperiment.")
    _REGISTRY[name] = cls


def registered(name: str) -> bool:
    _ensure_defaults()
    return name in _REGISTRY


def get(name: str) -> Type[BaseExperiment]:
    _ensure_defaults()
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Experiment '{name}' is not registered.") from exc


def create(name: str, *args, **kwargs) -> BaseExperiment:
    cls = get(name)
    return cls(*args, **kwargs)


def available() -> List[str]:
    _ensure_defaults()
    return sorted(_REGISTRY.keys())


def _ensure_defaults() -> None:
    global _LOADED_DEFAULTS
    if not _LOADED_DEFAULTS:
        from . import default  # noqa: F401  # side-effect: registers experiments
        _LOADED_DEFAULTS = True


__all__ = ["register", "registered", "get", "create", "available"]
