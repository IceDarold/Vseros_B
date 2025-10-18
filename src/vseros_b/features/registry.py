# src/vseros_b/features/registry.py
from __future__ import annotations
import importlib, pkgutil, inspect
from typing import Dict, Tuple, List
from vseros_b.features.base import FeatureSpec, FeatureBuilder

_REG: Dict[str, Tuple[FeatureSpec, FeatureBuilder]] = {}

def register_feature(spec: FeatureSpec):
    def deco(fn: FeatureBuilder):
        if spec.name in _REG:
            raise ValueError(f"Feature '{spec.name}' already registered")
        _REG[spec.name] = (spec, fn)
        return fn
    return deco

def get_registry() -> Dict[str, Tuple[FeatureSpec, FeatureBuilder]]:
    return dict(_REG)

def select(names: List[str]) -> Dict[str, Tuple[FeatureSpec, FeatureBuilder]]:
    reg = get_registry()
    missing = [n for n in names if n not in reg]
    if missing:
        raise KeyError(f"Features not registered: {missing}")
    return {n: reg[n] for n in names}

def autodiscover():
    """
    Импортирует все модули внутри vseros_b.features.*, чтобы сработали декораторы.
    Вызывай один раз перед build_matrix().
    """
    pkg = importlib.import_module("vseros_b.features")
    for m in pkgutil.iter_modules(pkg.__path__):
        # пропускаем внутренние
        if not m.name.startswith("feat_") and m.name not in ("feat_sources",):
            continue
        importlib.import_module(f"vseros_b.features.{m.name}")
