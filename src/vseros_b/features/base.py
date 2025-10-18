# src/vseros_b/features/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import pandas as pd

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    cols: Sequence[str]
    version: str = "v1"
    doc: str = ""

FeatureBuilder = Callable[[pd.DataFrame, dict, "FeatureResources"], pd.DataFrame]
