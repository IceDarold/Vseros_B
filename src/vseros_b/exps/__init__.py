"""
Helper module that keeps backward compatibility for experiment files that use
legacy relative imports such as ``from .base_exp import BaseExperiment``.

Historically these modules lived next to the shared helpers (``base_exp.py``,
``config.py`` …) and the imports still assume that layout.  The helpers were
later moved to the package root (``vseros_b``) which broke the relative
imports.  We keep the old paths alive by exposing the real modules under the
``vseros_b.exps`` namespace.
"""

from __future__ import annotations

import sys

from vseros_b import artifacts as _artifacts
from vseros_b import base_exp as _base_exp
from vseros_b import config as _config
from vseros_b import covis as _covis
from vseros_b import item2vec_lite as _item2vec_lite
from vseros_b import metrics as _metrics
from vseros_b import pop_decay as _pop_decay
from vseros_b import trending as _trending
from vseros_b import ppr as _ppr

_THIS_PKG = __name__

# Map the legacy relative import targets to their actual modules.
sys.modules[_THIS_PKG + ".base_exp"] = _base_exp
sys.modules[_THIS_PKG + ".config"] = _config
sys.modules[_THIS_PKG + ".artifacts"] = _artifacts
sys.modules[_THIS_PKG + ".metrics"] = _metrics
sys.modules[_THIS_PKG + ".covis"] = _covis
sys.modules[_THIS_PKG + ".item2vec_lite"] = _item2vec_lite
sys.modules[_THIS_PKG + ".pop_decay"] = _pop_decay
sys.modules[_THIS_PKG + ".trending"] = _trending
sys.modules[_THIS_PKG + ".ppr"] = _ppr

__all__ = [
    "artifacts",
    "base_exp",
    "config",
    "covis",
    "item2vec_lite",
    "metrics",
    "pop_decay",
    "trending",
    "ppr",
]

