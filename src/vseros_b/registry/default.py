"""Default experiment registrations shipped with the toolkit."""

from __future__ import annotations

from typing import Dict, Type

from ..base_exp import BaseExperiment
from ..exps.exp101_pop_decay import Exp101PopDecay
from ..exps.exp102_daily_trending import Exp102DailyTrending
from ..exps.exp103_covis_v1 import Exp103CoVisV1
from ..exps.exp104_covis_2hop import Exp104CoVis2Hop
from ..exps.exp105_item2vec import Exp105Item2Vec
from ..exps.exp106_lightgcn_import import Exp106LightGCNImport
from ..exps.exp107_ppr import Exp107PPR
from ..exps.exp108_recall_fusion import Exp108RecallFusion
from . import register

DEFAULT_EXPERIMENTS: Dict[str, Type[BaseExperiment]] = {
    "exp101_pop_decay": Exp101PopDecay,
    "exp102_daily_trending": Exp102DailyTrending,
    "exp103_covis_v1": Exp103CoVisV1,
    "exp104_covis_2hop": Exp104CoVis2Hop,
    "exp105_item2vec": Exp105Item2Vec,
    "exp106_lightgcn_import": Exp106LightGCNImport,
    "exp107_ppr": Exp107PPR,
    "exp108_recall_fusion": Exp108RecallFusion,
}

for name, cls in DEFAULT_EXPERIMENTS.items():
    register(name, cls)

__all__ = ["DEFAULT_EXPERIMENTS"]
