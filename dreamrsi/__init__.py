"""OpenDream-RSI: an open reproduction of *Dream-RSI: Recursive Self-Improvement
through Evolving Worlds* (Zheng et al., 2026).

The framework makes exploration explicit and programmable in a thin
orchestration layer while leaving the underlying coding agent unchanged, and
closes a recursive self-improvement loop at the meta-exploration layer:

* :mod:`dreamrsi.online`   -- Stage 1, online exploration into a discovery tree;
* :mod:`dreamrsi.world`    -- Stage 2, that tree served back as a replay simulator;
* :mod:`dreamrsi.dreaming` -- Stage 3, policy improvement by dreaming over the pool;
* :mod:`dreamrsi.rsi`      -- the outer loop that alternates the three.
"""

from __future__ import annotations

__version__ = "0.1.0"
__paper__ = "https://github.com/zhengkid/Dream-RSI"

from .objective import (  # noqa: F401
    attainment,
    attainment_auc,
    parallel_penalty,
    replay_value,
    summarize_sweep,
)
from .tree import DiscoveryTree, Node  # noqa: F401
from .types import (  # noqa: F401
    CellMeta,
    GridPlan,
    GridPlanningContext,
    Observation,
    ParetoSummary,
    ReplayScore,
    SimResult,
)
from .world import ReplayWorld, SimulatorPool  # noqa: F401

__all__ = [
    "DiscoveryTree",
    "Node",
    "ReplayWorld",
    "SimulatorPool",
    "Observation",
    "CellMeta",
    "GridPlan",
    "GridPlanningContext",
    "SimResult",
    "ReplayScore",
    "ParetoSummary",
    "replay_value",
    "attainment",
    "attainment_auc",
    "parallel_penalty",
    "summarize_sweep",
    "__version__",
]
