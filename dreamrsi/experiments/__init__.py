"""Experiments that measure whether the reproduced mechanism behaves as claimed.

The rest of the package *implements* Dream-RSI; this subpackage *interrogates*
it.  Three questions the paper's argument rests on, none of which the mechanism
tests can answer on their own:

* :mod:`~dreamrsi.experiments.fidelity` -- is a frozen history really an exact
  simulator, and how far does "the region that was explored" actually reach?
* :mod:`~dreamrsi.experiments.generalization` -- dreaming maximises replay score
  on ``H_t``; does the version it selects still win on a world it never saw, or
  is the non-regression guarantee an in-pool artefact?
* :mod:`~dreamrsi.experiments.benchmark` -- does Dream-RSI beat Recursive Fixed
  Exploration across seeds, and does *accumulating* worlds matter?

Everything here runs offline against the synthetic task.
"""

from __future__ import annotations

from .benchmark import (  # noqa: F401
    AblationRow,
    SeedComparison,
    ablate_pool_window,
    compare_across_seeds,
)
from .fidelity import (  # noqa: F401
    FidelityReport,
    SupportReport,
    replay_fidelity,
    support_coverage,
)
from .generalization import (  # noqa: F401
    GeneralizationReport,
    VersionScore,
    generalization_report,
    temporal_split,
)

__all__ = [
    "FidelityReport",
    "SupportReport",
    "replay_fidelity",
    "support_coverage",
    "GeneralizationReport",
    "VersionScore",
    "generalization_report",
    "temporal_split",
    "SeedComparison",
    "AblationRow",
    "compare_across_seeds",
    "ablate_pool_window",
]
