"""Core data types for Dream-RSI.

The paper models a discovery rollout as a rooted tree whose only atomic
operation is ``CONTINUE(v)``: resume the workspace stored at node ``v`` and
generate exactly one new child.  Continuing from the root opens a new *branch*;
continuing from a non-root leaf *refines* the end of an existing branch.

Because every non-root node has exactly one primary parent, the tree is a set of
chains and is equivalent to an irregular ``branch x attempt`` grid -- which is
precisely the environment the replay-policy prompt (Listing 2 of the paper)
describes.  We therefore address nodes by a stable cell id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT_ID = "root"

#: Failure taxonomy used by the policy prompt.  ``ok`` means the attempt was
#: evaluated end-to-end without an execution error -- it does *not* mean the
#: artifact was valid (see ``Observation.is_success``).
FAIL_CLASSES = (
    "ok",            # evaluated, no execution error
    "compile_other", # build/compile failure -- usually repairable
    "runtime",       # crashed while running -- usually repairable
    "correctness",   # produced output that failed the correctness check
    "resource",      # OOM / shared-memory / worker limits -- repairable
    "timeout",       # exceeded the wall-clock limit
    "env",           # environment or dependency failure (not the candidate's fault)
)

#: Failure classes that the prompt calls "normally repairable".
REPAIRABLE_FAIL_CLASSES = frozenset({"compile_other", "runtime", "correctness", "resource"})


def cell_id(branch: int, attempt: int) -> str:
    """Canonical id for the cell at ``(branch, attempt)``."""
    return f"b{branch}a{attempt}"


def parse_cell_id(cid: str) -> Tuple[int, int]:
    """Inverse of :func:`cell_id`.  Raises ``ValueError`` on a malformed id."""
    if not cid.startswith("b") or "a" not in cid:
        raise ValueError(f"malformed cell id: {cid!r}")
    b, _, a = cid[1:].partition("a")
    return int(b), int(a)


@dataclass(frozen=True)
class CellMeta:
    """Structural facts about a cell.

    Available *before* a cell is revealed -- the policy is allowed to use this
    to choose a batch, because it leaks no outcome.
    """

    cell_id: str
    branch: int
    attempt: int
    parent_id: str
    seq: int = -1
    tags: Tuple[str, ...] = ()

    @property
    def is_root(self) -> bool:
        return self.attempt == 0


@dataclass
class Observation:
    """What the policy learns when a cell is revealed.

    ``score`` follows the task's fixed scoring protocol: larger is better.
    """

    cell_id: str
    branch: int
    attempt: int
    score: float = float("-inf")
    evaluated: bool = False
    valid: bool = False
    fail_class: str = "ok"
    error: Optional[str] = None
    delta_vs_baseline: float = 0.0
    delta_vs_parent: float = 0.0
    n_valid: Optional[int] = None
    n_total: Optional[int] = None
    seq: int = -1
    tags: Tuple[str, ...] = ()
    cost: float = 1.0

    # -- success semantics, verbatim from the policy prompt -------------------
    @property
    def is_success(self) -> bool:
        """``error is None`` and ``fail_class == "ok"``.

        Note the prompt's emphasis: this is a *successful evaluation* even when
        ``valid is False`` or ``n_valid``/``n_total`` are unavailable.  Never
        treat ``valid is False`` on its own as a repairable failure signal.
        """
        return self.evaluated and self.error is None and self.fail_class == "ok"

    @property
    def is_repairable_failure(self) -> bool:
        return (not self.is_success) and self.fail_class in REPAIRABLE_FAIL_CLASSES

    @property
    def is_environment_failure(self) -> bool:
        return self.fail_class == "env"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        if d["score"] == float("-inf"):
            d["score"] = None
        return d


@dataclass(frozen=True)
class GridPlan:
    """Result of ``plan_grid``: how wide and how deep the *next* live grid is.

    ``branch_count`` (``W``) creates branches ``0..W-1``; ``refine_count``
    (``R``) is the number of refinements allowed after each root, so a branch
    has attempts ``0..R``.  Arbitrary integers are accepted -- there is no fixed
    set of presets.
    """

    branch_count: int
    refine_count: int
    reason: str = ""

    def __post_init__(self) -> None:
        if self.branch_count < 1:
            raise ValueError("branch_count must be >= 1")
        if self.refine_count < 0:
            raise ValueError("refine_count must be >= 0")

    @property
    def max_cells(self) -> int:
        return self.branch_count * (self.refine_count + 1)

    def clamped(self, ctx: "GridPlanningContext") -> "GridPlan":
        return GridPlan(
            branch_count=max(1, min(self.branch_count, ctx.hard_max_branch_count)),
            refine_count=max(0, min(self.refine_count, ctx.hard_max_refine_count)),
            reason=self.reason,
        )


@dataclass(frozen=True)
class GridPlanningContext:
    """Prefix-safe facts handed to ``plan_grid``.

    It deliberately carries *no* outcome from the current episode: only
    completed earlier live manifests, the caps, and (in replay) the structural
    support of the frozen trace.
    """

    history: Tuple[Dict[str, Any], ...] = ()
    fallback_branch_count: int = 4
    fallback_refine_count: int = 4
    hard_max_branch_count: int = 32
    hard_max_refine_count: int = 32
    worker_cap: int = 4
    trace_branch_count: Optional[int] = None
    trace_refine_count: Optional[int] = None
    mode: str = "live"  # "live" | "replay"

    def bootstrap_plan(self, reason: str = "insufficient history; conservative bootstrap") -> GridPlan:
        return GridPlan(
            branch_count=max(1, min(self.fallback_branch_count, self.hard_max_branch_count)),
            refine_count=max(0, min(self.fallback_refine_count, self.hard_max_refine_count)),
            reason=reason,
        )

    def supports(self, plan: GridPlan) -> bool:
        """Whether a replay of ``plan`` stays inside the frozen trace's support."""
        if self.trace_branch_count is None or self.trace_refine_count is None:
            return True
        return (plan.branch_count <= self.trace_branch_count
                and plan.refine_count <= self.trace_refine_count)


@dataclass
class SimResult:
    """Per-episode bookkeeping accumulated by ``_record_curve``."""

    #: ``(total_probes, best_score_so_far)`` sampled at every reveal.
    curve: List[Tuple[int, float]] = field(default_factory=list)
    total_probes: int = 0
    decision_rounds: int = 0
    effective_sequential_rounds: int = 0
    best_score: float = float("-inf")
    best_cell: Optional[str] = None
    revealed: List[str] = field(default_factory=list)
    stopped_reason: str = ""
    batches: List[List[str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curve": [[p, (None if s == float("-inf") else s)] for p, s in self.curve],
            "total_probes": self.total_probes,
            "decision_rounds": self.decision_rounds,
            "effective_sequential_rounds": self.effective_sequential_rounds,
            "best_score": None if self.best_score == float("-inf") else self.best_score,
            "best_cell": self.best_cell,
            "revealed": list(self.revealed),
            "stopped_reason": self.stopped_reason,
            "batches": [list(b) for b in self.batches],
        }


@dataclass
class ReplayScore:
    """Equation (1) of the paper, decomposed into its three terms."""

    discovery_quality: float
    execution_cost: float
    parallelism_bonus: float
    value: float
    revealed_nodes: int
    decision_rounds: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepPoint:
    """One ``(frozen trace, beta)`` replay episode."""

    world_id: str
    beta: float
    auc: float
    attainment: float
    total_probes: int
    decision_rounds: int
    effective_sequential_rounds: int
    parallel_penalty: float
    best_score: Optional[float]
    replay_value: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParetoSummary:
    """Aggregate of a beta sweep -- the quantity the dreaming loop maximises."""

    auc: float
    parallel_penalty: float
    reward: float
    attainment: float
    mean_probes: float
    replay_value: float
    per_beta: Dict[str, Dict[str, float]] = field(default_factory=dict)
    non_degenerate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def safe_max(values: Sequence[float]) -> float:
    best = float("-inf")
    for v in values:
        if v is not None and not math.isnan(v) and v > best:
            best = v
    return best
