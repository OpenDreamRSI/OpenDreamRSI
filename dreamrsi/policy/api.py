"""The stable API an exploration policy is written against.

This is the surface named in the paper's replay-improvement prompt
(Appendix B.2)::

    from see.policy.api import (
        LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result,
    )
    from see.policy.api import GridPlan, GridPlanningContext

Policies are plain Python modules.  A module must define ``NAME`` and a class
``OptimalPolicy(LLMDesignedMethod)`` implementing ``solve`` and ``plan_grid``.
Nothing else in the module is executed by the runner.

Design rules the runner enforces (see :mod:`dreamrsi.policy.loader`):

* **Prefix-only.**  Decisions may use revealed observations, ``baseline_score``,
  the legal sets, structural ``meta`` and helper signals.  Unrevealed scores,
  hardcoded winning cell ids and absolute score targets are off-limits.
* **Terminating.**  Replay calls ``solve`` with ``budget=None``; a policy must
  stop when it selects no batch rather than assuming a budget cap exists.
* **Legal batches.**  Distinct ids, all legal *before* the call, at most
  ``question.max_parallelism`` cells, never a parent together with its child.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..types import (  # re-exported: policies import these from here
    CellMeta,
    GridPlan,
    GridPlanningContext,
    Observation,
    SimResult,
)
from . import observation_signal
from .observation_signal import (
    branch_failed_hard,
    branch_promising,
    probe_improved_vs_baseline,
    probe_improved_vs_parent,
)

__all__ = [
    "LLMDesignedMethod",
    "SimResult",
    "Observation",
    "CellMeta",
    "GridPlan",
    "GridPlanningContext",
    "observation_signal",
    "branch_promising",
    "branch_failed_hard",
    "probe_improved_vs_parent",
    "probe_improved_vs_baseline",
    "_budget_done",
    "_record_curve",
    "finalize_result",
]

NAME = "LLMDesignedMethod"


class LLMDesignedMethod:
    """Base class for an executable exploration policy.

    Subclasses read exactly one scalar knob in ``__init__``::

        beta = float(self.config.get("beta", 0.6))

    ``beta`` is *fixed* within one replay or live episode.  High beta means more
    width, deeper patience and weaker pruning; low beta means fewer probes,
    earlier stagnation stops and stronger pruning.  Offline evaluation sweeps a
    beta grid to check that the policy really exposes that trade-off; that sweep
    is not online beta adaptation.
    """

    NAME = "LLMDesignedMethod"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})

    # -- the two methods a proposed policy must implement --------------------
    def solve(self, question, budget: Optional[int] = None) -> SimResult:
        raise NotImplementedError("a policy must implement solve()")

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        """Decide the width/depth of the *next* live grid.

        Runs before a new live grid is created; it must never inspect a current
        episode's outcomes, and it must always return a non-``None`` plan.  The
        base implementation returns the context's conservative bootstrap plan so
        that a policy which forgets to override it is still runnable -- but the
        runner flags it, because delegating grid choice to the fallback is
        explicitly not allowed for a proposed policy.
        """
        return context.bootstrap_plan("template stub: policy did not override plan_grid")

    # -- convenience ---------------------------------------------------------
    def _schedule(self, beta: float) -> Dict[str, float]:
        """Route every behavioural threshold through one beta schedule."""
        b = max(0.0, min(1.0, float(beta)))
        return {
            "beta": b,
            "width": b,
            "patience": b,
            "prune": 1.0 - b,
        }

    def describe(self) -> Dict[str, Any]:
        return {"name": getattr(self, "NAME", type(self).__name__), "config": dict(self.config)}


# -- helpers referenced verbatim by the prompt's minimal structure -----------
def _budget_done(question, budget: Optional[int] = None) -> bool:
    """Stop condition shared by the live and replay loops.

    ``budget`` is ``None`` during replay, so the decision-round cap is the only
    hard bound and the policy is expected to terminate on its own.
    """
    if question.decision_rounds >= question.max_decision_rounds:
        return True
    if budget is not None and question.budget_spent >= budget:
        return True
    return False


def _record_curve(res: SimResult, question) -> None:
    """Sample ``(total_probes, best_score_so_far)`` at every reveal."""
    res.total_probes = question.budget_spent
    res.decision_rounds = question.decision_rounds
    res.effective_sequential_rounds = question.effective_sequential_rounds
    if question.best_so_far > res.best_score:
        res.best_score = question.best_so_far
        res.best_cell = question.best_cell
    res.curve.append((question.budget_spent, question.best_so_far))


def finalize_result(question, res: SimResult, stopped_reason: str = "") -> SimResult:
    """Close out an episode with the environment's own bookkeeping."""
    res.total_probes = question.budget_spent
    res.decision_rounds = question.decision_rounds
    res.effective_sequential_rounds = question.effective_sequential_rounds
    res.best_score = max(res.best_score, question.best_so_far)
    res.best_cell = question.best_cell or res.best_cell
    res.revealed = sorted(question.observed())
    res.batches = [list(b) for b in getattr(question, "batches", [])]
    if stopped_reason:
        res.stopped_reason = stopped_reason
    elif not res.stopped_reason:
        res.stopped_reason = (
            "decision_round_cap" if question.decision_rounds >= question.max_decision_rounds
            else "empty_batch"
        )
    if not res.curve:
        res.curve = [(0, float("-inf"))]
    return res


def legal_batch(question, cells: Sequence[str]) -> List[str]:
    """Filter a wish list down to a batch the environment will accept."""
    legal = set(question.legal_actions())
    out: List[str] = []
    seen = set()
    for c in cells:
        if c in seen or c not in legal:
            continue
        b, _, a = c[1:].partition("a")
        parent = f"b{b}a{int(a) - 1}" if int(a) > 0 else None
        if parent is not None and parent in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= question.max_parallelism:
            break
    return out
