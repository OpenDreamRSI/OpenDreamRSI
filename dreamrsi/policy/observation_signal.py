"""Prefix-derived signals a policy may use.

Every helper here reads only revealed observations, so using them cannot leak an
unrevealed outcome.  They implement the success semantics the policy prompt
insists on: an evaluated observation with ``error is None`` and
``fail_class == "ok"`` is a *successful evaluation*, even when ``valid`` is
false or ``n_valid``/``n_total`` are unavailable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..types import REPAIRABLE_FAIL_CLASSES, Observation


def is_success(obs: Observation) -> bool:
    return obs.is_success


def probe_improved_vs_parent(obs: Observation, *, eps: float = 0.0) -> bool:
    """Did this attempt improve on the workspace it resumed from?"""
    return obs.is_success and obs.delta_vs_parent > eps


def probe_improved_vs_baseline(obs: Observation, *, eps: float = 0.0) -> bool:
    """Did this attempt beat the task baseline?"""
    return obs.is_success and obs.delta_vs_baseline > eps


def branch_trajectory(observed: Dict[str, Observation], branch: int) -> List[Observation]:
    """The branch's ordered prefix trajectory -- not just its latest observation."""
    return sorted((o for o in observed.values() if o.branch == branch), key=lambda o: o.attempt)


def successful_anchor(observed: Dict[str, Observation], branch: int) -> Optional[Observation]:
    """Best score on the branch that came from a *successful* evaluation."""
    best = None
    for o in branch_trajectory(observed, branch):
        if o.is_success and (best is None or o.score > best.score):
            best = o
    return best


def branch_promising(observed: Dict[str, Observation], branch: int,
                     *, baseline: float = 0.0, eps: float = 0.0) -> bool:
    """Is there prefix evidence worth spending another probe on?

    True when the branch has a successful anchor above the baseline, or when it
    is still shallow enough that a weak early score is not yet evidence.
    """
    traj = branch_trajectory(observed, branch)
    if not traj:
        return True
    anchor = successful_anchor(observed, branch)
    if anchor is not None and anchor.score - baseline > eps:
        return True
    if len(traj) <= 1:
        return True  # shallow weak scores are not enough to discard a branch
    return any(probe_improved_vs_parent(o, eps=eps) for o in traj)


def branch_failed_hard(observed, branch: int = None) -> bool:
    """Hard-unrecoverable evidence for a branch (or for a single observation).

    Accepts either ``(observed_dict, branch)`` or a single :class:`Observation`,
    because the prompt uses ``branch_failed_hard(obs)`` in both shapes.  This is
    a *signal*, not unconditional closure: a later successful result reopens the
    branch and cancels a closure based only on an earlier failure.
    """
    if isinstance(observed, Observation):
        o = observed
        return (not o.is_success) and o.fail_class not in REPAIRABLE_FAIL_CLASSES \
            and o.fail_class != "env"
    traj = branch_trajectory(observed, branch)
    if not traj:
        return False
    latest = traj[-1]
    if latest.is_success:
        return False  # a later success cancels an earlier closure
    if latest.fail_class in REPAIRABLE_FAIL_CLASSES or latest.fail_class == "env":
        return False
    # unrecoverable only with cumulative evidence
    tail = [o for o in traj[-2:] if not o.is_success]
    return len(tail) >= 2 or (len(traj) == 1 and latest.fail_class == "timeout")


def repairable_failure(obs: Observation) -> bool:
    """Output/correctness mismatch, resource limits and code errors are repairable."""
    return obs.is_repairable_failure


def zero_valid(obs: Observation) -> bool:
    """``n_valid == 0`` -- a signal, never automatic closure."""
    return obs.n_valid == 0


def remaining_depth(observed: Dict[str, Observation], branch: int, refine_count: int) -> int:
    traj = branch_trajectory(observed, branch)
    return max(0, refine_count - (traj[-1].attempt if traj else -1))


def score_trend(observed: Dict[str, Observation], branch: int, window: int = 3) -> float:
    """Mean per-attempt gain over the last ``window`` successful attempts."""
    traj = [o for o in branch_trajectory(observed, branch) if o.is_success]
    if len(traj) < 2:
        return 0.0
    tail = traj[-window:]
    if len(tail) < 2:
        return 0.0
    return (tail[-1].score - tail[0].score) / (len(tail) - 1)


def best_observed(observed: Dict[str, Observation]) -> Optional[Observation]:
    best = None
    for o in observed.values():
        if best is None or o.score > best.score:
            best = o
    return best
