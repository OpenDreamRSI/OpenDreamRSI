"""Replay objectives.

Two closely-related scores appear in the paper and are both implemented here.

1. Equation (1) -- the replay objective of Section 3::

       V_i^m = max_{v in T_i^{m,k*}} s_v  -  beta1 * N_i^m  +  beta2 * N_i^m / max(1, k_i^{m,*})

   discovery quality, execution cost, and a parallelism bonus that rewards
   batching useful continuations rather than executing them sequentially.

2. The ``pareto`` reward the policy-development prompt (Listing 2) optimises::

       pareto.reward = pareto.auc - lambda * parallel_penalty

   where ``auc`` rewards reaching high per-trace attainment with few total
   probes and ``parallel_penalty`` is the mean of
   ``effective_sequential_rounds / total_probes`` over the beta sweep.  A serial
   policy has penalty near 1; useful full batches approach ``1 / W``.

The sweep also reports whether it is *non-degenerate*: beta must actually move
the attainment/work trade-off, otherwise the knob is cosmetic.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .types import ParetoSummary, ReplayScore, SimResult, SweepPoint

#: Default coefficients for Equation (1).  Scores are normalised to [0, 1]
#: attainment before the penalty terms are applied, so these are unitless.
DEFAULT_BETA1 = 0.002
DEFAULT_BETA2 = 0.02
#: Default ``lambda`` for the pareto reward.
DEFAULT_LAMBDA = 0.25


def replay_value(result: SimResult, *, beta1: float = DEFAULT_BETA1,
                 beta2: float = DEFAULT_BETA2, quality: Optional[float] = None) -> ReplayScore:
    """Equation (1) for a single ``(policy version, frozen tree)`` replay."""
    n = int(result.total_probes)
    k = max(1, int(result.decision_rounds))
    q = result.best_score if quality is None else quality
    if q == float("-inf"):
        q = 0.0
    cost = beta1 * n
    bonus = beta2 * (n / k) if n else 0.0
    return ReplayScore(
        discovery_quality=float(q),
        execution_cost=float(cost),
        parallelism_bonus=float(bonus),
        value=float(q - cost + bonus),
        revealed_nodes=n,
        decision_rounds=int(result.decision_rounds),
    )


def attainment(best_score: float, baseline: float, ceiling: float) -> float:
    """Per-trace attainment in ``[0, 1]``.

    ``0`` reproduces the baseline, ``1`` reaches the frozen trace's own ceiling.
    Using the trace ceiling (rather than an absolute target) keeps the objective
    comparable across worlds with very different score scales.
    """
    if best_score == float("-inf"):
        return 0.0
    span = ceiling - baseline
    if not math.isfinite(span) or span <= 1e-12:
        return 1.0 if best_score >= ceiling - 1e-12 else 0.0
    return max(0.0, min(1.0, (best_score - baseline) / span))


def attainment_auc(result: SimResult, *, baseline: float, ceiling: float,
                   horizon: int) -> float:
    """Area under the attainment-vs-probes curve, normalised by ``horizon``.

    The curve is held flat after the policy stops, so stopping early at high
    attainment is rewarded and burning probes for a late gain is not.
    """
    if horizon <= 0:
        return 0.0
    curve = sorted(result.curve)
    total, idx, cur = 0.0, 0, 0.0
    for p in range(1, horizon + 1):
        while idx < len(curve) and curve[idx][0] <= p:
            cur = max(cur, attainment(curve[idx][1], baseline, ceiling))
            idx += 1
        total += cur
    return total / horizon


def envelope_auc(curves: Sequence[Sequence[Tuple[int, float]]], horizon: int) -> float:
    """Area under the upper envelope of several attainment-vs-probes curves.

    This is ``pareto.auc``: it aggregates a whole beta sweep on one frozen trace
    by asking, for every probe budget, the best attainment *any* setting of the
    knob reached by then.  A knob that genuinely trades work for attainment lifts
    the envelope at both ends; a cosmetic knob leaves it equal to the single
    curve it always produces -- which is exactly the degenerate case the
    improvement prompt warns about.
    """
    if horizon <= 0 or not curves:
        return 0.0
    best_at = [0.0] * (horizon + 1)
    for curve in curves:
        for probes, att in curve:
            idx = min(max(int(probes), 0), horizon)
            if att > best_at[idx]:
                best_at[idx] = att
    total, cur = 0.0, 0.0
    for p in range(1, horizon + 1):
        cur = max(cur, best_at[p])
        total += cur
    return total / horizon


def parallel_penalty(result: SimResult) -> float:
    """``effective_sequential_rounds / total_probes`` for one episode."""
    if result.total_probes <= 0:
        return 1.0
    return result.effective_sequential_rounds / result.total_probes


def summarize_sweep(points: Sequence[SweepPoint], *, lam: float = DEFAULT_LAMBDA,
                    frontier_auc: Optional[float] = None) -> ParetoSummary:
    """Aggregate a beta sweep into the reward the dreaming loop maximises.

    ``pareto.reward = pareto.auc - lambda * parallel_penalty`` where
    ``pareto.auc`` is the frontier area across the sweep (falling back to the
    mean per-episode AUC when no frontier was supplied) and ``parallel_penalty``
    is, as the prompt specifies, the *mean* of
    ``effective_sequential_rounds / total_probes`` over the sweep.
    """
    if not points:
        return ParetoSummary(auc=0.0, parallel_penalty=1.0, reward=-1.0,
                             attainment=0.0, mean_probes=0.0, replay_value=0.0)
    auc = _mean(p.auc for p in points) if frontier_auc is None else float(frontier_auc)
    pen = _mean(p.parallel_penalty for p in points)
    att = _mean(p.attainment for p in points)
    probes = _mean(float(p.total_probes) for p in points)
    value = _mean(p.replay_value for p in points)

    per_beta: Dict[str, Dict[str, float]] = {}
    for p in points:
        key = f"{p.beta:.3f}"
        slot = per_beta.setdefault(key, {"auc": 0.0, "attainment": 0.0, "total_probes": 0.0,
                                         "parallel_penalty": 0.0, "replay_value": 0.0, "n": 0.0})
        slot["auc"] += p.auc
        slot["attainment"] += p.attainment
        slot["total_probes"] += p.total_probes
        slot["parallel_penalty"] += p.parallel_penalty
        slot["replay_value"] += p.replay_value
        slot["n"] += 1
    for slot in per_beta.values():
        n = slot.pop("n") or 1.0
        for k in list(slot):
            slot[k] = slot[k] / n

    atts = [s["attainment"] for s in per_beta.values()]
    works = [s["total_probes"] for s in per_beta.values()]
    non_degenerate = (len(per_beta) > 1
                      and (max(atts) - min(atts) > 1e-6 or max(works) - min(works) > 0.5))

    return ParetoSummary(
        auc=auc,
        parallel_penalty=pen,
        reward=auc - lam * pen,
        attainment=att,
        mean_probes=probes,
        replay_value=value,
        per_beta=per_beta,
        non_degenerate=bool(non_degenerate),
    )


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else 0.0
