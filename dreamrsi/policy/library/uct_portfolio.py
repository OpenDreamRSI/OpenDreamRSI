"""Confidence-ranked portfolio exploration policy.

Prefix signals
    Same trajectory reconstruction as the portfolio policy -- successful anchor,
    parent->child gains, stall run, failure/repair sequence -- but the ranking
    adds an explicit optimism term: a branch that has been probed few times
    keeps an exploration bonus, so a weak *shallow* score cannot bury a
    direction before it has been given depth.

Batch rule
    Rank every legal frontier by ``anchor + c(beta) * sqrt(log(1 + probes) /
    attempts)``, rank unopened roots at the mean optimism of an unprobed arm,
    and fill the batch from the merged ranking up to
    ``question.max_parallelism``, reserving one slot for an actual repairable
    failure when one is eligible.

Beta schedule
    ``beta`` scales the confidence coefficient ``c``, the stall patience and the
    closure margin through a single ``_schedule``.  It is fixed for the episode.

Default-beta rationale
    0.55 -- the optimism term already supplies exploration pressure, so the
    default sits a little below the portfolio policy's to avoid paying for width
    twice.

Grid-planning rule
    Width follows how much of the optimism budget went to roots that then paid
    off; depth follows realized late gains.  With no usable history it returns
    the context's conservative bootstrap plan.

Safeguards
    A branch is closed only on cumulative evidence; the optimism term decays
    with attempts rather than with score, so a repairable failure lowers
    priority without erasing the branch; batches are filled before a worker is
    left idle; stopping requires the whole revealed portfolio to be unpromising.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from see.policy.api import (
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    Observation,
    SimResult,
    _budget_done,
    _record_curve,
    finalize_result,
)
from see.policy.observation_signal import (
    branch_failed_hard,
    branch_trajectory,
    score_trend,
    successful_anchor,
)

NAME = "OptimalPolicy"

_EPS = 1e-12


class OptimalPolicy(LLMDesignedMethod):
    """Upper-confidence allocation over an irregular branch x attempt grid."""

    NAME = "OptimalPolicy"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.55))

    def _schedule(self, beta: float) -> Dict[str, float]:
        b = max(0.0, min(1.0, float(beta)))
        return {
            "beta": b,
            "c": 0.15 + 0.85 * b,          # confidence coefficient
            "patience": 1.0 + 3.0 * b,     # stalls tolerated before closure
            "prune_margin": 0.6 - 0.45 * b,
            "stop_patience": 1.0 + 3.0 * b,
            "progress_eps": 0.01 + 0.04 * (1.0 - b),
            "recovery": 1.0 if b >= 0.15 else 0.0,
        }

    def solve(self, question, budget: Optional[int] = None) -> SimResult:
        question.reset()
        sch = self._schedule(self.beta)
        res = SimResult()
        closed: set = set()
        stale, best_seen, reason = 0, None, ""

        while not _budget_done(question, budget):
            prefix = question.observed()
            self._update_closed(closed, prefix, question, sch)

            current = _prefix_best(prefix)
            if current is not None:
                if best_seen is None or current > best_seen + _progress_eps(prefix, sch):
                    best_seen, stale = current, 0
                else:
                    stale += 1
            if stale >= sch["stop_patience"]:
                reason = f"no portfolio gain for {stale} decision rounds at beta={sch['beta']:.2f}"
                break

            batch = self._select_batch(prefix, question, closed, sch)
            if not batch:
                reason = "every legal candidate was closed on cumulative evidence"
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))

        return finalize_result(question, res, reason)

    def _update_closed(self, closed: set, prefix: Dict[str, Observation], question,
                       sch: Dict[str, float]) -> None:
        stats = self._stats(prefix, question)
        leader = max((s["anchor_rel"] for s in stats.values()), default=0.0)
        for branch, s in stats.items():
            if s["latest_success"] and s["stalls"] == 0:
                closed.discard(branch)
                continue
            if branch_failed_hard(prefix, branch):
                closed.add(branch)
                continue
            if s["repairable"]:
                continue
            if (s["attempts"] >= 2 and s["stalls"] >= sch["patience"]
                    and (leader - s["anchor_rel"]) > sch["prune_margin"]):
                closed.add(branch)

    def _select_batch(self, prefix: Dict[str, Observation], question, closed: set,
                      sch: Dict[str, float]) -> List[str]:
        width = question.max_parallelism
        legal = question.legal_actions()
        if not legal:
            return []
        roots = set(question.legal_roots())
        stats = self._stats(prefix, question)
        total = max(1, sum(s["attempts"] for s in stats.values()))

        ranked: List[Tuple[float, str]] = []
        recovery: List[Tuple[float, str]] = []
        for cid in legal:
            branch = question.meta(cid).branch
            if cid in roots:
                # an unprobed arm carries the full optimism bonus
                ranked.append((sch["c"] * math.sqrt(math.log(1.0 + total)), cid))
                continue
            if branch in closed:
                continue
            s = stats.get(branch, {})
            n = max(1, s.get("attempts", 1))
            bonus = sch["c"] * math.sqrt(math.log(1.0 + total) / n)
            score = s.get("anchor_rel", 0.0) + 0.5 * s.get("trend", 0.0) + bonus
            if s.get("repairable"):
                recovery.append((score * 0.5, cid))
            else:
                ranked.append((score, cid))

        if not ranked and not recovery:
            return []
        ranked.sort(key=lambda t: (-t[0], t[1]))
        recovery.sort(key=lambda t: (-t[0], t[1]))

        batch: List[str] = []
        if recovery and sch["recovery"] > 0:
            batch.append(recovery[0][1])
        for _, cid in ranked:
            if len(batch) >= width:
                break
            if cid not in batch:
                batch.append(cid)
        for _, cid in recovery[1:]:
            if len(batch) >= width:
                break
            if cid not in batch:
                batch.append(cid)
        return batch[:width]

    def _stats(self, prefix: Dict[str, Observation], question) -> Dict[int, dict]:
        baseline = question.baseline_score
        branches = sorted({o.branch for o in prefix.values()})
        anchors = {b: successful_anchor(prefix, b) for b in branches}
        scores = [a.score for a in anchors.values() if a is not None]
        top = max(scores) if scores else baseline
        span = max(abs(top - baseline), _EPS)

        out: Dict[int, dict] = {}
        for b in branches:
            traj = branch_trajectory(prefix, b)
            anchor = anchors[b]
            stalls = 0
            for o in reversed(traj):
                if o.is_success and o.delta_vs_parent <= 0:
                    stalls += 1
                elif o.is_success:
                    break
            latest = traj[-1] if traj else None
            out[b] = {
                "attempts": len(traj),
                "anchor_rel": ((anchor.score - baseline) / span) if anchor is not None else 0.0,
                "trend": score_trend(prefix, b) / span,
                "stalls": stalls,
                "latest_success": bool(latest and latest.is_success),
                "repairable": bool(latest and latest.is_repairable_failure),
            }
        return out

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        hist = [h for h in context.history if isinstance(h, dict)][-3:]
        if not hist:
            return context.bootstrap_plan(
                "no completed live manifest; conservative bootstrap from context fallbacks")
        last = hist[-1]
        w = int(last.get("planned_branch_count") or context.fallback_branch_count)
        r = int(last.get("planned_refine_count") or context.fallback_refine_count)
        early = float(last.get("early_gain_share", 0.0))
        late = float(last.get("late_gain_share", 0.0))
        uncovered = int(last.get("unopened_roots", 0))

        if late > early + 0.2:
            plan = GridPlan(w, r + 1, "optimism kept paying at depth: deepen")
        elif uncovered == 0 and early >= late:
            plan = GridPlan(w + 1, r, "every root was opened and roots carried the gain: widen")
        else:
            plan = GridPlan(w, r, "confidence ranking still covers the planned grid: hold")
        return plan.clamped(context)


def _prefix_best(prefix: Dict[str, Observation]) -> Optional[float]:
    scores = [o.score for o in prefix.values() if o.is_success]
    return max(scores) if scores else None


def _progress_eps(prefix: Dict[str, Observation], sch: Dict[str, float]) -> float:
    scores = [o.score for o in prefix.values() if o.is_success]
    if len(scores) < 2:
        return 0.0
    return sch["progress_eps"] * max(max(scores) - min(scores), _EPS)
