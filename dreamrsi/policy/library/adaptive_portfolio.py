"""Adaptive portfolio exploration policy.

Prefix signals
    Per-branch ordered trajectories (not just the latest observation): the
    successful anchor, the parent->child gain sequence, the stall run, the
    failure/repair sequence, and the remaining depth.  Every threshold is
    expressed relative to the revealed prefix -- no absolute score cutoff, no
    cell id, no target is baked in.

Batch rule
    One dynamic portfolio per decision round, sized up to
    ``question.max_parallelism``: strong normal refinements (exploitation), new
    roots or underexplored branches (exploration), and at most one actual
    repairable failure (recovery).  Exploration and justified recovery get a
    slot before the remaining slots are filled by priority; roles are not fixed
    quotas.  A parent and its child are never in the same batch, and a batch is
    never left short while an eligible candidate exists and a worker is idle.

Beta schedule
    ``beta`` is read once in ``__init__`` and is fixed for the episode.  It is
    routed through a single ``_schedule(beta)``: high beta widens the portfolio,
    lengthens stall patience, weakens pruning and keeps recovery eligible; low
    beta narrows, stops on stagnation sooner and prunes harder.

Default-beta rationale
    0.62 -- the "history is insufficient or evidence conflicts" default of the
    cross-cycle rule, biased slightly exploratory because the replay ceiling of
    a frozen trace is not a live stopping signal.

Grid-planning rule
    ``plan_grid`` reads only completed live manifests: it widens when distinct
    roots keep paying off early while refinements stall, deepens when gains
    arrive late on a small repeatable set, widens again when everything
    plateaus at sufficient depth, and shrinks conservatively after repeated
    unrecoverable failures.  With no usable history it returns the context's
    explicit conservative bootstrap plan.

Safeguards
    Over-pruning: a branch is closed only on cumulative evidence, and a
    successful anchor is never erased by a later repairable failure.
    Over-stopping: the loop stops only after the whole revealed portfolio
    (active, underexplored, recoverable, unopened) has been considered.
    Starvation: a repairable failure keeps eligibility; it is deprioritised, not
    deleted.  Serial probes: single-cell batches are only emitted when exactly
    one candidate is eligible.
"""

from __future__ import annotations

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
    probe_improved_vs_parent,
    score_trend,
    successful_anchor,
)

NAME = "OptimalPolicy"

_EPS = 1e-12


class OptimalPolicy(LLMDesignedMethod):
    """Trajectory-ranked portfolio search over an irregular branch x attempt grid."""

    NAME = "OptimalPolicy"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.62))

    # -- one schedule for every behavioural threshold ------------------------
    def _schedule(self, beta: float) -> Dict[str, float]:
        b = max(0.0, min(1.0, float(beta)))
        return {
            "beta": b,
            # how many branches we are willing to keep open, as a fraction of W
            "width": 0.35 + 0.65 * b,
            # consecutive non-improving successful attempts tolerated
            "patience": 1.0 + 2.0 * b,
            # how far below the leader a branch may sit before it is pruned
            "prune_margin": 0.55 - 0.40 * b,
            # relative gain below which an attempt counts as a stall
            "stall_eps": 0.02 * (1.0 - b),
            # recovery is eligible at all, and how far it is deprioritised
            "recovery": 1.0 if b >= 0.2 else 0.0,
            "recovery_discount": 0.35 + 0.35 * (1.0 - b),
            # weight on a rising trend versus a high anchor
            "trend_weight": 0.5 + 1.0 * b,
            # decision rounds without a portfolio-level gain before stopping
            "stop_patience": 1.0 + 3.0 * b,
            # relative gain that counts as portfolio progress
            "progress_eps": 0.01 + 0.04 * (1.0 - b),
        }

    # -- main loop -----------------------------------------------------------
    def solve(self, question, budget: Optional[int] = None) -> SimResult:
        question.reset()
        sch = self._schedule(self.beta)
        res = SimResult()
        closed: set = set()
        stale, best_seen, reason = 0, None, ""

        while not _budget_done(question, budget):
            prefix = question.observed()
            self._update_closed(closed, prefix, question, sch)

            # portfolio-level progress, derived from the revealed prefix only
            current = _prefix_best(prefix)
            if current is not None:
                if best_seen is None or current > best_seen + self._eps(prefix, sch):
                    best_seen, stale = current, 0
                else:
                    stale += 1
            if stale >= sch["stop_patience"] and not self._has_recovery(prefix, question, closed):
                reason = (f"portfolio stagnated for {stale} decision rounds at beta="
                          f"{sch['beta']:.2f}; no eligible recovery remained")
                break

            batch = self._select_batch(prefix, question, closed, sch)
            if not batch:
                reason = "no eligible candidate remained in the revealed portfolio"
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))

        return finalize_result(question, res, reason)

    @staticmethod
    def _eps(prefix: Dict[str, Observation], sch: Dict[str, float]) -> float:
        """Progress threshold, scaled to the spread already revealed."""
        scores = [o.score for o in prefix.values() if o.is_success]
        if len(scores) < 2:
            return 0.0
        span = max(scores) - min(scores)
        return sch["progress_eps"] * max(span, _EPS)

    def _has_recovery(self, prefix: Dict[str, Observation], question, closed: set) -> bool:
        """An eligible repairable failure blocks a stagnation stop."""
        if not prefix:
            return False
        roots = set(question.legal_roots())
        for cid in question.legal_actions():
            if cid in roots:
                continue
            branch = question.meta(cid).branch
            if branch in closed:
                continue
            traj = branch_trajectory(prefix, branch)
            if traj and traj[-1].is_repairable_failure:
                return True
        return False

    # -- step 1: close only on cumulative evidence ---------------------------
    def _update_closed(self, closed: set, prefix: Dict[str, Observation], question,
                       sch: Dict[str, float]) -> None:
        stats = self._branch_stats(prefix, question)
        leader = max((s["anchor_rel"] for s in stats.values()), default=0.0)
        for branch, s in stats.items():
            if s["anchor"] is not None and s["latest_success"] and s["stalls"] == 0:
                closed.discard(branch)  # a later improving success reopens the branch
                continue
            if branch_failed_hard(prefix, branch):
                closed.add(branch)
                continue
            if s["repairable"]:
                continue  # eligibility is retained; priority is lowered instead
            stalled = s["stalls"] >= sch["patience"]
            far_behind = (leader - s["anchor_rel"]) > sch["prune_margin"]
            exhausted = s["attempts"] >= 2 and s["trend"] <= 0.0 and stalled
            if s["attempts"] >= 2 and (exhausted or (stalled and far_behind)):
                closed.add(branch)

    # -- steps 2-5: rank, then build one dynamic portfolio --------------------
    def _select_batch(self, prefix: Dict[str, Observation], question, closed: set,
                      sch: Dict[str, float]) -> List[str]:
        width = question.max_parallelism
        legal = question.legal_actions()
        if not legal:
            return []
        roots = set(question.legal_roots())
        frontiers = [c for c in legal if c not in roots]
        stats = self._branch_stats(prefix, question)
        opened = question.opened_branches()
        active = [b for b in opened if b not in closed]

        # rank open frontiers, splitting normal refinements from recoveries
        exploit: List[Tuple[float, str]] = []
        recover: List[Tuple[float, str]] = []
        for cid in frontiers:
            branch = question.meta(cid).branch
            if branch in closed:
                continue
            s = stats.get(branch, {})
            pr = self._frontier_priority(s, sch)
            if s.get("repairable"):
                recover.append((pr * sch["recovery_discount"], cid))
            else:
                exploit.append((pr, cid))
        exploit.sort(key=lambda t: (-t[0], t[1]))
        recover.sort(key=lambda t: (-t[0], t[1]))

        # reserve threshold: how far below the leader a refinement may sit and
        # still be worth a worker.  Relative to the prefix, never absolute.
        reserve = -sch["prune_margin"]
        eligible_exploit = [(pr, cid) for pr, cid in exploit if pr >= reserve]

        # exploration pressure
        stagnant = bool(active) and all(
            stats.get(b, {}).get("stalls", 0) >= sch["patience"] for b in active
        )
        target_width = max(1, min(len(opened) + len(roots),
                                  int(round((0.6 + sch["width"]) * width)) or 1))
        may_widen = bool(roots) and (len(active) < target_width or stagnant
                                     or not eligible_exploit)

        # step 5: stop only after considering the whole revealed portfolio --
        # active, underexplored, recoverable, unopened and remaining legal.
        if not eligible_exploit and not recover and not may_widen and not exploit:
            return []

        batch: List[str] = []
        if may_widen:
            batch.append(sorted(roots)[0])                 # exploration first
        if recover and sch["recovery"] > 0 and len(batch) < width:
            batch.append(recover[0][1])                    # at most one recovery
        for _, cid in eligible_exploit:                    # then by priority
            if len(batch) >= width:
                break
            if cid not in batch:
                batch.append(cid)
        # Never leave a worker idle while legal, non-closed work remains: extra
        # roots and reserved (below-threshold but not closed) frontiers are free
        # in decision-round terms and keep the batch parallel.
        for cid in sorted(roots):
            if len(batch) >= width:
                break
            if cid not in batch:
                batch.append(cid)
        for _, cid in exploit:
            if len(batch) >= width:
                break
            if cid not in batch:
                batch.append(cid)
        for _, cid in recover[1:]:
            if len(batch) >= width:
                break
            if cid not in batch:
                batch.append(cid)

        return batch[:width]

    # -- prefix-derived branch statistics ------------------------------------
    def _branch_stats(self, prefix: Dict[str, Observation], question) -> Dict[int, dict]:
        baseline = question.baseline_score
        anchors: Dict[int, Optional[Observation]] = {}
        out: Dict[int, dict] = {}
        branches = sorted({o.branch for o in prefix.values()})
        for b in branches:
            anchors[b] = successful_anchor(prefix, b)

        scores = [a.score for a in anchors.values() if a is not None]
        top = max(scores) if scores else baseline
        span = max(abs(top - baseline), _EPS)

        for b in branches:
            traj = branch_trajectory(prefix, b)
            anchor = anchors[b]
            stalls = 0
            for o in reversed(traj):
                if o.is_success and not probe_improved_vs_parent(o, eps=span * 1e-6):
                    stalls += 1
                elif o.is_success:
                    break
            latest = traj[-1] if traj else None
            out[b] = {
                "branch": b,
                "attempts": len(traj),
                "anchor": anchor,
                "anchor_rel": ((anchor.score - baseline) / span) if anchor is not None else 0.0,
                "trend": score_trend(prefix, b) / span,
                "stalls": stalls,
                "latest_success": bool(latest and latest.is_success),
                "repairable": bool(latest and latest.is_repairable_failure),
                "fails": sum(1 for o in traj if not o.is_success),
                "depth": (latest.attempt + 1) if latest else 0,
            }
        return out

    @staticmethod
    def _frontier_priority(s: Dict, sch: Dict[str, float]) -> float:
        if not s:
            return 0.0
        return (
            s.get("anchor_rel", 0.0)
            + sch["trend_weight"] * s.get("trend", 0.0)
            - 0.12 * s.get("stalls", 0) * (1.0 - sch["beta"] * 0.5)
            - 0.05 * s.get("fails", 0)
        )

    # -- next-cycle grid planning -------------------------------------------
    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        hist = [h for h in context.history if isinstance(h, dict)][-3:]
        if not hist:
            return context.bootstrap_plan(
                "no completed live manifest yet; conservative bootstrap from context fallbacks"
            )

        last = hist[-1]
        # Anchor on the grid that was *planned*, not on the depth the controller
        # happened to use: anchoring on realized depth ratchets the grid down
        # every time the policy stops early.
        w = int(last.get("planned_branch_count") or context.fallback_branch_count)
        r = int(last.get("planned_refine_count") or context.fallback_refine_count)
        used_w = int(last.get("effective_branch_count") or 0)
        used_r = int(last.get("effective_refine_count") or 0)

        early = float(last.get("early_gain_share", 0.0))    # share of gain from roots
        late = float(last.get("late_gain_share", 0.0))      # share from deep refinements
        plateau = bool(last.get("plateaued", False))
        uncovered = int(last.get("unopened_roots", 0))
        hard_fails = int(last.get("hard_failures", 0))
        probes = max(1, int(last.get("probes", 1)))

        improving = len(hist) >= 2 and _best(hist[-1]) > _best(hist[-2]) + 1e-12
        # width and depth are only worth holding if they are actually being used
        underused_depth = len(hist) >= 2 and all(
            int(h.get("effective_refine_count") or 0) + 2 < int(h.get("planned_refine_count") or 0)
            for h in hist[-2:]
        )
        underused_width = len(hist) >= 2 and all(
            int(h.get("effective_branch_count") or 0) < int(h.get("planned_branch_count") or 0)
            for h in hist[-2:]
        )

        if hard_fails * 3 >= probes:
            plan = GridPlan(max(1, w - 1), max(1, r - 1),
                            "repeated unrecoverable failures: shrinking width and depth conservatively")
        elif late > early + 0.15:
            plan = GridPlan(max(2, w - 1) if w > 2 else w, r + 1,
                            "gains arrived late on a small repeatable set: deepen, hold width")
        elif early > late + 0.15 and plateau:
            plan = GridPlan(w + 1, max(1, r - 1),
                            "distinct roots paid off early while refinements stalled: widen, trim depth")
        elif plateau and uncovered > 0:
            plan = GridPlan(w + 1, r,
                            "all explored directions plateaued while direction classes remain uncovered: widen")
        elif underused_depth:
            plan = GridPlan(w, max(used_r + 2, r - 1),
                            f"planned depth {r} but only {used_r} was used for two rounds: trim to fit")
        elif underused_width:
            plan = GridPlan(max(used_w + 1, w - 1), r,
                            f"planned width {w} but only {used_w} roots were opened twice running: "
                            "trim to fit")
        elif improving:
            plan = GridPlan(w, r, "live best still improving: hold the planned grid")
        else:
            plan = GridPlan(w, r, "evidence conflicts: hold the last planned grid")

        return plan.clamped(context)


def _prefix_best(prefix: Dict[str, Observation]) -> Optional[float]:
    """Best score among successful evaluations in the revealed prefix."""
    scores = [o.score for o in prefix.values() if o.is_success]
    return max(scores) if scores else None


def _best(manifest: Dict) -> float:
    v = manifest.get("best_score")
    return float("-inf") if v is None else float(v)
