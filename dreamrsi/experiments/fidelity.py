"""Is a frozen discovery history really an *exact* simulator?

Dream-RSI's whole economy rests on one sentence in Section 2: replaying a
different exploration policy over a completed trace means traversing different
branches, in a different order, with different parallel groupings -- "and every
outcome is already stored".  If that is true, one expensive online run funds
thousands of zero-cost off-policy evaluations.  If it is only approximately
true, every offline reward the dreaming loop maximises is measured against a
simulator that quietly disagrees with the live environment, and the selected
policy is optimised for a fiction.

The claim decomposes into two measurable halves, and they do not have the same
answer.

**Exactness (this module's** :func:`replay_fidelity` **).**  Deploy ``pi`` live,
freeze the tree, write it to disk, load it back and replay the *same* ``pi``.
The two trajectories must agree cell for cell: same batches in the same order,
same revealed scores, same best score, same probe count.  This is a real
end-to-end check -- it runs through serialisation, so it fails if freezing,
storing or serving a world ever stops reproducing what the live run saw.

**Reach (**:func:`support_coverage` **).**  Exactness only ever covers *the
region that was explored*.  A replay world offers the policy a strictly smaller
menu than the live grid did: roots the online run never opened cannot be opened
in the dream, and a branch cannot be refined past its recorded depth.  So a
dreaming policy is not evaluated on the decisions it would actually make online
-- it is evaluated on the closest decisions the frozen trace can serve.  This
module measures that restriction directly, by reconstructing at every replay
decision round the legal set the *live* grid would have offered for the same
revealed prefix and comparing it with what the trace can actually serve.

The distinction matters for reading any offline number in this repository:
:func:`replay_fidelity` says the simulator does not lie, and
:func:`support_coverage` says how much of the world it declines to show.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..online import run_online_cycle
from ..policy.loader import PolicySpec, load_policy_source
from ..tasks.base import Task
from ..tree import DiscoveryTree
from ..types import GridPlan, cell_id, parse_cell_id
from ..world import ReplayWorld


@dataclass
class FidelityReport:
    """Live trajectory vs. its own replay, compared decision round by decision round."""

    world_id: str
    policy: str
    beta: float
    identical: bool
    live_rounds: int
    replay_rounds: int
    matched_rounds: int
    live_probes: int
    replay_probes: int
    live_best: Optional[float]
    replay_best: Optional[float]
    divergence_round: Optional[int] = None
    divergence: Optional[Dict[str, List[str]]] = None
    score_mismatches: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        """The full claim: identical trajectory *and* identical revealed outcomes."""
        return self.identical and not self.score_mismatches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id": self.world_id,
            "policy": self.policy,
            "beta": self.beta,
            "exact": self.exact,
            "identical_trajectory": self.identical,
            "live_rounds": self.live_rounds,
            "replay_rounds": self.replay_rounds,
            "matched_rounds": self.matched_rounds,
            "live_probes": self.live_probes,
            "replay_probes": self.replay_probes,
            "live_best": self.live_best,
            "replay_best": self.replay_best,
            "divergence_round": self.divergence_round,
            "divergence": self.divergence,
            "score_mismatches": self.score_mismatches,
        }


@dataclass
class SupportReport:
    """How much of the live decision space a frozen trace can actually serve."""

    world_id: str
    policy: str
    beta: float
    rounds: int
    mean_action_coverage: float
    restricted_rounds: int
    unavailable_actions: int
    unavailable_roots: int
    unavailable_refinements: int
    planned_branch_count: Optional[int]
    planned_refine_count: Optional[int]
    trace_branch_count: int
    trace_refine_count: int
    probes: int
    best_score: Optional[float]
    terminal_live_actions: int = 0
    terminal_replay_actions: int = 0
    stopped_reason: str = ""
    per_round: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def stopped_by_exhaustion(self) -> bool:
        """The episode ended because the trace ran out, not because the policy chose to.

        This is the restriction that ``mean_action_coverage`` cannot see.  Coverage
        averages over the rounds the policy *took*, and a replay that dies of
        exhaustion takes no round at the moment it dies -- the policy asks for the
        legal set, gets nothing, and stops.  A replay ending this way was cut off
        mid-search: the live grid still had somewhere to go and the frozen history
        did not, so whatever the policy would have done next was never scored.
        """
        return self.terminal_replay_actions == 0 and self.terminal_live_actions > 0

    @property
    def planned_grid_known(self) -> bool:
        """Whether the trace recorded the live grid it was collected under.

        Without it there is nothing to compare against and coverage is reported
        as ``1.0`` by construction -- which is an absence of evidence, not
        evidence that the trace covers everything.
        """
        return self.planned_branch_count is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id": self.world_id,
            "policy": self.policy,
            "beta": self.beta,
            "rounds": self.rounds,
            "mean_action_coverage": self.mean_action_coverage,
            "restricted_rounds": self.restricted_rounds,
            "unavailable_actions": self.unavailable_actions,
            "unavailable_roots": self.unavailable_roots,
            "unavailable_refinements": self.unavailable_refinements,
            "planned_grid_known": self.planned_grid_known,
            "planned_branch_count": self.planned_branch_count,
            "planned_refine_count": self.planned_refine_count,
            "trace_branch_count": self.trace_branch_count,
            "trace_refine_count": self.trace_refine_count,
            "probes": self.probes,
            "best_score": self.best_score,
            "terminal_live_actions": self.terminal_live_actions,
            "terminal_replay_actions": self.terminal_replay_actions,
            "stopped_by_exhaustion": self.stopped_by_exhaustion,
            "stopped_reason": self.stopped_reason,
            "per_round": self.per_round,
        }


class _AuditedReplayWorld(ReplayWorld):
    """A replay world that records what the live grid *would* have offered.

    At every decision round, before charging the batch, it reconstructs the legal
    set :class:`~dreamrsi.online.LiveWorld` would expose for the same revealed
    prefix under the grid the trace was collected with.  The difference between
    that set and the trace's own legal set is exactly the part of the decision
    space dreaming cannot reach.
    """

    def __init__(self, tree: DiscoveryTree, *, planned: Optional[GridPlan] = None,
                 **kwargs: Any) -> None:
        super().__init__(tree, **kwargs)
        self.planned = planned
        self.audit: List[Dict[str, Any]] = []

    def _live_legal(self) -> List[str]:
        """``LiveWorld.legal_actions()`` for the current revealed prefix."""
        if self.planned is None:
            return list(self.legal_actions())
        opened = set(self.opened_branches())
        out = [cell_id(b, 0) for b in range(self.planned.branch_count) if b not in opened]
        for b in sorted(opened):
            depth = max((o.attempt for o in self._observed.values() if o.branch == b),
                        default=-1)
            if depth + 1 <= self.planned.refine_count:
                out.append(cell_id(b, depth + 1))
        return out

    def probe_batch(self, cells, on_reveal=None):
        if not list(cells):
            # An empty batch costs no decision round, so auditing one would
            # invent a round that never happened and dilute the average.
            return super().probe_batch(cells, on_reveal=on_reveal)
        live_legal = set(self._live_legal())
        replay_legal = set(self.legal_actions())
        missing = sorted(live_legal - replay_legal)
        self.audit.append({
            "round": self.decision_rounds + 1,
            "prefix_size": len(self._observed),
            "live_legal": len(live_legal),
            "replay_legal": len(replay_legal),
            "coverage": (len(replay_legal) / len(live_legal)) if live_legal else 1.0,
            "unavailable": missing,
        })
        return super().probe_batch(cells, on_reveal=on_reveal)


def replay_fidelity(tree: DiscoveryTree, spec: PolicySpec, *,
                    live_batches: Sequence[Sequence[str]],
                    live_observed: Dict[str, float],
                    live_best: float,
                    live_probes: int,
                    config: Optional[Dict[str, Any]] = None,
                    label: Optional[str] = None,
                    max_parallelism: int = 4,
                    max_decision_rounds: int = 64) -> FidelityReport:
    """Replay ``spec`` over its own frozen ``tree`` and diff against the live run.

    ``live_observed`` maps every cell the online rollout revealed to the score it
    saw at the time.  Comparing against that -- rather than against the tree the
    replay world is reading from -- is what makes this a check of the whole
    freeze-store-serve path instead of a tautology.

    ``label`` names the policy in the report.  Every candidate module declares
    ``NAME = "OptimalPolicy"`` as the prompt requires, so ``spec.name`` cannot
    tell two of them apart and the caller supplies something that can.
    """
    beta = float((config or {}).get("beta", 0.6))
    world = ReplayWorld(tree, max_parallelism=max_parallelism,
                        max_decision_rounds=max_decision_rounds)
    result = spec.instantiate(dict(config or {})).solve(world, None)

    replay_batches = [list(b) for b in world.batches]
    live = [list(b) for b in live_batches]

    matched = 0
    divergence_round: Optional[int] = None
    divergence: Optional[Dict[str, List[str]]] = None
    for i, (a, b) in enumerate(zip(live, replay_batches)):
        if a != b:
            divergence_round = i + 1
            divergence = {"live": a, "replay": b}
            break
        matched += 1
    if divergence_round is None and len(live) != len(replay_batches):
        divergence_round = matched + 1
        longer = "live" if len(live) > len(replay_batches) else "replay"
        extra = (live if longer == "live" else replay_batches)[matched]
        divergence = {"live": extra if longer == "live" else [],
                      "replay": extra if longer == "replay" else []}

    mismatches: List[Dict[str, Any]] = []
    for cid, obs in world.observed().items():
        if cid not in live_observed:
            continue  # only cells the live run actually revealed are comparable
        recorded, replayed = live_observed[cid], obs.score
        if recorded != replayed and not (_is_neg_inf(recorded) and _is_neg_inf(replayed)):
            mismatches.append({"cell": cid,
                               "live": _clean(recorded),
                               "replay": _clean(replayed)})

    return FidelityReport(
        world_id=tree.world_id,
        policy=label or spec.name,
        beta=beta,
        identical=(divergence_round is None),
        live_rounds=len(live),
        replay_rounds=len(replay_batches),
        matched_rounds=matched,
        live_probes=int(live_probes),
        replay_probes=int(result.total_probes),
        live_best=_clean(live_best),
        replay_best=_clean(result.best_score),
        divergence_round=divergence_round,
        divergence=divergence,
        score_mismatches=mismatches,
    )


def support_coverage(tree: DiscoveryTree, spec: PolicySpec, *,
                     config: Optional[Dict[str, Any]] = None,
                     label: Optional[str] = None,
                     max_parallelism: int = 4,
                     max_decision_rounds: int = 64) -> SupportReport:
    """Replay ``spec`` over ``tree`` and measure what the trace could not offer.

    ``spec`` need not be the policy that produced ``tree`` -- in fact the
    interesting case is when it is not, because that is what dreaming does.
    """
    planned = _planned_grid(tree)
    world = _AuditedReplayWorld(tree, planned=planned,
                                max_parallelism=max_parallelism,
                                max_decision_rounds=max_decision_rounds)
    result = spec.instantiate(dict(config or {})).solve(world, None)

    coverages = [r["coverage"] for r in world.audit]
    missing_roots, missing_refinements, missing_total = 0, 0, 0
    for row in world.audit:
        for cid in row["unavailable"]:
            missing_total += 1
            if parse_cell_id(cid)[1] == 0:
                missing_roots += 1
            else:
                missing_refinements += 1

    # The state the episode ended in.  A policy that stops because it has run
    # out of recorded cells never takes a decision round at that prefix, so this
    # restriction is invisible to the per-round audit above.
    terminal_live = world._live_legal()
    terminal_replay = world.legal_actions()

    return SupportReport(
        world_id=tree.world_id,
        policy=label or spec.name,
        beta=float((config or {}).get("beta", 0.6)),
        rounds=len(world.audit),
        mean_action_coverage=(sum(coverages) / len(coverages)) if coverages else 1.0,
        restricted_rounds=sum(1 for r in world.audit if r["unavailable"]),
        unavailable_actions=missing_total,
        unavailable_roots=missing_roots,
        unavailable_refinements=missing_refinements,
        planned_branch_count=None if planned is None else planned.branch_count,
        planned_refine_count=None if planned is None else planned.refine_count,
        trace_branch_count=tree.branch_count,
        trace_refine_count=tree.refine_count,
        probes=int(result.total_probes),
        best_score=_clean(result.best_score),
        terminal_live_actions=len(terminal_live),
        terminal_replay_actions=len(terminal_replay),
        stopped_reason=result.stopped_reason,
        per_round=world.audit,
    )


def fidelity_probe(task: Task, agent, source: str, *, run_dir: str, history_dir: str,
                   baseline_dir: str, problem_file: str, baseline_score: float,
                   plan: GridPlan, beta: float = 0.62, name: str = "policy",
                   max_parallelism: int = 6, max_decision_rounds: int = 40,
                   budget: Optional[int] = None, cycle: int = 1) -> Dict[str, Any]:
    """Deploy a policy live, freeze it, reload it from disk and replay it.

    Returns the fidelity report, the support report for the same policy on its
    own trace, and the trace path so callers can reuse the world.
    """
    spec = load_policy_source(source, name=name, strict=False,
                              path=os.path.join(run_dir, "method.py"))
    config = {"beta": beta}
    cycle_out = run_online_cycle(
        task=task, agent=agent, policy=spec.instantiate(config), plan=plan,
        run_dir=run_dir, history_dir=history_dir, baseline_dir=baseline_dir,
        problem_file=problem_file, baseline_score=baseline_score,
        max_parallelism=max_parallelism, max_decision_rounds=max_decision_rounds,
        budget=budget, world_id=f"{name}-b{beta}", cycle=cycle,
    )
    live_world = cycle_out["world"]
    live_observed = {cid: obs.score for cid, obs in live_world.observed().items()}

    # Reload rather than reuse: the claim is about a *stored* history.
    frozen = DiscoveryTree.load(cycle_out["trace_path"])

    report = replay_fidelity(
        frozen, spec,
        live_batches=live_world.batches,
        live_observed=live_observed,
        live_best=cycle_out["result"].best_score,
        live_probes=cycle_out["result"].total_probes,
        config=config,
        label=name,
        max_parallelism=max_parallelism,
        max_decision_rounds=max_decision_rounds,
    )
    support = support_coverage(frozen, spec, config=config, label=name,
                               max_parallelism=max_parallelism,
                               max_decision_rounds=max_decision_rounds)
    return {"fidelity": report, "support": support, "tree": frozen,
            "trace_path": cycle_out["trace_path"], "spec": spec}


def _planned_grid(tree: DiscoveryTree) -> Optional[GridPlan]:
    plan = (tree.meta or {}).get("plan") or {}
    try:
        return GridPlan(branch_count=int(plan["branch_count"]),
                        refine_count=int(plan["refine_count"]))
    except (KeyError, TypeError, ValueError):
        return None


def _is_neg_inf(v: float) -> bool:
    return v == float("-inf")


def _clean(v: float) -> Optional[float]:
    return None if v == float("-inf") else float(v)
