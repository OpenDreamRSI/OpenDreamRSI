"""Stages 2 and 3 -- Simulator Construction and Dreaming-based Policy Improvement.

During the offline phase of outer iteration ``t`` the history ``H_t`` is frozen
while the exploration policy undergoes ``M`` code revisions.  Every version is
evaluated on every recorded tree, so the evaluation score of version ``pi_t^m``
is its average replay score across the fixed history,
``V^m = (1/t) * sum_i V_i^m``.

The candidate set always contains the current policy ``pi_t^0 = pi_t``, so the
selected ``pi_{t+1} = pi_t^{m*}`` satisfies ``V^{m*} >= V^0``: the next online
policy is never worse than the current one in average replay score on ``H_t``.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents.base import PolicyDeveloper, RevisionRequest
from .objective import (
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_LAMBDA,
    attainment,
    attainment_auc,
    envelope_auc,
    parallel_penalty,
    replay_value,
    summarize_sweep,
)
from .policy.api import LLMDesignedMethod, finalize_result
from .policy.loader import PolicyRejected, PolicySpec, load_policy_source
from .types import ParetoSummary, SimResult, SweepPoint
from .world import IllegalBatch, ReplayWorld, SimulatorPool

#: The beta grid swept during offline evaluation.
DEFAULT_BETA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class PolicyEvaluation:
    """Everything one policy version earned on the frozen history."""

    version: int
    name: str
    source: str
    summary: ParetoSummary
    points: List[SweepPoint] = field(default_factory=list)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def score(self) -> float:
        """``V^m`` -- the quantity policy selection maximises."""
        return float("-inf") if self.error else self.summary.reward

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "error": self.error,
            "warnings": self.warnings,
            "score": None if self.score == float("-inf") else self.score,
            "pareto": self.summary.to_dict(),
            "points": [p.to_dict() for p in self.points],
        }


def replay_episode(policy: LLMDesignedMethod, world: ReplayWorld) -> Tuple[SimResult, Optional[str]]:
    """Run one replay episode; illegal or crashing policies fail softly."""
    try:
        result = policy.solve(world, None)
    except IllegalBatch as exc:
        return _salvage(world, f"illegal_batch: {exc}"), f"illegal batch: {exc}"
    except Exception as exc:  # noqa: BLE001 - the developer agent wrote this code
        return (_salvage(world, f"crash: {exc}"),
                f"policy crashed: {type(exc).__name__}: {exc}")
    return result, None


def _salvage(world: ReplayWorld, reason: str) -> SimResult:
    """Close a failed episode with the *world's* record of what it really did.

    A policy that fails does not get to file its own report -- it never reaches
    ``finalize_result`` -- and returning a blank :class:`SimResult` scores it as
    though it had done nothing. That is wrong in a way that matters to the loop:
    a revision that explored competently for three rounds and then hit a typo
    becomes indistinguishable from one that never selected a batch, both landing
    on the same floor reward, and the reflector Appendix B.2 asks to "diagnose
    from the trajectory" is handed no trajectory to diagnose.

    The world knows better, so it is asked instead. Its round log carries
    ``(total_probes, best_so_far)`` after every completed round, which is the
    attainment curve, and is recorded by the simulator rather than by the policy
    -- so it survives exactly the failure that lost the policy's own copy.
    """
    result = SimResult()
    result.curve = [
        (int(row["total_probes"]),
         float("-inf") if row["best_so_far"] is None else float(row["best_so_far"]))
        for row in world.round_log
    ]
    return finalize_result(world, result, reason)


def evaluate_policy(spec: PolicySpec, pool: SimulatorPool, *,
                    betas: Sequence[float] = DEFAULT_BETA_GRID,
                    lam: float = DEFAULT_LAMBDA,
                    beta1: float = DEFAULT_BETA1,
                    beta2: float = DEFAULT_BETA2,
                    base_config: Optional[Dict[str, Any]] = None,
                    version: int = 0,
                    collect_traces: bool = True) -> PolicyEvaluation:
    """Sweep ``beta`` over every frozen tree in the pool.

    One replay episode per ``(frozen trace, beta)``.  Because every episode only
    reveals recorded outcomes, this costs no generation and no evaluation.
    """
    points: List[SweepPoint] = []
    traces: List[Dict[str, Any]] = []
    errors: List[str] = []
    # per world: one attainment-vs-probes curve for every beta in the sweep
    curves: Dict[str, List[List[Tuple[int, float]]]] = {}
    horizons: Dict[str, int] = {}

    for beta in betas:
        config = dict(base_config or {})
        config["beta"] = float(beta)
        for world in pool.worlds():
            try:
                policy = spec.instantiate(config)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"instantiation failed: {exc!r}")
                continue
            result, err = replay_episode(policy, world)
            if err:
                errors.append(f"beta={beta} world={world.world_id}: {err}")
            horizon = max(1, len(world.tree))
            horizons[world.world_id] = horizon
            curves.setdefault(world.world_id, []).append([
                (p, attainment(s, world.baseline_score, world.ceiling))
                for p, s in result.curve
            ])
            att = attainment(result.best_score, world.baseline_score, world.ceiling)
            auc = attainment_auc(result, baseline=world.baseline_score,
                                 ceiling=world.ceiling, horizon=horizon)
            rv = replay_value(result, beta1=beta1, beta2=beta2, quality=att)
            points.append(SweepPoint(
                world_id=world.world_id,
                beta=float(beta),
                auc=auc,
                attainment=att,
                total_probes=result.total_probes,
                decision_rounds=result.decision_rounds,
                effective_sequential_rounds=result.effective_sequential_rounds,
                parallel_penalty=parallel_penalty(result),
                best_score=None if result.best_score == float("-inf") else result.best_score,
                replay_value=rv.value,
            ))
            if collect_traces:
                traces.append({
                    "policy_version": version,
                    "world_id": world.world_id,
                    "beta": float(beta),
                    "horizon": horizon,
                    "stopped_reason": result.stopped_reason,
                    "error": err,
                    "rounds": world.round_log,
                    "result": result.to_dict(),
                    "replay_value": rv.to_dict(),
                })

    frontier = (_mean([envelope_auc(cs, horizons[wid]) for wid, cs in curves.items()])
                if curves else None)
    summary = summarize_sweep(points, lam=lam, frontier_auc=frontier)
    return PolicyEvaluation(
        version=version,
        name=spec.name,
        source=spec.source,
        summary=summary,
        points=points,
        traces=traces,
        warnings=list(spec.warnings),
        # Any failed episode disqualifies the version, which makes ``score``
        # ``-inf`` and keeps it out of selection. It is not enough to score a
        # crasher low: the selected policy is *redeployed online*, where the
        # same crash costs a whole round of real discovery-agent calls. Before
        # :func:`_salvage` this was accidentally load-bearing in a different
        # way -- a crashed episode still appended a zeroed ``SweepPoint``, so
        # ``points`` was never empty, so this branch never fired, and only the
        # floor reward of an empty episode kept crashers from winning. Salvaging
        # the trajectory removes that floor, so the disqualification has to be
        # real. The points and traces are still recorded: the reflector is meant
        # to read how far the version got before it broke.
        error="; ".join(errors[:3]) if errors else None,
    )


def dream(current_source: str, pool: SimulatorPool, developer: PolicyDeveloper, *,
          revisions: int = 3, out_dir: str, history_dir: str = "", trace_pool_dir: str = "",
          betas: Sequence[float] = DEFAULT_BETA_GRID, lam: float = DEFAULT_LAMBDA,
          base_config: Optional[Dict[str, Any]] = None,
          live_trend: Sequence[float] = (), method_file: str = "method.py",
          strict: bool = False) -> Dict[str, Any]:
    """Evaluate ``pi^0`` and ``M`` revisions on the frozen pool; return the winner.

    ``M`` is ``revisions``.  Every version is replayed on the same worlds with
    the same beta grid, so the comparison is apples-to-apples, and the incumbent
    is always in the candidate set.
    """
    os.makedirs(out_dir, exist_ok=True)
    proposals_dir = os.path.join(out_dir, "proposal_results")
    os.makedirs(proposals_dir, exist_ok=True)

    evaluations: List[PolicyEvaluation] = []
    sources: List[str] = [current_source]

    # -- pi^0: the incumbent, so selection can never regress -----------------
    ev0 = _evaluate_source(current_source, pool, version=0, betas=betas, lam=lam,
                           base_config=base_config, strict=False,
                           path=os.path.join(proposals_dir, "v000_method.py"))
    evaluations.append(ev0)

    feedback = _feedback(ev0, live_trend, base_config)
    for m in range(revisions):
        request = RevisionRequest(
            method_file=method_file,
            current_source=sources[-1],
            history_dir=history_dir,
            trace_pool_dir=trace_pool_dir,
            feedback=feedback,
            revision_index=m,
        )
        try:
            proposed = developer.revise(request)
        except Exception as exc:  # noqa: BLE001
            evaluations.append(PolicyEvaluation(
                version=m + 1, name="rejected", source="",
                summary=summarize_sweep([], lam=lam),
                error=f"developer failed: {type(exc).__name__}: {exc}",
            ))
            continue
        sources.append(proposed)
        ev = _evaluate_source(proposed, pool, version=m + 1, betas=betas, lam=lam,
                              base_config=base_config, strict=strict,
                              path=os.path.join(proposals_dir, f"v{m + 1:03d}_method.py"))
        evaluations.append(ev)
        feedback = _feedback(ev, live_trend, base_config)

    best = max(evaluations, key=lambda e: e.score)
    if best.score == float("-inf"):
        best = ev0  # nothing ran: keep the incumbent

    _archive(proposals_dir, evaluations, best)
    return {
        "best": best,
        "evaluations": evaluations,
        "improved": best.version != 0 and best.score > ev0.score + 1e-12,
        "incumbent": ev0,
    }


def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else 0.0


def _evaluate_source(source: str, pool: SimulatorPool, *, version: int,
                     betas: Sequence[float], lam: float,
                     base_config: Optional[Dict[str, Any]], strict: bool,
                     path: str) -> PolicyEvaluation:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    try:
        spec = load_policy_source(source, name=f"v{version:03d}", strict=strict, path=path)
    except PolicyRejected as exc:
        return PolicyEvaluation(version=version, name=f"v{version:03d}", source=source,
                                summary=summarize_sweep([], lam=lam),
                                error=f"rejected: {exc}")
    except Exception as exc:  # noqa: BLE001
        return PolicyEvaluation(version=version, name=f"v{version:03d}", source=source,
                                summary=summarize_sweep([], lam=lam),
                                error=f"load failed: {type(exc).__name__}: {exc}\n"
                                      f"{traceback.format_exc(limit=2)}")
    return evaluate_policy(spec, pool, betas=betas, lam=lam,
                           base_config=base_config, version=version)


def _feedback(ev: PolicyEvaluation, live_trend: Sequence[float],
              base_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "current_beta": float((base_config or {}).get("beta", 0.6)),
        "live_trend": list(live_trend),
        "pareto": ev.summary.to_dict(),
        "per_beta": ev.summary.per_beta,
        "non_degenerate": ev.summary.non_degenerate,
        "warnings": ev.warnings,
        "error": ev.error,
    }


def _archive(proposals_dir: str, evaluations: Sequence[PolicyEvaluation],
             best: PolicyEvaluation) -> None:
    """Write ``beta_sweep.json`` and ``policy_execution_traces.jsonl``.

    These are between-round feedback: the improvement prompt tells the agent to
    read them outside ``solve()`` and never to copy a trace-specific cell id,
    score or target into policy logic.
    """
    sweep = {
        "selected_version": best.version,
        "versions": [e.to_dict() for e in evaluations],
        "pareto": best.summary.to_dict(),
    }
    with open(os.path.join(proposals_dir, "beta_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(sweep, fh, indent=2)
        fh.write("\n")

    with open(os.path.join(proposals_dir, "policy_execution_traces.jsonl"), "w",
              encoding="utf-8") as fh:
        for ev in evaluations:
            for tr in ev.traces:
                fh.write(json.dumps(tr) + "\n")
