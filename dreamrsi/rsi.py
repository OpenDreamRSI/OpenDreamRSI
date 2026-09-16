"""The outer recursive self-improvement loop.

Iteration ``t``:

1. **Online exploration.** ``pi_t`` guides the discovery agent over a live grid
   chosen by ``pi_t.plan_grid``; the completed tree ``T_t`` is appended to the
   history, ``H_t = H_{t-1} u {T_t}``.
2. **Simulator construction.** ``T_t`` joins the simulator pool.
3. **Dreaming.** ``M`` policy revisions are replayed over the whole frozen pool;
   the best evaluated version becomes ``pi_{t+1}``.

``Mode.FIXED`` skips step 3 and keeps the policy static: that is the paper's
controlled baseline, Recursive Fixed Exploration.  Both modes share the same
discovery agent, evaluator, initialisation and per-round resource budget.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .agents import make_discovery_agent, make_policy_developer
from .agents.base import DiscoveryAgent, PolicyDeveloper
from .dreaming import DEFAULT_BETA_GRID, dream
from .objective import DEFAULT_LAMBDA
from .online import run_online_cycle
from .policy.loader import load_policy_source
from .storage import RunLayout, write_json
from .tasks import make_task
from .tasks.base import Task
from .types import GridPlanningContext
from .world import SimulatorPool

FIXED = "fixed"
DREAM = "dream"


@dataclass
class RSIConfig:
    """Everything a run needs; mirrors ``configs/*.yaml``."""

    name: str = "run"
    task: str = "synthetic"
    task_config: Dict[str, Any] = field(default_factory=dict)
    mode: str = DREAM                      # "dream" (Dream-RSI) or "fixed" (baseline)
    rounds: int = 5                        # outer iterations t = 1..T
    revisions: int = 3                     # M policy revisions per offline phase
    max_parallelism: int = 6               # W concurrent workers
    max_decision_rounds: int = 40          # K1 online / K2 replay
    branch_count: int = 6                  # initial grid width
    refine_count: int = 12                 # initial grid depth (R refinements)
    hard_max_branch_count: int = 32
    hard_max_refine_count: int = 32
    budget: Optional[int] = None           # probe cap per online round
    beta: float = 0.62
    betas: Sequence[float] = DEFAULT_BETA_GRID
    lam: float = DEFAULT_LAMBDA
    seed: int = 0
    #: Ablation knob: dream over only the newest ``pool_window`` worlds instead
    #: of all of ``H_t``.  ``None`` keeps the paper's evolving-worlds setting,
    #: where the simulator pool grows with every completed round.
    pool_window: Optional[int] = None
    out_dir: str = "runs"
    discovery_agent: str = "mock"
    policy_developer: str = "mock"
    initial_policy: Optional[str] = None   # path to the starting policy source
    direction_guidance: str = ""
    strict_policy_check: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RSIConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _progress(*args: Any) -> None:
    """Progress goes to stderr; stdout stays machine-readable."""
    print(*args, file=sys.stderr, flush=True)


def build_components(config: RSIConfig):
    """The task, discovery agent and policy developer ``config`` asks for.

    ``FIXED`` gets no developer at all, which is what makes it the paper's
    controlled baseline rather than a dreaming run with the result ignored.
    """
    task = make_task(config.task, config.task_config)
    agent = make_discovery_agent(config.discovery_agent, task,
                                 {"seed": config.seed, **config.task_config.get("agent", {})})
    # ``developer`` falls back to the discovery agent's settings, because the
    # usual case is one CLI or one API key driving both roles; ``policy_agent``
    # overrides it when the two should differ.
    dev_config = dict(config.task_config.get("policy_agent")
                      or config.task_config.get("agent", {}))
    developer = (make_policy_developer(config.policy_developer, dev_config)
                 if config.mode == DREAM else None)
    return task, agent, developer


def run_rsi(config: RSIConfig, *, task: Task, agent: DiscoveryAgent,
            developer: Optional[PolicyDeveloper] = None,
            initial_source: Optional[str] = None,
            log=_progress) -> Dict[str, Any]:
    """Execute ``config.rounds`` outer iterations and return the run report."""
    layout = RunLayout(os.path.join(config.out_dir, config.name))
    layout.create()

    paths = task.prepare(layout.history_dir)
    baseline_score = task.baseline_score(layout.history_dir)

    source = initial_source or _read_initial_policy(config)
    spec = load_policy_source(source, name="pi_1", strict=False,
                              path=os.path.join(layout.history_dir, "r0000_init", "method.py"))
    policy_config: Dict[str, Any] = {"beta": config.beta}

    pool = SimulatorPool(max_parallelism=config.max_parallelism,
                         max_decision_rounds=config.max_decision_rounds)
    manifests: List[Dict[str, Any]] = []
    live_trend: List[float] = []
    rounds: List[Dict[str, Any]] = []
    cumulative_calls = 0
    started = time.time()

    for t in range(1, config.rounds + 1):
        policy = spec.instantiate(policy_config)

        ctx = GridPlanningContext(
            history=tuple(manifests),
            fallback_branch_count=config.branch_count,
            fallback_refine_count=config.refine_count,
            hard_max_branch_count=config.hard_max_branch_count,
            hard_max_refine_count=config.hard_max_refine_count,
            worker_cap=config.max_parallelism,
            mode="live",
        )
        plan = policy.plan_grid(ctx)
        if plan is None:
            plan = ctx.bootstrap_plan("policy returned None; runner fallback")
        plan = plan.clamped(ctx)

        round_dir = layout.round_dir(t, config.mode)
        log(f"[{config.name}] round {t}/{config.rounds} "
            f"grid W={plan.branch_count} R={plan.refine_count} "
            f"({plan.reason or 'no reason given'})")

        cycle = run_online_cycle(
            task=task, agent=agent, policy=policy, plan=plan,
            run_dir=round_dir,
            history_dir=layout.history_dir,
            baseline_dir=paths["baseline_dir"],
            problem_file=paths["problem_file"],
            baseline_score=baseline_score,
            max_parallelism=config.max_parallelism,
            max_decision_rounds=config.max_decision_rounds,
            budget=config.budget,
            world_id=f"{config.name}-iter{t}",
            direction_guidance=config.direction_guidance,
            cycle=t,
        )

        manifest = cycle["manifest"]
        manifests.append(manifest)
        pool.append(cycle["tree"])
        cumulative_calls += int(manifest.get("agent_calls", 0))
        best = manifest.get("best_score")
        live_trend.append(float("-inf") if best is None else float(best))

        layout.publish_manifest(t, manifest)
        with open(os.path.join(round_dir, "policy_used.py"), "w", encoding="utf-8") as fh:
            fh.write(spec.source)

        record: Dict[str, Any] = {
            "round": t,
            "mode": config.mode,
            "policy": spec.name,
            "beta": policy_config.get("beta"),
            "plan": {"branch_count": plan.branch_count, "refine_count": plan.refine_count,
                     "reason": plan.reason},
            "manifest": manifest,
            "cumulative_agent_calls": cumulative_calls,
        }

        # -- offline phase: dreaming over the whole accumulated pool ---------
        if config.mode == DREAM and developer is not None and t < config.rounds:
            dream_pool = _dream_pool(pool, config)
            log(f"[{config.name}]   dreaming over {len(dream_pool)} world(s), "
                f"M={config.revisions} revisions")
            out = dream(
                current_source=spec.source,
                pool=dream_pool,
                developer=developer,
                revisions=config.revisions,
                out_dir=round_dir,
                history_dir=layout.history_dir,
                trace_pool_dir=layout.trace_pool_dir,
                betas=config.betas,
                lam=config.lam,
                base_config=policy_config,
                live_trend=[v for v in live_trend if v != float("-inf")],
                strict=config.strict_policy_check,
            )
            chosen = out["best"]
            record["dream"] = {
                "selected_version": chosen.version,
                "pool_worlds": len(dream_pool),
                "improved": bool(out["improved"]),
                "incumbent_reward": out["incumbent"].summary.reward,
                "selected_reward": chosen.summary.reward,
                "candidates": [
                    {"version": e.version, "reward": (None if e.score == float("-inf") else e.score),
                     "error": e.error}
                    for e in out["evaluations"]
                ],
            }
            log(f"[{config.name}]   selected v{chosen.version} "
                f"(reward {chosen.summary.reward:.4f} vs incumbent "
                f"{out['incumbent'].summary.reward:.4f})")
            if chosen.version != 0:
                next_path = os.path.join(layout.history_dir,
                                         f"r{t:04d}_selected", "method.py")
                os.makedirs(os.path.dirname(next_path), exist_ok=True)
                spec = load_policy_source(chosen.source, name=f"pi_{t + 1}",
                                          strict=False, path=next_path)
                policy_config = {"beta": _baked_beta(chosen.source, policy_config.get("beta", 0.6))}

        rounds.append(record)

    report = {
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in config.__dict__.items()},
        "baseline_score": baseline_score,
        "rounds": rounds,
        "pool": pool.summary(),
        "best_score": max((r["manifest"].get("best_score") or float("-inf")) for r in rounds),
        "cumulative_agent_calls": cumulative_calls,
        "seconds": round(time.time() - started, 3),
    }
    write_json(os.path.join(layout.root, "report.json"), report)
    return report


def _dream_pool(pool: SimulatorPool, config: RSIConfig) -> SimulatorPool:
    """The worlds this round's offline phase is allowed to dream over.

    Dream-RSI dreams over the whole accumulated history, which is what makes the
    simulator improve as the loop runs.  ``pool_window`` exists to ablate exactly
    that: restricted to the newest ``K`` worlds, the pool stops growing and the
    offline phase can only ever see a sliding window of recent history.
    """
    window = config.pool_window
    if not window or window >= len(pool):
        return pool
    return SimulatorPool(pool.trees[-int(window):],
                         max_parallelism=pool.max_parallelism,
                         max_decision_rounds=pool.max_decision_rounds)


def _baked_beta(source: str, fallback: float) -> float:
    import re

    m = re.search(r'self\.config\.get\("beta",\s*([0-9.]+)\)', source)
    return float(m.group(1)) if m else float(fallback)


def _read_initial_policy(config: RSIConfig) -> str:
    """pi_1: a caller-supplied policy, else the manual parallel-refine policy."""
    if config.initial_policy:
        with open(config.initial_policy, "r", encoding="utf-8") as fh:
            return fh.read()
    from .policy import baseline

    return baseline.source()
