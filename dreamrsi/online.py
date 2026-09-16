"""Stage 1 -- Online Exploration.

The current policy ``pi_t`` guides the *discovery agent* to collect a discovery
trace ``T_t`` with tree structure.  Continuing from the root opens a new branch;
continuing from a non-root leaf refines the workspace at the end of an existing
branch.  Each attempt contributes a new child carrying its updated workspace,
artifact, score and diagnostics.

The live world exposes exactly the same decision interface as the replay world;
the only difference is that here a continuation *generates* a new outcome (one
discovery-agent call plus one evaluation) instead of revealing a recorded one.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from .agents.base import AttemptContext, DiscoveryAgent
from .tasks.base import EvalResult, Task
from .tree import DiscoveryTree, Node
from .types import ROOT_ID, CellMeta, GridPlan, Observation, cell_id
from .world import BaseWorld


class LiveWorld(BaseWorld):
    """The online environment: probing a cell really runs the discovery agent."""

    def __init__(self, task: Task, agent: DiscoveryAgent, plan: GridPlan, *,
                 run_dir: str, history_dir: str, baseline_dir: str, problem_file: str,
                 baseline_score: float = 0.0, max_parallelism: int = 4,
                 max_decision_rounds: int = 64, world_id: str = "live",
                 direction_guidance: str = "", max_workers: Optional[int] = None,
                 cycle: int = 1,
                 on_node: Optional[Callable[[Node], None]] = None) -> None:
        super().__init__(max_parallelism=max_parallelism,
                         max_decision_rounds=max_decision_rounds,
                         baseline_score=baseline_score, world_id=world_id)
        self.task = task
        self.agent = agent
        self.plan = plan
        self.run_dir = run_dir
        self.history_dir = history_dir
        self.baseline_dir = baseline_dir
        self.problem_file = problem_file
        self.direction_guidance = direction_guidance
        self.cycle = int(cycle)
        self.max_workers = int(max_workers or max_parallelism)
        self.on_node = on_node
        self.tree = DiscoveryTree(world_id=world_id, baseline_score=baseline_score,
                                  task=task.name,
                                  meta={"plan": {"branch_count": plan.branch_count,
                                                 "refine_count": plan.refine_count,
                                                 "reason": plan.reason}})
        self.wall_clock = 0.0
        self.agent_calls = 0

    # -- structure defined by the grid plan ----------------------------------
    def legal_roots(self) -> List[str]:
        opened = set(self.opened_branches())
        return [cell_id(b, 0) for b in range(self.plan.branch_count) if b not in opened]

    def frontiers(self) -> List[str]:
        out: List[str] = []
        for b in self.opened_branches():
            depth = max((o.attempt for o in self._observed.values() if o.branch == b), default=-1)
            if depth + 1 <= self.plan.refine_count:
                out.append(cell_id(b, depth + 1))
        return out

    def legal_actions(self) -> List[str]:
        return self.legal_roots() + self.frontiers()

    def meta(self, cid: str) -> CellMeta:
        node = self.tree.get(cid)
        if node is not None:
            return self.tree.meta_of(cid)
        b, _, a = cid[1:].partition("a")
        b, a = int(b), int(a)
        return CellMeta(cell_id=cid, branch=b, attempt=a,
                        parent_id=ROOT_ID if a == 0 else cell_id(b, a - 1))

    # -- transition: generate a new outcome ----------------------------------
    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], Any]] = None) -> List[Observation]:
        batch = self._validate(cells)
        if not batch:
            return []
        started = time.time()
        if len(batch) == 1 or self.max_workers <= 1:
            results = [self._run_one(c) for c in batch]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as pool:
                results = list(pool.map(self._run_one, batch))
        self.wall_clock += time.time() - started
        prefix_size = len(self._observed)
        self._charge(batch)

        out: List[Observation] = []
        for cid, (node, _) in zip(batch, results):
            self.tree.add(node)
            if self.on_node is not None:
                self.on_node(node)
            obs = self.tree.observation_of(cid)
            self._record(obs)
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        self._log_round(batch, out, prefix_size)
        return out

    # -- one CONTINUE(v) -----------------------------------------------------
    def _run_one(self, cid: str):
        meta = self.meta(cid)
        # Every path handed to an agent is absolute.  A CLI agent is launched
        # *inside* the attempt directory, so a path relative to the runner's
        # working directory would silently fail to resolve -- and the failure is
        # invisible, because the prompt tells the agent to read the history and
        # an agent that cannot find it simply proposes without one.
        node_dir = os.path.abspath(os.path.join(self.run_dir, "attempts", f"attempt_{cid}"))
        os.makedirs(node_dir, exist_ok=True)
        parent_dir = None
        if meta.attempt > 0:
            parent_dir = os.path.abspath(
                os.path.join(self.run_dir, "attempts", f"attempt_{meta.parent_id}"))

        siblings = []
        attempts_root = os.path.abspath(os.path.join(self.run_dir, "attempts"))
        if os.path.isdir(attempts_root):
            siblings = [os.path.join(attempts_root, d) for d in sorted(os.listdir(attempts_root))
                        if d.startswith("attempt_") and os.path.join(attempts_root, d) != node_dir]

        ctx = AttemptContext(
            node_dir=node_dir,
            run_dir=os.path.abspath(self.run_dir),
            history_dir=os.path.abspath(self.history_dir),
            baseline_dir=os.path.abspath(self.baseline_dir),
            problem_file=os.path.abspath(self.problem_file),
            eval_program=self.task.eval_program,
            branch=meta.branch,
            attempt=meta.attempt,
            cycle=self.cycle,
            parent_dir=parent_dir,
            sibling_dirs=siblings,
            direction_guidance=self.direction_guidance,
        )

        # Resume the parent's workspace before the agent edits it.
        if parent_dir and os.path.isdir(parent_dir):
            src = os.path.join(parent_dir, self.task.eval_program)
            if os.path.exists(src):
                with open(src, "r", encoding="utf-8") as fh:
                    payload = fh.read()
                with open(os.path.join(node_dir, self.task.eval_program), "w", encoding="utf-8") as fh:
                    fh.write(payload)

        t0 = time.time()
        try:
            self.agent.propose(ctx)
            agent_error = None
        except Exception as exc:  # noqa: BLE001 - agent failures are data, not crashes
            agent_error = f"{type(exc).__name__}: {exc}"
        self.agent_calls += 1

        if agent_error is not None:
            result = EvalResult(evaluated=False, fail_class="env", error=agent_error)
        else:
            try:
                result = self.task.evaluate(node_dir)
            except Exception as exc:  # noqa: BLE001
                result = EvalResult(evaluated=True, fail_class="env",
                                    error=f"evaluator crashed: {type(exc).__name__}: {exc}")
        result.write(node_dir)
        elapsed = time.time() - t0

        node = Node(
            cell_id=cid,
            branch=meta.branch,
            attempt=meta.attempt,
            parent_id=meta.parent_id,
            seq=len(self.tree),
            score=result.score,
            evaluated=result.evaluated,
            valid=result.valid,
            fail_class=result.fail_class,
            error=result.error,
            n_valid=result.n_valid,
            n_total=result.n_total,
            cost=1.0,
            workspace=node_dir,
            artifact=os.path.join(node_dir, self.task.eval_program),
            proposal=os.path.join(node_dir, "proposal.md"),
            diagnostics={**result.diagnostics, "seconds": round(elapsed, 4)},
        )
        return node, result


def run_online_cycle(task: Task, agent: DiscoveryAgent, policy, plan: GridPlan, *,
                     run_dir: str, history_dir: str, baseline_dir: str, problem_file: str,
                     baseline_score: float, max_parallelism: int = 4,
                     max_decision_rounds: int = 64, budget: Optional[int] = None,
                     world_id: str = "live", direction_guidance: str = "",
                     max_workers: Optional[int] = None, cycle: int = 1) -> Dict[str, Any]:
    """Deploy ``policy`` online for one rollout and freeze the resulting trace.

    The rollout ends when the policy selects an empty batch or the decision-round
    cap ``K1`` is reached.  Returns the trace, the episode result and the live
    cycle manifest that later ``plan_grid`` calls read.
    """
    os.makedirs(run_dir, exist_ok=True)
    world = LiveWorld(task, agent, plan, run_dir=run_dir, history_dir=history_dir,
                      baseline_dir=baseline_dir, problem_file=problem_file,
                      baseline_score=baseline_score, max_parallelism=max_parallelism,
                      max_decision_rounds=max_decision_rounds, world_id=world_id,
                      direction_guidance=direction_guidance, max_workers=max_workers,
                      cycle=cycle)
    result = policy.solve(world, budget)
    tree = world.tree
    trace_path = os.path.join(run_dir, "trace.json")
    tree.save(trace_path)

    manifest = live_cycle_manifest(tree, result, plan, world, policy=policy)
    with open(os.path.join(run_dir, "live_cycle_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return {"tree": tree, "result": result, "manifest": manifest,
            "trace_path": trace_path, "world": world}


def live_cycle_manifest(tree: DiscoveryTree, result, plan: GridPlan, world: LiveWorld,
                        policy=None) -> Dict[str, Any]:
    """The prefix-safe sidecar ``plan_grid`` is allowed to read next round.

    It records what the grid *was*, what was actually opened, how the gain was
    distributed between opening roots and deepening branches, and whether the
    round plateaued -- never a raw per-cell outcome.
    """
    baseline = tree.baseline_score
    roots_gain, deep_gain = 0.0, 0.0
    for node in tree:
        parent = tree.get(node.parent_id)
        ref = parent.score if parent is not None else baseline
        gain = max(0.0, (node.score - ref)) if node.score != float("-inf") else 0.0
        if node.attempt == 0:
            roots_gain += max(0.0, node.score - baseline) if node.score != float("-inf") else 0.0
        else:
            deep_gain += gain
    total_gain = roots_gain + deep_gain
    best = tree.best_node()

    curve = [s for _, s in result.curve if s != float("-inf")]
    plateaued = len(curve) >= 4 and (curve[-1] - curve[len(curve) // 2]) <= 1e-9

    hard = sum(1 for n in tree
               if n.fail_class not in ("ok", "compile_other", "runtime", "correctness", "resource"))
    return {
        "world_id": tree.world_id,
        "task": tree.task,
        "planned_branch_count": plan.branch_count,
        "planned_refine_count": plan.refine_count,
        "plan_reason": plan.reason,
        "effective_branch_count": tree.branch_count,
        "effective_refine_count": tree.refine_count,
        "unopened_roots": max(0, plan.branch_count - tree.branch_count),
        "probes": result.total_probes,
        "decision_rounds": result.decision_rounds,
        "effective_sequential_rounds": result.effective_sequential_rounds,
        "agent_calls": world.agent_calls,
        "seconds": round(world.wall_clock, 3),
        "best_score": None if result.best_score == float("-inf") else result.best_score,
        "best_cell": best.cell_id if best else None,
        "baseline_score": baseline,
        "early_gain_share": (roots_gain / total_gain) if total_gain > 0 else 0.0,
        "late_gain_share": (deep_gain / total_gain) if total_gain > 0 else 0.0,
        "plateaued": bool(plateaued),
        "hard_failures": hard,
        "beta": _beta_of(policy),
        "stopped_reason": result.stopped_reason,
    }


def _beta_of(policy) -> Optional[float]:
    """The beta actually baked into the policy that produced this round."""
    beta = getattr(policy, "beta", None)
    if beta is None and hasattr(policy, "config"):
        beta = policy.config.get("beta")
    return None if beta is None else float(beta)
