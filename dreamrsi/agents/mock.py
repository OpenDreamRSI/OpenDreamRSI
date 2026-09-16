"""Deterministic stand-ins for the two LLM agents.

They make the whole recursive loop runnable -- and testable -- with no API key,
no CLI and no network.  The mock discovery agent is a weak seeded hill-climber
on the synthetic task: good enough that depth pays off, noisy enough that some
branches are traps and some attempts fail in a repairable way.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional

from ..policy import library
from .base import AttemptContext, DiscoveryAgent, PolicyDeveloper, RevisionRequest, write_proposal


def _seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


class MockDiscoveryAgent(DiscoveryAgent):
    """Seeded hill-climber over the synthetic task's search space.

    A root attempt samples a fresh basin (the branch's "direction"); a
    refinement resumes the parent's workspace and takes a decaying local step,
    keeping the best of a few internal candidates.  A seeded minority of
    attempts fail the way real coding-agent attempts fail: an out-of-range
    coordinate or a malformed program.
    """

    name = "mock-discovery-agent"

    def __init__(self, task, *, seed: int = 0, failure_rate: float = 0.12,
                 internal_candidates: int = 3) -> None:
        self.task = task
        self.seed = int(seed)
        self.failure_rate = float(failure_rate)
        self.internal_candidates = int(internal_candidates)
        self._history_cache: Dict[str, tuple] = {}

    # -- "read the complete history first" ----------------------------------
    def _history_best(self, ctx: AttemptContext) -> Optional[List[float]]:
        """Best scoring candidate recorded in earlier rounds of this run.

        Only *completed* rounds are read: the current round's directory is
        excluded, so an attempt never sees a sibling that is still running.
        """
        key = (ctx.history_dir, ctx.cycle)
        cached = self._history_cache.get(key)
        if cached is not None:
            return cached[0]

        best_x, best_score = None, float("-inf")
        for round_dir in sorted(_subdirs(ctx.history_dir)):
            if os.path.abspath(round_dir) == os.path.abspath(ctx.run_dir):
                continue
            attempts = os.path.join(round_dir, "attempts")
            for node_dir in sorted(_subdirs(attempts)):
                score = _read_score(os.path.join(node_dir, "eval", "score.json"))
                if score is None or score <= best_score:
                    continue
                x = _read_x(os.path.join(node_dir, ctx.eval_program))
                if x is not None:
                    best_x, best_score = x, score
        self._history_cache[key] = (best_x, best_score)
        return best_x

    def propose(self, ctx: AttemptContext) -> None:
        rng = random.Random(_seed(self.seed, ctx.cycle, ctx.branch, ctx.attempt))
        bound = 5.12
        dim = getattr(self.task, "dim", 4)

        parent_x: Optional[List[float]] = None
        if ctx.parent_dir:
            parent_x = _read_x(os.path.join(ctx.parent_dir, ctx.eval_program))

        if parent_x is None:
            # Open a new direction.  Following the exploration prompt, the agent
            # reads the complete history first: half of the new roots re-enter
            # the neighbourhood of the best historical candidate, the rest probe
            # genuinely fresh regions so the search does not collapse.
            brng = random.Random(_seed(self.seed, ctx.cycle, "root", ctx.branch))
            best = self._history_best(ctx)
            if best is not None and ctx.branch % 2 == 1:
                radius = bound * 0.5
                x = [max(-bound, min(bound, v + brng.gauss(0.0, radius))) for v in best]
                mechanism = (f"new direction {ctx.branch}: re-entry near the best historical "
                             f"candidate (radius {radius:.2f})")
            else:
                x = [brng.uniform(-bound, bound) for _ in range(dim)]
                mechanism = f"new direction {ctx.branch}: fresh basin, uncorrelated with history"
        else:
            step = bound * 0.45 * (0.62 ** ctx.attempt)
            best, best_obj = None, float("inf")
            for _ in range(self.internal_candidates):
                cand = [max(-bound, min(bound, v + rng.gauss(0.0, step))) for v in parent_x]
                obj = self.task.objective(cand)
                if obj < best_obj:
                    best, best_obj = cand, obj
            x = best if best is not None else list(parent_x)
            mechanism = (f"local refinement of branch {ctx.branch} at depth {ctx.attempt} "
                         f"(step sigma={step:.3f}, {self.internal_candidates} internal candidates)")

        failure = rng.random() < self.failure_rate
        if failure:
            mode = rng.choice(["out_of_range", "malformed"])
            if mode == "out_of_range":
                x = list(x)
                x[rng.randrange(dim)] = bound * 2.5
                mechanism += " -- BUG: an unclamped coordinate escaped the box"
            else:
                _write_raw(os.path.join(ctx.node_dir, ctx.eval_program), '{"x": [broken,')
                write_proposal(ctx.node_dir, _proposal_text(ctx, mechanism + " -- BUG: truncated program"))
                return

        _write_raw(os.path.join(ctx.node_dir, ctx.eval_program),
                   json.dumps({"x": x, "mechanism": mechanism}, indent=2) + "\n")
        write_proposal(ctx.node_dir, _proposal_text(ctx, mechanism))


def _proposal_text(ctx: AttemptContext, mechanism: str) -> str:
    return (
        f"# Attempt b{ctx.branch}a{ctx.attempt}\n\n"
        f"**Mechanism.** {mechanism}\n\n"
        f"**Evidence from history.** resumed `{ctx.parent_dir or ctx.baseline_dir}`; "
        f"{len(ctx.sibling_dirs)} sibling attempts were read before proposing.\n\n"
        "**Why it is not a repeat.** the step is drawn from this node's own seed, so no two "
        "attempts in the tree share a candidate.\n\n"
        "**Expected benefit / risk.** benefit: a local improvement inside the branch's basin; "
        "risk: the basin may be a trap, in which case the trajectory will flatten.\n"
    )


class MockPolicyDeveloper(PolicyDeveloper):
    """Proposes policy versions from a fixed library of mechanisms.

    Revision ``m`` picks the ``m``-th candidate source and bakes in a default
    beta chosen by the cross-cycle rule from the feedback it is handed.  This
    stands in for an LLM rewriting ``method.py`` and lets the recursive loop be
    exercised end-to-end deterministically.
    """

    name = "mock-policy-developer"

    def __init__(self, sources: Optional[List[str]] = None,
                 betas: Optional[List[float]] = None) -> None:
        self.sources = list(sources or library.available())
        self.betas = list(betas or [0.62, 0.45, 0.8])

    def revise(self, request: RevisionRequest) -> str:
        name = self.sources[request.revision_index % len(self.sources)]
        source = library.source_of(name)
        beta = self._next_beta(request)
        return _rebake_beta(source, beta, request.revision_index)

    # -- "Beta: fixed per run, adaptive across cycles" -----------------------
    def _next_beta(self, request: RevisionRequest) -> float:
        fb = request.feedback or {}
        current = float(fb.get("current_beta", 0.6))
        live = fb.get("live_trend", [])
        sweep = fb.get("per_beta", {})

        improving = len(live) >= 2 and live[-1] > live[-2] + 1e-12
        if improving and sweep:
            best = max(sweep.items(), key=lambda kv: kv[1].get("auc", 0.0))[0]
            return float(best)
        if sweep:
            hi = {float(k): v for k, v in sweep.items()}
            top = max(hi.items(), key=lambda kv: kv[1].get("attainment", 0.0))
            if top[0] > current and top[1].get("attainment", 0.0) > 1e-9:
                return round(min(1.0, current + 0.15), 3)
            if top[0] < current:
                return round(max(0.0, current - 0.15), 3)
            return current
        idx = request.revision_index % len(self.betas)
        return self.betas[idx]


def _rebake_beta(source: str, beta: float, revision: int) -> str:
    import re

    patched, n = re.subn(
        r'self\.config\.get\("beta",\s*[0-9.]+\)',
        f'self.config.get("beta", {beta})',
        source,
        count=1,
    )
    if n == 0:  # pragma: no cover - library sources always carry the knob
        patched = source
    header = (f"# revision {revision}: default beta baked in at {beta} "
              f"using the cross-cycle default-beta rule.\n")
    return header + patched


def _subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, d) for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))]


def _read_score(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    score = payload.get("score")
    return None if score is None else float(score)


def _read_x(path: str) -> Optional[List[float]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    x = payload.get("x")
    if isinstance(x, list) and all(isinstance(v, (int, float)) for v in x):
        return [float(v) for v in x]
    return None


def _write_raw(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
