"""Replay worlds: a completed discovery history served back as a simulator.

Section 2 of the paper: "a completed discovery process inherently records a
structured tree of past exploration decisions and their realized code-execution
outcomes".  Replaying a different exploration policy over that tree means
traversing a different subset of branches, in a different order, with different
parallel groupings and stopping decisions -- and every outcome is already
stored, so a replay costs no generation and no evaluation.

``ReplayWorld`` is the object the policy prompt calls ``question``.  The same
interface is implemented by :class:`dreamrsi.online.LiveWorld`, where a
continuation instead *creates* a new outcome by calling the discovery agent.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .tree import DiscoveryTree
from .types import (
    CellMeta,
    GridPlanningContext,
    Observation,
    ceil_div,
    cell_id,
)


class IllegalBatch(ValueError):
    """Raised when a policy asks for a batch the environment cannot serve."""


class BaseWorld:
    """Shared batch accounting for the live and replay phases.

    Both phases use the same tree-based decision interface; they differ only in
    how a continuation produces its next observation.
    """

    def __init__(self, *, max_parallelism: int = 4, max_decision_rounds: int = 64,
                 baseline_score: float = 0.0, world_id: str = "world") -> None:
        self.world_id = world_id
        self.max_parallelism = int(max_parallelism)
        self.max_decision_rounds = int(max_decision_rounds)
        self.baseline_score = float(baseline_score)
        self._observed: Dict[str, Observation] = {}
        self.total_probes = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0
        self.batches: List[List[str]] = []
        #: Per-decision-round diagnostics, archived as policy_execution_traces.jsonl.
        self.round_log: List[Dict[str, Any]] = []
        self.best_so_far: float = float("-inf")
        self.best_cell: Optional[str] = None

    # -- bookkeeping (explicitly *not* for decision-making) -------------------
    @property
    def budget_spent(self) -> int:
        return self.total_probes

    def reset(self) -> None:
        self._observed.clear()
        self.total_probes = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0
        self.batches = []
        self.round_log = []
        self.best_so_far = float("-inf")
        self.best_cell = None

    # -- revealed prefix -----------------------------------------------------
    def observed(self) -> Dict[str, Observation]:
        """The revealed prefix only.  Unrevealed scores stay unknown."""
        return dict(self._observed)

    def opened_branches(self) -> List[int]:
        return sorted({o.branch for o in self._observed.values()})

    def _record(self, obs: Observation) -> None:
        self._observed[obs.cell_id] = obs
        if obs.score > self.best_so_far:
            self.best_so_far = obs.score
            self.best_cell = obs.cell_id

    def _charge(self, batch: Sequence[str]) -> None:
        """One decision round plus ``ceil(k / W)`` effective sequential rounds."""
        self.total_probes += len(batch)
        self.decision_rounds += 1
        self.effective_sequential_rounds += ceil_div(len(batch), self.max_parallelism)
        self.batches.append(list(batch))

    def _log_round(self, batch: Sequence[str], revealed: Sequence[Observation],
                   prefix_size: int) -> None:
        self.round_log.append({
            "round": self.decision_rounds,
            "prefix_size": prefix_size,
            "batch": list(batch),
            "batch_size": len(batch),
            "effective_sequential_rounds": ceil_div(len(batch), self.max_parallelism),
            "revealed": [
                {"cell": o.cell_id, "branch": o.branch, "attempt": o.attempt,
                 "score": (None if o.score == float("-inf") else o.score),
                 "fail_class": o.fail_class, "success": o.is_success,
                 "delta_vs_parent": o.delta_vs_parent}
                for o in revealed
            ],
            "best_so_far": (None if self.best_so_far == float("-inf") else self.best_so_far),
            "total_probes": self.total_probes,
        })

    def _validate(self, cells: Sequence[str]) -> List[str]:
        batch = list(cells)
        if not batch:
            return batch
        if len(set(batch)) != len(batch):
            raise IllegalBatch(f"batch contains duplicate cells: {batch}")
        if len(batch) > self.max_parallelism:
            raise IllegalBatch(
                f"batch of {len(batch)} exceeds max_parallelism={self.max_parallelism}"
            )
        legal = set(self.legal_actions())
        bad = [c for c in batch if c not in legal]
        if bad:
            raise IllegalBatch(f"illegal cells (not legal before the call): {bad}")
        # "It must never contain a parent and its child together."
        inbatch = set(batch)
        for c in batch:
            b, a = _coords(c)
            if a > 0 and cell_id(b, a - 1) in inbatch:
                raise IllegalBatch(f"batch contains both {cell_id(b, a - 1)} and its child {c}")
        return batch

    # -- interface implemented by subclasses ---------------------------------
    def legal_actions(self) -> List[str]:
        raise NotImplementedError

    def legal_roots(self) -> List[str]:
        raise NotImplementedError

    def meta(self, cid: str) -> CellMeta:
        raise NotImplementedError

    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], Any]] = None) -> List[Observation]:
        raise NotImplementedError


class ReplayWorld(BaseWorld):
    """A frozen discovery tree exposed as a zero-execution-cost simulator.

    Replay starts from ``T^{m,0} = {r}``.  At decision round ``k`` the policy
    selects a batch of eligible continuations; the transition is deterministic
    -- revealing ``Child(v; T)`` from the recorded tree.  If a selected node has
    no recorded child in the frozen trace, the revealed set is empty for it and
    the branch is exhausted.
    """

    def __init__(self, tree: DiscoveryTree, *, max_parallelism: int = 4,
                 max_decision_rounds: int = 64) -> None:
        super().__init__(
            max_parallelism=max_parallelism,
            max_decision_rounds=max_decision_rounds,
            baseline_score=tree.baseline_score,
            world_id=tree.world_id,
        )
        self.tree = tree

    # -- structure -----------------------------------------------------------
    @property
    def branch_count(self) -> int:
        return self.tree.branch_count

    @property
    def refine_count(self) -> int:
        return self.tree.refine_count

    @property
    def ceiling(self) -> float:
        return self.tree.ceiling

    def legal_roots(self) -> List[str]:
        """Unopened roots only."""
        opened = set(self.opened_branches())
        return [cell_id(b, 0) for b in self.tree.branches if b not in opened]

    def frontiers(self) -> List[str]:
        """The next unrevealed attempt of each already-opened branch."""
        out: List[str] = []
        for b in self.opened_branches():
            depth = max(
                (o.attempt for o in self._observed.values() if o.branch == b), default=-1
            )
            nxt = cell_id(b, depth + 1)
            if nxt in self.tree:
                out.append(nxt)
        return out

    def legal_actions(self) -> List[str]:
        """Roots plus opened-branch frontiers."""
        return self.legal_roots() + self.frontiers()

    def meta(self, cid: str) -> CellMeta:
        return self.tree.meta_of(cid)

    # -- transition ----------------------------------------------------------
    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], Any]] = None) -> List[Observation]:
        batch = self._validate(cells)
        if not batch:
            return []
        prefix_size = len(self._observed)
        self._charge(batch)
        out: List[Observation] = []
        for cid in batch:
            obs = self.tree.observation_of(cid)
            self._record(obs)
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        self._log_round(batch, out, prefix_size)
        return out

    # -- planning context ----------------------------------------------------
    def planning_context(self, history: Iterable[Dict[str, Any]] = (), **caps: Any) -> GridPlanningContext:
        return GridPlanningContext(
            history=tuple(history),
            fallback_branch_count=caps.get("fallback_branch_count", min(4, self.branch_count or 1)),
            fallback_refine_count=caps.get("fallback_refine_count", min(4, self.refine_count)),
            hard_max_branch_count=caps.get("hard_max_branch_count", max(1, self.branch_count)),
            hard_max_refine_count=caps.get("hard_max_refine_count", max(0, self.refine_count)),
            worker_cap=self.max_parallelism,
            trace_branch_count=self.branch_count,
            trace_refine_count=self.refine_count,
            mode="replay",
        )


class SimulatorPool:
    """The growing pool of replay worlds built from accumulated history.

    ``H_t = (T_1, ..., T_t)``: every online rollout appends one frozen tree, so
    the pool -- and therefore the fidelity of dreaming -- expands with every
    outer iteration.
    """

    def __init__(self, trees: Optional[Iterable[DiscoveryTree]] = None,
                 *, max_parallelism: int = 4, max_decision_rounds: int = 64) -> None:
        self.trees: List[DiscoveryTree] = list(trees or [])
        self.max_parallelism = int(max_parallelism)
        self.max_decision_rounds = int(max_decision_rounds)

    def __len__(self) -> int:
        return len(self.trees)

    def __iter__(self):
        return iter(self.worlds())

    def append(self, tree: DiscoveryTree) -> None:
        self.trees.append(tree)

    def worlds(self) -> List[ReplayWorld]:
        return [
            ReplayWorld(t, max_parallelism=self.max_parallelism,
                        max_decision_rounds=self.max_decision_rounds)
            for t in self.trees
        ]

    def summary(self) -> List[Dict[str, Any]]:
        return [t.summary() for t in self.trees]


def _coords(cid: str):
    b, _, a = cid[1:].partition("a")
    return int(b), int(a)
