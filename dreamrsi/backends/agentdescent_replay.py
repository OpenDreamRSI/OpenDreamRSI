"""Stage 2 on ``agentdescent``: its world, replayed under this repository's rules.

The world being replayed here is an ``agentdescent.dream.DiscoveryTree`` -- the
object ``explore()`` produced and ``SimulatorPool`` holds -- and the walk over it
uses that tree's own structure queries and the same revealed-cursor discipline
``ReplayWorld.replay`` uses. What does *not* move is the rule set, because the
two packages read three parts of Appendix B.2 differently and this repository's
reading is its published semantics:

**Barren picks.** ``agentdescent``'s ``A(T) = {root} u leaves`` includes a leaf
whose branch ended in the record; picking it reveals nothing and costs a worker,
"which is how a replay charges a policy for a worker it would have wasted
online". :class:`dreamrsi.world.ReplayWorld` never offers that move --
``frontiers()`` only names a cell the trace can serve. Both are defensible; they
score a policy differently, so they cannot both be in effect.

**Illegal batches.** ``agentdescent`` counts them and lets the reward carry the
message, on the grounds that the policy is a model's rewrite. This repository
raises :class:`~dreamrsi.world.IllegalBatch`, so that "a policy that cheats fails
loudly during dreaming" rather than quietly scoring a little worse.

**Naming a branch.** ``agentdescent`` gives the root a multiplicity of ``W`` and
reveals ``children(root)[taken]`` positionally, so a batch of roots opens
branches in recorded order. Here a policy names the branch it wants -- ``b2a0``
reveals *that* cell -- which is what makes cell ids meaningful to a policy and
what the replay-fidelity evidence rests on.

So this module is the substitution the constraint allows: the world, its
storage, and the structural queries are ``agentdescent``'s; the legal set, the
failure behaviour, the addressing and the cost accounting are this repository's,
inherited from :class:`~dreamrsi.world.BaseWorld` rather than restated.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..types import ROOT_ID, CellMeta, GridPlanningContext, Observation, cell_id
from ..world import BaseWorld
from .agentdescent_tree import branch_roots, cell_of, check_shape, index_of


class _TreeView:
    """Just enough of a :class:`dreamrsi.tree.DiscoveryTree` for the callers.

    ``evaluate_policy`` asks a world for ``len(world.tree)`` to size the replay
    horizon. That is the count of *attempts*, which is one less than the
    ``agentdescent`` node count because node 0 is the root.
    """

    def __init__(self, adt) -> None:
        self.adt = adt

    def __len__(self) -> int:
        return max(0, len(self.adt) - 1)


class AgentDescentReplayWorld(BaseWorld):
    """An ``agentdescent`` world, served under this repository's replay rules."""

    def __init__(self, adt, *, max_parallelism: int = 4,
                 max_decision_rounds: int = 64, task: str = "") -> None:
        check_shape(adt)
        root = adt.nodes[0]
        super().__init__(max_parallelism=max_parallelism,
                         max_decision_rounds=max_decision_rounds,
                         baseline_score=0.0 if root.score is None else float(root.score),
                         world_id=adt.name)
        self.adt = adt
        self.task = task or str((root.detail or {}).get("task", ""))
        self.tree = _TreeView(adt)
        self._roots = branch_roots(adt)
        #: node index -> how many of its recorded children are revealed. The
        #: same cursor ``ReplayWorld.replay`` keeps, so "what comes next on this
        #: branch" is answered by the recorded order rather than recomputed.
        self._cursor: Dict[int, int] = {}

    # -- structure ----------------------------------------------------------
    @property
    def branch_count(self) -> int:
        return len(self._roots)

    @property
    def refine_count(self) -> int:
        return max((self.adt.depth(n.index) - 1 for n in self.adt.nodes[1:]), default=0)

    @property
    def ceiling(self) -> float:
        """Best score among the *attempts* -- the root carries the baseline."""
        scores = [n.score for n in self.adt.nodes[1:] if n.score is not None]
        return max(scores) if scores else float("-inf")

    def reset(self) -> None:
        super().reset()
        self._cursor = {}

    # -- the legal set (this repository's, not agentdescent's) ---------------
    def legal_roots(self) -> List[str]:
        """Unopened branches, named -- never a branch the record does not have."""
        taken = self._cursor.get(0, 0)
        opened = set(self.opened_branches())
        return [cell_of(self.adt, index) for position, index in enumerate(self._roots)
                if position not in opened and taken <= len(self._roots)]

    def frontiers(self) -> List[str]:
        """The next *recorded* attempt of each opened branch, and only that."""
        out: List[str] = []
        for branch in self.opened_branches():
            depth = max((o.attempt for o in self._observed.values()
                         if o.branch == branch), default=-1)
            nxt = cell_id(branch, depth + 1)
            try:
                index_of(self.adt, nxt)
            except KeyError:
                continue          # the branch ended here: not offered, not barren
            out.append(nxt)
        return out

    def legal_actions(self) -> List[str]:
        return self.legal_roots() + self.frontiers()

    def meta(self, cid: str) -> CellMeta:
        index = index_of(self.adt, cid)
        branch, attempt = _coords(cid)
        return CellMeta(cell_id=cid, branch=branch, attempt=attempt,
                        parent_id=ROOT_ID if attempt == 0 else cell_id(branch, attempt - 1),
                        seq=int((self.adt.nodes[index].detail or {}).get("seq", index - 1)),
                        tags=_tags(self.adt.nodes[index].detail))

    # -- transition ----------------------------------------------------------
    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], Any]] = None
                    ) -> List[Observation]:
        batch = self._validate(cells)
        if not batch:
            return []
        prefix_size = len(self._observed)
        self._charge(batch)
        out: List[Observation] = []
        for cid in batch:
            index = index_of(self.adt, cid)
            parent = self.adt.nodes[index].parent
            self._cursor[parent] = self._cursor.get(parent, 0) + 1
            obs = self._observation(index, cid)
            self._record(obs)
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        self._log_round(batch, out, prefix_size)
        return out

    def _observation(self, index: int, cid: str) -> Observation:
        node = self.adt.nodes[index]
        detail = dict(node.detail or {})
        branch, attempt = _coords(cid)
        score = float("-inf") if node.score is None else float(node.score)
        parent = self.adt.nodes[node.parent] if node.parent is not None else None
        parent_score = (self.baseline_score if parent is None or parent.index == 0
                        else (float("-inf") if parent.score is None else float(parent.score)))
        return Observation(
            cell_id=cid, branch=branch, attempt=attempt, score=score,
            evaluated=bool(detail.get("evaluated", node.score is not None)),
            valid=bool(node.valid),
            fail_class=str(detail.get("fail_class", "ok")),
            error=(detail.get("error") or None),
            delta_vs_baseline=_delta(score, self.baseline_score),
            delta_vs_parent=_delta(score, parent_score),
            n_valid=detail.get("n_valid"), n_total=detail.get("n_total"),
            seq=int(detail.get("seq", index - 1)), tags=_tags(detail),
            cost=float(detail.get("cost", 1.0)),
        )

    # -- planning ------------------------------------------------------------
    def planning_context(self, history: Iterable[Dict[str, Any]] = (),
                         **caps: Any) -> GridPlanningContext:
        return GridPlanningContext(
            history=tuple(history),
            fallback_branch_count=caps.get("fallback_branch_count",
                                           min(4, self.branch_count or 1)),
            fallback_refine_count=caps.get("fallback_refine_count",
                                           min(4, self.refine_count)),
            hard_max_branch_count=caps.get("hard_max_branch_count",
                                           max(1, self.branch_count)),
            hard_max_refine_count=caps.get("hard_max_refine_count",
                                           max(0, self.refine_count)),
            worker_cap=self.max_parallelism,
            trace_branch_count=self.branch_count,
            trace_refine_count=self.refine_count,
            mode="replay",
        )


class AgentDescentSimulatorPool:
    """``H_t``, holding ``agentdescent`` worlds.

    A drop-in for :class:`dreamrsi.world.SimulatorPool`: same constructor shape,
    same ``worlds()``, so :func:`dreamrsi.dreaming.evaluate_policy` and
    :func:`~dreamrsi.dreaming.dream` do not know which pool they were handed.
    """

    def __init__(self, trees: Optional[Iterable[Any]] = None, *,
                 max_parallelism: int = 4, max_decision_rounds: int = 64,
                 task: str = "") -> None:
        self.trees: List[Any] = list(trees or [])
        self.max_parallelism = int(max_parallelism)
        self.max_decision_rounds = int(max_decision_rounds)
        self.task = task

    def __len__(self) -> int:
        return len(self.trees)

    def __iter__(self):
        return iter(self.worlds())

    def append(self, tree: Any) -> None:
        self.trees.append(tree)

    def worlds(self) -> List[AgentDescentReplayWorld]:
        return [AgentDescentReplayWorld(t, max_parallelism=self.max_parallelism,
                                        max_decision_rounds=self.max_decision_rounds,
                                        task=self.task)
                for t in self.trees]

    def summary(self) -> List[Dict[str, Any]]:
        """The same rows :meth:`dreamrsi.world.SimulatorPool.summary` produces.

        Both the tie-break and the success criterion are copied deliberately.
        ``DiscoveryTree.best_node`` scans in node order with a strict ``>``, so
        the *first* node holding the best score wins; taking a ``max`` over
        ``(score, index)`` instead returns the *last*, which is invisible until a
        world has ties -- and a real one does, because several attempts failing
        to beat the baseline all land on exactly the same score.
        """
        out = []
        for world in self.worlds():
            best_score, best_index = float("-inf"), None
            for node in world.adt.nodes[1:]:
                if node.score is not None and float(node.score) > best_score:
                    best_score, best_index = float(node.score), node.index
            out.append({
                "world_id": world.world_id, "task": world.task,
                "nodes": len(world.tree), "branch_count": world.branch_count,
                "refine_count": world.refine_count,
                "successful": sum(1 for n in world.adt.nodes[1:]
                                  if _successful(n)),
                "baseline_score": world.baseline_score,
                "ceiling": None if best_index is None else best_score,
                "best_cell": (None if best_index is None
                              else cell_of(world.adt, best_index)),
            })
        return out


def _successful(node) -> bool:
    """``evaluated and error is None and fail_class == "ok"`` -- the tree's rule.

    Not "has a score": the failure taxonomy distinguishes an attempt that was
    measured and is bad from one that broke, and both can carry a number.
    """
    detail = node.detail or {}
    return (bool(detail.get("evaluated", node.score is not None))
            and not detail.get("error")
            and str(detail.get("fail_class", "ok")) == "ok")


def _coords(cid: str):
    b, _, a = cid[1:].partition("a")
    return int(b), int(a)


def _tags(detail: Optional[Dict[str, Any]]):
    return tuple(t for t in str((detail or {}).get("tags", "")).split(",") if t)


def _delta(score: float, reference: float) -> float:
    if score == float("-inf") or reference == float("-inf"):
        return 0.0
    return float(score - reference)


__all__ = ["AgentDescentReplayWorld", "AgentDescentSimulatorPool"]
