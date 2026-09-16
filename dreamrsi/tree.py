"""Discovery trees: the artefact an online rollout produces and a replay world consumes.

A rollout incrementally builds a rooted tree that initially contains only the
task root ``r`` (the initial workspace).  Every non-root node ``v`` records the
single attempt that created it: the workspace snapshot, the generated artifact,
the evaluation diagnostics and the score ``s_v``.

Because ``CONTINUE`` is the only atomic operation and every node has exactly one
primary parent, the tree is a forest of chains hanging off the root.  We keep
the ``(branch, attempt)`` coordinates explicit so the same object can be served
to a policy as the irregular grid it expects.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .types import ROOT_ID, CellMeta, Observation, cell_id


@dataclass
class Node:
    """One completed generation-evaluation attempt."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: str
    seq: int
    score: float = float("-inf")
    evaluated: bool = False
    valid: bool = False
    fail_class: str = "ok"
    error: Optional[str] = None
    n_valid: Optional[int] = None
    n_total: Optional[int] = None
    cost: float = 1.0
    tags: Tuple[str, ...] = ()
    workspace: Optional[str] = None
    artifact: Optional[str] = None
    proposal: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        if d["score"] == float("-inf"):
            d["score"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Node":
        d = dict(d)
        d["tags"] = tuple(d.get("tags") or ())
        if d.get("score") is None:
            d["score"] = float("-inf")
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        d = {k: v for k, v in d.items() if k in known}
        node = cls(**d)
        if extra:
            node.diagnostics.setdefault("_extra", {}).update(extra)
        return node


class DiscoveryTree:
    """A completed (or in-progress) discovery trace.

    Frozen, it is exactly the replay simulator of Section 2: every stored
    outcome can be revealed again at zero execution cost.
    """

    def __init__(
        self,
        world_id: str = "trace",
        baseline_score: float = 0.0,
        task: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.world_id = world_id
        self.baseline_score = float(baseline_score)
        self.task = task
        self.meta: Dict[str, Any] = dict(meta or {})
        self._nodes: Dict[str, Node] = {}
        self._by_branch: Dict[int, List[Node]] = {}
        self._seq = 0

    # -- construction --------------------------------------------------------
    def add(self, node: Node) -> Node:
        if node.cell_id in self._nodes:
            raise ValueError(f"duplicate cell {node.cell_id}")
        if node.attempt == 0:
            if node.parent_id != ROOT_ID:
                raise ValueError("a root attempt must hang off the task root")
        else:
            expected = cell_id(node.branch, node.attempt - 1)
            if node.parent_id != expected:
                raise ValueError(
                    f"{node.cell_id}: primary parent must be {expected}, got {node.parent_id}"
                )
            if expected not in self._nodes:
                raise ValueError(f"{node.cell_id} refines a node that does not exist yet")
        if node.seq < 0:
            node.seq = self._seq
        self._seq = max(self._seq, node.seq) + 1
        self._nodes[node.cell_id] = node
        self._by_branch.setdefault(node.branch, []).append(node)
        self._by_branch[node.branch].sort(key=lambda n: n.attempt)
        return node

    def continue_from(
        self,
        parent_id: str,
        *,
        score: float = float("-inf"),
        branch: Optional[int] = None,
        **kwargs: Any,
    ) -> Node:
        """Apply ``CONTINUE(parent_id)`` and attach the resulting child."""
        if parent_id == ROOT_ID:
            if branch is None:
                branch = (max(self._by_branch) + 1) if self._by_branch else 0
            attempt = 0
        else:
            parent = self._nodes[parent_id]
            if self.children_of(parent_id):
                raise ValueError(
                    f"{parent_id} already has a child; only leaves and the root are eligible"
                )
            branch, attempt = parent.branch, parent.attempt + 1
        node = Node(
            cell_id=cell_id(branch, attempt),
            branch=branch,
            attempt=attempt,
            parent_id=parent_id,
            seq=self._seq,
            score=score,
            **kwargs,
        )
        return self.add(node)

    # -- queries -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, cid: object) -> bool:
        return cid in self._nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(sorted(self._nodes.values(), key=lambda n: n.seq))

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes

    def get(self, cid: str) -> Optional[Node]:
        return self._nodes.get(cid)

    def children_of(self, cid: str) -> List[Node]:
        if cid == ROOT_ID:
            return [b[0] for b in self._by_branch.values() if b]
        node = self._nodes.get(cid)
        if node is None:
            return []
        child = self._nodes.get(cell_id(node.branch, node.attempt + 1))
        return [child] if child is not None else []

    def branch_nodes(self, branch: int) -> List[Node]:
        return list(self._by_branch.get(branch, ()))

    def branch_depth(self, branch: int) -> int:
        """Number of recorded attempts in ``branch`` (0 if the branch is absent)."""
        return len(self._by_branch.get(branch, ()))

    @property
    def branches(self) -> List[int]:
        return sorted(self._by_branch)

    @property
    def branch_count(self) -> int:
        return len(self._by_branch)

    @property
    def refine_count(self) -> int:
        """Deepest realized refinement index, i.e. ``R`` of the realized grid."""
        return max((self.branch_depth(b) - 1 for b in self._by_branch), default=0)

    @property
    def ceiling(self) -> float:
        """Best score anywhere in the frozen trace -- the attainment target."""
        best = float("-inf")
        for n in self._nodes.values():
            if n.score > best:
                best = n.score
        return best

    def best_node(self) -> Optional[Node]:
        best, out = float("-inf"), None
        for n in self._nodes.values():
            if n.score > best:
                best, out = n.score, n
        return out

    def path_to(self, cid: str) -> List[Node]:
        node = self._nodes[cid]
        return [self._nodes[cell_id(node.branch, a)] for a in range(node.attempt + 1)]

    def is_leaf(self, cid: str) -> bool:
        return not self.children_of(cid)

    def eligible_continuations(self) -> List[str]:
        """``A(T) = {r} u {v in T : v is a leaf}`` from Section 3."""
        return [ROOT_ID] + [n.cell_id for n in self._nodes.values() if self.is_leaf(n.cell_id)]

    # -- policy-facing views -------------------------------------------------
    def meta_of(self, cid: str) -> CellMeta:
        node = self._nodes[cid]
        return CellMeta(
            cell_id=node.cell_id,
            branch=node.branch,
            attempt=node.attempt,
            parent_id=node.parent_id,
            seq=node.seq,
            tags=tuple(node.tags),
        )

    def observation_of(self, cid: str) -> Observation:
        node = self._nodes[cid]
        parent = self._nodes.get(node.parent_id)
        parent_score = parent.score if parent is not None else self.baseline_score
        return Observation(
            cell_id=node.cell_id,
            branch=node.branch,
            attempt=node.attempt,
            score=node.score,
            evaluated=node.evaluated,
            valid=node.valid,
            fail_class=node.fail_class,
            error=node.error,
            delta_vs_baseline=_delta(node.score, self.baseline_score),
            delta_vs_parent=_delta(node.score, parent_score),
            n_valid=node.n_valid,
            n_total=node.n_total,
            seq=node.seq,
            tags=tuple(node.tags),
            cost=node.cost,
        )

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "world_id": self.world_id,
            "task": self.task,
            "baseline_score": self.baseline_score,
            "meta": self.meta,
            "nodes": [n.to_dict() for n in self],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscoveryTree":
        tree = cls(
            world_id=d.get("world_id", "trace"),
            baseline_score=d.get("baseline_score", 0.0),
            task=d.get("task", ""),
            meta=d.get("meta") or {},
        )
        for nd in sorted(d.get("nodes", []), key=lambda n: (n.get("branch", 0), n.get("attempt", 0))):
            tree.add(Node.from_dict(nd))
        return tree

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=False)
            fh.write("\n")
        return path

    @classmethod
    def load(cls, path: str) -> "DiscoveryTree":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def summary(self) -> Dict[str, Any]:
        best = self.best_node()
        succ = sum(1 for n in self._nodes.values() if n.evaluated and n.error is None and n.fail_class == "ok")
        return {
            "world_id": self.world_id,
            "task": self.task,
            "nodes": len(self._nodes),
            "branch_count": self.branch_count,
            "refine_count": self.refine_count,
            "successful": succ,
            "baseline_score": self.baseline_score,
            "ceiling": None if self.ceiling == float("-inf") else self.ceiling,
            "best_cell": best.cell_id if best else None,
        }


def _delta(score: float, reference: float) -> float:
    if score == float("-inf") or reference == float("-inf"):
        return 0.0
    return float(score - reference)


def load_pool(paths: Iterable[str]) -> List[DiscoveryTree]:
    """Load a simulator pool from a list of trace files or directories."""
    out: List[DiscoveryTree] = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.endswith(".json"):
                    out.append(DiscoveryTree.load(os.path.join(p, name)))
        else:
            out.append(DiscoveryTree.load(p))
    return out
