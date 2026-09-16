"""Making the two discovery-tree models interchangeable.

Both packages record the same object -- a rooted tree of ``CONTINUE`` attempts --
and disagree on three things, none of them deep:

===================  ==============================  =============================
                     :mod:`dreamrsi.tree`            ``agentdescent.dream``
===================  ==============================  =============================
addressing           ``b{branch}a{attempt}``         an integer index
the root             implicit; ``baseline_score``    node ``0``, carrying the
                     is a field of the tree          baseline as its score
no score             ``float("-inf")``               ``None``
===================  ==============================  =============================

The addressing difference is a bijection rather than a loss, because a
paper-shaped tree carries the coordinates structurally: a node's branch is the
root-child that heads it, and its attempt is its depth below that child.  This
module computes it in both directions so a world recorded by one engine can be
replayed by the other, which is what lets the existing test suite stand as the
specification for both.

The root is the one place where care is needed.  ``dreamrsi`` has no node for it
-- ``b0a0``'s parent is the sentinel string ``"root"`` and the baseline lives on
the tree -- while ``agentdescent`` gives it index ``0`` and a real score.  So the
conversions shift by one node in each direction rather than mapping node-for-node,
and the baseline travels as the root's score rather than being dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..tree import DiscoveryTree, Node
from ..types import ROOT_ID, cell_id, parse_cell_id

#: ``detail`` keys that carry a ``dreamrsi`` node's own fields through a world.
#: Everything here survives JSON, because a world has to.
_CARRIED = ("evaluated", "valid", "fail_class", "error", "n_valid", "n_total",
            "cost", "workspace", "artifact", "proposal", "seq")


class ShapeError(ValueError):
    """A tree cannot be addressed as a ``branch x attempt`` grid.

    Raised for a tree whose interior nodes have several recorded children. The
    paper's online rule continues a leaf at most once, so that shape does not
    arise from :func:`dreamrsi.online.run_online_cycle` or from
    ``agentdescent.dream.explore`` -- but it does arise from a search that
    re-expands interior nodes, and such a tree has no ``b{branch}a{attempt}``
    name for its second child.  ``agentdescent`` tolerates it (those nodes are
    simply unreachable during replay, and it counts them); ``dreamrsi`` cannot
    name them at all, so converting says so instead of silently dropping them.
    """


# -- coordinates -------------------------------------------------------------
def branch_roots(adt) -> List[int]:
    """The root's children, in creation order: branch ``b`` is ``[b]``."""
    return list(adt.children(0))


def cell_of(adt, index: int) -> str:
    """The ``b{branch}a{attempt}`` name of ``agentdescent`` node ``index``."""
    if index == 0:
        return ROOT_ID
    roots = branch_roots(adt)
    head = adt.branch(index)
    if head not in roots:  # pragma: no cover - defensive
        raise ShapeError(f"node {index} hangs off no root child")
    return cell_id(roots.index(head), adt.depth(index) - 1)


def index_of(adt, cid: str) -> int:
    """The ``agentdescent`` node index named by ``cid``.

    Raises ``KeyError`` when the trace does not record that cell, which is the
    same answer :class:`dreamrsi.world.ReplayWorld` gives by not offering it.
    """
    if cid == ROOT_ID:
        return 0
    branch, attempt = parse_cell_id(cid)
    roots = branch_roots(adt)
    if branch >= len(roots):
        raise KeyError(cid)
    cursor = roots[branch]
    for _ in range(attempt):
        kids = adt.children(cursor)
        if not kids:
            raise KeyError(cid)
        cursor = kids[0]
    return cursor


def check_shape(adt) -> None:
    """Raise :class:`ShapeError` unless ``adt`` is a ``branch x attempt`` grid."""
    if not adt.paper_shaped:
        raise ShapeError(
            f"tree {adt.name!r} re-expands interior nodes: {adt.unreachable} "
            "recorded node(s) have no b{branch}a{attempt} name. "
            "dreamrsi addresses a tree as a grid; agentdescent does not require it."
        )


# -- dreamrsi -> agentdescent ------------------------------------------------
def to_agentdescent(tree: DiscoveryTree):
    """Record ``tree`` as an ``agentdescent`` world.

    The baseline becomes the root's score, which is where a replaying
    ``agentdescent`` policy looks for it, and every ``dreamrsi`` field that a
    policy or a report reads travels in ``detail``.
    """
    dream = _dream()
    adt = dream.DiscoveryTree.rooted(
        _score_out(tree.baseline_score), name=tree.world_id,
        detail={"cell": ROOT_ID, "task": tree.task, "role": "baseline"})
    # Index order must have parents before children; sorting by (branch, attempt)
    # guarantees it, because a cell's parent is its own branch one attempt up.
    by_cell: Dict[str, int] = {ROOT_ID: 0}
    for node in sorted(tree, key=lambda n: (n.branch, n.attempt)):
        parent = by_cell[node.parent_id]
        added = adt.add(parent, dream.Attempt(
            score=_score_out(node.score), valid=bool(node.valid),
            detail=_detail_of(node)))
        by_cell[node.cell_id] = added.index
    adt.decision_rounds = int((tree.meta or {}).get("decision_rounds", 0))
    return adt


def _detail_of(node: Node) -> Dict[str, Any]:
    detail: Dict[str, Any] = {"cell": node.cell_id}
    for key in _CARRIED:
        value = getattr(node, key, None)
        if value is not None:
            detail[key] = value
    if node.tags:
        detail["tags"] = ",".join(node.tags)
    if node.diagnostics:
        detail["diagnostics"] = dict(node.diagnostics)
    return detail


# -- agentdescent -> dreamrsi ------------------------------------------------
def from_agentdescent(adt, *, task: str = "", world_id: Optional[str] = None,
                      meta: Optional[Mapping[str, Any]] = None) -> DiscoveryTree:
    """Read an ``agentdescent`` world back as a ``dreamrsi`` tree.

    The root's score becomes ``baseline_score``; the root itself does not become
    a node, because ``dreamrsi`` has no cell for it.
    """
    check_shape(adt)
    root = adt.nodes[0] if adt.nodes else None
    tree = DiscoveryTree(
        world_id=world_id or adt.name,
        baseline_score=_score_in(root.score) if root is not None else 0.0,
        task=task or str((root.detail if root else {}).get("task", "")),
        meta={**dict(meta or {}), "decision_rounds": adt.decision_rounds},
    )
    for node in adt.nodes[1:]:
        cid = cell_of(adt, node.index)
        branch, attempt = parse_cell_id(cid)
        detail = dict(node.detail or {})
        tree.add(Node(
            cell_id=cid,
            branch=branch,
            attempt=attempt,
            parent_id=ROOT_ID if attempt == 0 else cell_id(branch, attempt - 1),
            seq=int(detail.get("seq", node.index - 1)),
            score=_score_in(node.score),
            evaluated=bool(detail.get("evaluated", node.score is not None)),
            valid=bool(node.valid),
            fail_class=str(detail.get("fail_class", "ok")),
            error=detail.get("error"),
            n_valid=detail.get("n_valid"),
            n_total=detail.get("n_total"),
            cost=float(detail.get("cost", 1.0)),
            tags=tuple(t for t in str(detail.get("tags", "")).split(",") if t),
            workspace=detail.get("workspace"),
            artifact=detail.get("artifact"),
            proposal=detail.get("proposal"),
            diagnostics=dict(detail.get("diagnostics") or {}),
        ))
    return tree


# -- the two sentinels -------------------------------------------------------
def _score_out(score: float) -> Optional[float]:
    """``dreamrsi``'s ``-inf`` is ``agentdescent``'s ``None`` (and JSON's)."""
    return None if score == float("-inf") else float(score)


def _score_in(score: Optional[float]) -> float:
    return float("-inf") if score is None else float(score)


def _dream():
    from . import require_agentdescent

    return require_agentdescent()


__all__ = ["to_agentdescent", "from_agentdescent", "cell_of", "index_of",
           "branch_roots", "check_shape", "ShapeError"]
