"""Replay-world legality rules and batch accounting."""

import pytest

from dreamrsi.tree import DiscoveryTree
from dreamrsi.types import ROOT_ID, cell_id
from dreamrsi.world import IllegalBatch, ReplayWorld, SimulatorPool


def build(branches=4, depth=4):
    tree = DiscoveryTree("w", baseline_score=0.0)
    for b in range(branches):
        parent = ROOT_ID
        for a in range(depth):
            parent = tree.continue_from(parent, score=float(b) + 0.1 * a,
                                        evaluated=True, valid=True).cell_id
    return tree


def world(**kw):
    return ReplayWorld(build(**kw.pop("tree", {})), **kw) if "tree" in kw else \
        ReplayWorld(build(), **kw)


def test_legal_actions_are_roots_plus_frontiers():
    w = world(max_parallelism=4)
    assert set(w.legal_actions()) == {cell_id(b, 0) for b in range(4)}
    w.probe_batch([cell_id(0, 0)])
    assert set(w.legal_actions()) == {cell_id(b, 0) for b in range(1, 4)} | {cell_id(0, 1)}
    assert w.legal_roots() == [cell_id(b, 0) for b in range(1, 4)]
    assert w.opened_branches() == [0]


def test_batch_accounting_matches_the_prompt():
    """k probes cost one decision round and ceil(k / W) sequential rounds."""
    w = world(max_parallelism=2)
    w.probe_batch([cell_id(0, 0), cell_id(1, 0)])
    assert (w.total_probes, w.decision_rounds, w.effective_sequential_rounds) == (2, 1, 1)
    w.probe_batch([cell_id(0, 1)])
    assert (w.total_probes, w.decision_rounds, w.effective_sequential_rounds) == (3, 2, 2)


def test_illegal_batches_are_rejected():
    w = world(max_parallelism=2)
    with pytest.raises(IllegalBatch):
        w.probe_batch([cell_id(0, 0), cell_id(0, 0)])          # duplicate
    with pytest.raises(IllegalBatch):
        w.probe_batch([cell_id(b, 0) for b in range(3)])       # exceeds W
    with pytest.raises(IllegalBatch):
        w.probe_batch([cell_id(0, 1)])                         # not legal yet
    w.probe_batch([cell_id(0, 0)])
    with pytest.raises(IllegalBatch):
        w.probe_batch([cell_id(0, 1), cell_id(0, 2)])          # parent and child


def test_empty_batch_terminates_without_cost():
    w = world()
    assert w.probe_batch([]) == []
    assert (w.total_probes, w.decision_rounds) == (0, 0)


def test_reset_clears_the_prefix():
    w = world()
    w.probe_batch([cell_id(0, 0)])
    w.reset()
    assert w.observed() == {} and w.total_probes == 0 and w.best_so_far == float("-inf")


def test_replay_is_deterministic_and_costs_no_execution():
    w = world()
    first = [o.score for o in w.probe_batch([cell_id(0, 0), cell_id(1, 0)])]
    w.reset()
    second = [o.score for o in w.probe_batch([cell_id(0, 0), cell_id(1, 0)])]
    assert first == second


def test_frontier_stops_at_the_recorded_depth():
    tree = DiscoveryTree("w")
    tree.continue_from(ROOT_ID, score=1.0, evaluated=True)   # a branch of depth 1
    w = ReplayWorld(tree, max_parallelism=2)
    w.probe_batch([cell_id(0, 0)])
    assert w.legal_actions() == []       # nothing recorded beyond it


def test_pool_builds_one_world_per_tree():
    pool = SimulatorPool([build(), build(branches=2)], max_parallelism=3)
    worlds = pool.worlds()
    assert len(pool) == 2 and len(worlds) == 2
    assert all(x.max_parallelism == 3 for x in worlds)
