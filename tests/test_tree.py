"""Discovery-tree invariants."""

import pytest

from dreamrsi.tree import DiscoveryTree
from dreamrsi.types import ROOT_ID, cell_id, parse_cell_id


def build(branches=3, depth=3, baseline=0.0):
    tree = DiscoveryTree("t", baseline_score=baseline)
    for b in range(branches):
        parent = ROOT_ID
        for a in range(depth):
            node = tree.continue_from(parent, score=0.1 * b + 0.01 * a,
                                      evaluated=True, valid=True)
            parent = node.cell_id
    return tree


def test_cell_id_roundtrip():
    assert parse_cell_id(cell_id(7, 11)) == (7, 11)
    with pytest.raises(ValueError):
        parse_cell_id("nonsense")


def test_continue_from_root_opens_a_branch():
    tree = DiscoveryTree("t")
    a = tree.continue_from(ROOT_ID, score=1.0)
    b = tree.continue_from(ROOT_ID, score=2.0)
    assert (a.branch, a.attempt) == (0, 0)
    assert (b.branch, b.attempt) == (1, 0)
    assert tree.branch_count == 2


def test_continue_from_leaf_extends_the_same_branch():
    tree = DiscoveryTree("t")
    root = tree.continue_from(ROOT_ID, score=1.0)
    child = tree.continue_from(root.cell_id, score=2.0)
    assert (child.branch, child.attempt) == (0, 1)
    assert child.parent_id == root.cell_id


def test_only_leaves_and_the_root_are_eligible():
    tree = build(branches=2, depth=2)
    eligible = set(tree.eligible_continuations())
    assert ROOT_ID in eligible
    assert cell_id(0, 1) in eligible      # the leaf
    assert cell_id(0, 0) not in eligible  # already has a child
    with pytest.raises(ValueError):
        tree.continue_from(cell_id(0, 0), score=0.0)


def test_observation_deltas_use_the_primary_parent():
    tree = DiscoveryTree("t", baseline_score=1.0)
    root = tree.continue_from(ROOT_ID, score=2.0, evaluated=True, valid=True)
    child = tree.continue_from(root.cell_id, score=5.0, evaluated=True, valid=True)
    obs = tree.observation_of(child.cell_id)
    assert obs.delta_vs_parent == pytest.approx(3.0)
    assert obs.delta_vs_baseline == pytest.approx(4.0)
    root_obs = tree.observation_of(root.cell_id)
    assert root_obs.delta_vs_parent == pytest.approx(1.0)  # parent is the baseline


def test_roundtrip_through_json(tmp_path):
    tree = build(branches=3, depth=4, baseline=0.5)
    path = tree.save(str(tmp_path / "trace.json"))
    again = DiscoveryTree.load(path)
    assert again.summary() == tree.summary()
    assert len(again) == len(tree)


def test_ceiling_and_grid_shape():
    tree = build(branches=4, depth=5)
    assert tree.branch_count == 4
    assert tree.refine_count == 4          # attempts 0..4
    assert tree.ceiling == pytest.approx(max(n.score for n in tree))
