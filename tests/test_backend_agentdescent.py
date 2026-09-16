"""The agentdescent engine backend must not be observable from outside.

Swapping which engine walks the tree is an implementation choice; cell ids, the
policy API, the cost accounting, the recorded trace and the run layout are the
semantics. So these tests are mostly one assertion made many ways: run the same
thing on both engines and diff everything a caller can see.
"""

import os

import pytest

from dreamrsi.agents import make_discovery_agent
from dreamrsi.backends import available
from dreamrsi.online import run_online_cycle as native_cycle
from dreamrsi.policy import baseline, library
from dreamrsi.policy.loader import load_policy_source
from dreamrsi.rsi import DREAM, ENGINES, FIXED, RSIConfig, online_engine, run_rsi
from dreamrsi.tasks import make_task
from dreamrsi.tree import DiscoveryTree, Node
from dreamrsi.types import ROOT_ID, GridPlan, cell_id

pytestmark = pytest.mark.skipif(not available(), reason="agentdescent is not installed")

CANDIDATES = [("pi_1", baseline.source())] + [(n, library.source_of(n))
                                              for n in library.available()]


def _cycle(engine, tmp_path, source, *, beta, branch_count=4, refine_count=3,
           max_parallelism=3, name="p"):
    """One online cycle on ``engine``, reduced to everything a caller can see."""
    root = str(tmp_path / f"{engine}-{name}-{beta}-{branch_count}x{refine_count}")
    history = os.path.join(root, "history")
    os.makedirs(history, exist_ok=True)
    task = make_task("synthetic", {})
    agent = make_discovery_agent("mock", task, {"seed": 0})
    paths = task.prepare(history)
    spec = load_policy_source(source, name=name, strict=False)
    out = online_engine(engine)(
        task=task, agent=agent, policy=spec.instantiate({"beta": beta}),
        plan=GridPlan(branch_count=branch_count, refine_count=refine_count,
                      reason="test"),
        run_dir=os.path.join(history, "r0001"), history_dir=history,
        baseline_dir=paths["baseline_dir"], problem_file=paths["problem_file"],
        baseline_score=task.baseline_score(history),
        max_parallelism=max_parallelism, max_decision_rounds=12,
        world_id="w", cycle=1)
    world, result, tree = out["world"], out["result"], out["tree"]
    return {
        "batches": [list(b) for b in world.batches],
        "probes": result.total_probes,
        "decision_rounds": result.decision_rounds,
        "effective_sequential_rounds": result.effective_sequential_rounds,
        "best_score": result.best_score,
        "best_cell": result.best_cell,
        "revealed": sorted(result.revealed),
        "curve": list(result.curve),
        "stopped_reason": result.stopped_reason,
        "agent_calls": world.agent_calls,
        "round_log": world.round_log,
        "cells": sorted(tree.nodes),
        "scores": {c: n.score for c, n in sorted(tree.nodes.items())},
        "seqs": {c: n.seq for c, n in sorted(tree.nodes.items())},
        "fail_classes": {c: n.fail_class for c, n in sorted(tree.nodes.items())},
        # `seconds` is wall clock, which is never reproducible; that it is
        # *measured at all* on both engines is asserted separately below.
        "manifest": {k: v for k, v in out["manifest"].items() if k != "seconds"},
        "measured_seconds": out["manifest"]["seconds"] >= 0.0,
    }


@pytest.mark.parametrize("name,source", CANDIDATES, ids=[n for n, _ in CANDIDATES])
@pytest.mark.parametrize("beta", [0.0, 0.62, 1.0])
def test_both_engines_produce_the_same_episode(tmp_path, name, source, beta):
    """Same policy, same agent, same grid -- same everything."""
    a = _cycle("native", tmp_path, source, beta=beta, name=name)
    b = _cycle("agentdescent", tmp_path, source, beta=beta, name=name)
    assert a == b


@pytest.mark.parametrize("grid", [(2, 5, 2), (6, 2, 4)])
def test_the_agreement_holds_across_grid_shapes(tmp_path, grid):
    branch_count, refine_count, workers = grid
    for name, source in CANDIDATES:
        a = _cycle("native", tmp_path, source, beta=0.62, name=name,
                   branch_count=branch_count, refine_count=refine_count,
                   max_parallelism=workers)
        b = _cycle("agentdescent", tmp_path, source, beta=0.62, name=name,
                   branch_count=branch_count, refine_count=refine_count,
                   max_parallelism=workers)
        assert a == b, name


def test_the_trace_on_disk_is_the_same_world(tmp_path):
    """Including through serialisation -- a world has to survive the disk."""
    for engine in ENGINES:
        _cycle(engine, tmp_path, baseline.source(), beta=1.0)
    loaded = {}
    for engine in ENGINES:
        path = os.path.join(str(tmp_path), f"{engine}-p-1.0-4x3", "history",
                            "r0001", "trace.json")
        tree = DiscoveryTree.load(path)
        loaded[engine] = (tree.summary(), sorted(tree.nodes),
                          {c: n.score for c, n in sorted(tree.nodes.items())})
    assert loaded["native"] == loaded["agentdescent"]


def test_the_engine_is_recorded_so_a_trace_says_who_made_it(tmp_path):
    _cycle("agentdescent", tmp_path, baseline.source(), beta=1.0)
    path = os.path.join(str(tmp_path), "agentdescent-p-1.0-4x3", "history",
                        "r0001", "trace.json")
    assert DiscoveryTree.load(path).meta["engine"] == "agentdescent"


def test_a_policy_that_crashes_surfaces_on_the_caller_thread(tmp_path):
    """The policy runs on a worker thread; its exception must not be swallowed."""
    source = (
        "from see.policy.api import LLMDesignedMethod, GridPlan\n"
        "NAME = 'OptimalPolicy'\n"
        "class OptimalPolicy(LLMDesignedMethod):\n"
        "    def solve(self, question, budget=None):\n"
        "        question.reset()\n"
        "        question.probe_batch(question.legal_roots()[:1])\n"
        "        raise RuntimeError('policy exploded')\n"
        "    def plan_grid(self, context):\n"
        "        return GridPlan(branch_count=2, refine_count=1, reason='x')\n"
    )
    with pytest.raises(RuntimeError, match="policy exploded"):
        _cycle("agentdescent", tmp_path, source, beta=0.5, name="boom")


def test_an_illegal_batch_still_fails_loudly(tmp_path):
    """The legality rules are the world's, and the bridge must not soften them."""
    source = (
        "from see.policy.api import LLMDesignedMethod, GridPlan\n"
        "NAME = 'OptimalPolicy'\n"
        "class OptimalPolicy(LLMDesignedMethod):\n"
        "    def solve(self, question, budget=None):\n"
        "        question.reset()\n"
        "        question.probe_batch(['b0a0', 'b0a0'])\n"   # a duplicate
        "    def plan_grid(self, context):\n"
        "        return GridPlan(branch_count=2, refine_count=1, reason='x')\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        _cycle("agentdescent", tmp_path, source, beta=0.5, name="dup")


def test_the_decision_round_cap_still_stops_the_rollout(tmp_path):
    """K1 belongs to the engine; a policy that would keep going is cut off."""
    root = str(tmp_path / "cap")
    history = os.path.join(root, "history")
    os.makedirs(history, exist_ok=True)
    task = make_task("synthetic", {})
    agent = make_discovery_agent("mock", task, {"seed": 0})
    paths = task.prepare(history)
    spec = load_policy_source(baseline.source(), name="pi_1", strict=False)
    out = online_engine("agentdescent")(
        task=task, agent=agent, policy=spec.instantiate({"beta": 1.0}),
        plan=GridPlan(branch_count=3, refine_count=9, reason="deep"),
        run_dir=os.path.join(history, "r0001"), history_dir=history,
        baseline_dir=paths["baseline_dir"], problem_file=paths["problem_file"],
        baseline_score=task.baseline_score(history),
        max_parallelism=3, max_decision_rounds=2, world_id="w", cycle=1)
    assert out["result"].decision_rounds == 2
    assert len(out["tree"]) == 6                      # two full batches, no more


def test_the_full_loop_runs_on_either_engine(tmp_path):
    """And the engine choice does not change the report."""
    reports = {}
    for engine in ENGINES:
        config = RSIConfig(name=f"loop-{engine}", task="synthetic", mode=DREAM,
                           rounds=2, revisions=1, max_parallelism=3,
                           max_decision_rounds=10, branch_count=3, refine_count=2,
                           engine=engine, out_dir=str(tmp_path))
        task = make_task("synthetic", {})
        agent = make_discovery_agent("mock", task, {"seed": 0})
        from dreamrsi.agents import make_policy_developer

        report = run_rsi(config, task=task, agent=agent,
                         developer=make_policy_developer("mock"),
                         log=lambda *a, **k: None)
        reports[engine] = {
            "best_score": report["best_score"],
            "cumulative_agent_calls": report["cumulative_agent_calls"],
            # world_id carries the run name, which differs by construction here.
            "pool": [{k: v for k, v in w.items() if k != "world_id"}
                     for w in report["pool"]],
            "rounds": [{k: v for k, v in r.items() if k != "manifest"}
                       for r in report["rounds"]],
        }
    assert reports["native"] == reports["agentdescent"]


def test_an_unknown_engine_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="native"):
        online_engine("nope")


# -- stage 2: the replay half ------------------------------------------------
def _worlds(tmp_path, n=3):
    """A few real recorded worlds, built once with the native engine."""
    history = os.path.join(str(tmp_path), "pool", "history")
    os.makedirs(history, exist_ok=True)
    task = make_task("synthetic", {})
    agent = make_discovery_agent("mock", task, {"seed": 0})
    paths = task.prepare(history)
    trees = []
    for i, (name, source) in enumerate(CANDIDATES[:n]):
        spec = load_policy_source(source, name=name, strict=False)
        out = native_cycle(
            task=task, agent=agent, policy=spec.instantiate({"beta": 0.62}),
            plan=GridPlan(branch_count=4, refine_count=3, reason="pool"),
            run_dir=os.path.join(history, f"r{i}"), history_dir=history,
            baseline_dir=paths["baseline_dir"], problem_file=paths["problem_file"],
            baseline_score=task.baseline_score(history), max_parallelism=3,
            max_decision_rounds=12, world_id=f"w{i}", cycle=1)
        trees.append(out["tree"])
    return trees


def _pools(trees):
    from dreamrsi.backends.agentdescent_replay import AgentDescentSimulatorPool
    from dreamrsi.backends.agentdescent_tree import to_agentdescent
    from dreamrsi.world import SimulatorPool

    return (SimulatorPool(trees, max_parallelism=3, max_decision_rounds=12),
            AgentDescentSimulatorPool([to_agentdescent(t) for t in trees],
                                      max_parallelism=3, max_decision_rounds=12,
                                      task="synthetic"))


def test_both_pools_describe_the_same_history(tmp_path):
    native, agentdescent = _pools(_worlds(tmp_path))
    assert native.summary() == agentdescent.summary()
    assert len(native) == len(agentdescent)


@pytest.mark.parametrize("name,source", CANDIDATES, ids=[n for n, _ in CANDIDATES])
def test_the_beta_sweep_is_the_same_on_either_pool(tmp_path, name, source):
    """The replay objective is what selection reads, so it must not move."""
    from dreamrsi.dreaming import evaluate_policy

    native, agentdescent = _pools(_worlds(tmp_path))
    spec = load_policy_source(source, name=name, strict=False)
    betas = (0.0, 0.5, 1.0)
    a = evaluate_policy(spec, native, betas=betas, base_config={"beta": 0.62})
    b = evaluate_policy(spec, agentdescent, betas=betas, base_config={"beta": 0.62})

    assert a.summary.to_dict() == b.summary.to_dict()
    assert [p.to_dict() for p in a.points] == [p.to_dict() for p in b.points]
    assert a.score == b.score


def test_replay_on_the_agentdescent_world_reveals_the_named_branch(tmp_path):
    """Not the next one in recorded order, which is agentdescent's own rule.

    ``agentdescent`` gives the root a multiplicity of ``W`` and reveals
    ``children(root)[taken]`` positionally. Here a policy names the branch, and
    the cell it names is the cell it gets -- which is what makes a cell id mean
    anything to a policy.
    """
    from dreamrsi.backends.agentdescent_replay import AgentDescentReplayWorld
    from dreamrsi.backends.agentdescent_tree import to_agentdescent

    trees = _worlds(tmp_path, n=1)
    world = AgentDescentReplayWorld(to_agentdescent(trees[0]), max_parallelism=3,
                                    max_decision_rounds=12)
    world.reset()
    (obs,) = world.probe_batch(["b2a0"])
    assert obs.cell_id == "b2a0"
    assert obs.score == trees[0].nodes["b2a0"].score


def test_a_branch_that_ended_is_not_offered_rather_than_barren(tmp_path):
    """agentdescent would offer it and charge a wasted worker; this repo does not."""
    from dreamrsi.backends import require_agentdescent
    from dreamrsi.backends.agentdescent_replay import AgentDescentReplayWorld

    dream = require_agentdescent()
    adt = dream.DiscoveryTree.rooted(0.0, name="short", detail={"cell": "root"})
    first = adt.add(0, dream.Attempt(score=1.0, detail={"cell": "b0a0"}))
    adt.add(first.index, dream.Attempt(score=2.0, detail={"cell": "b0a1"}))

    world = AgentDescentReplayWorld(adt, max_parallelism=2, max_decision_rounds=8)
    world.reset()
    world.probe_batch(["b0a0"])
    assert world.legal_actions() == ["b0a1"]
    world.probe_batch(["b0a1"])
    assert world.legal_actions() == []          # exhausted, and simply not offered


def test_an_illegal_batch_raises_instead_of_being_counted(tmp_path):
    """agentdescent normalises and scores it; this repo fails loudly."""
    from dreamrsi.backends.agentdescent_replay import AgentDescentReplayWorld
    from dreamrsi.backends.agentdescent_tree import to_agentdescent
    from dreamrsi.world import IllegalBatch

    world = AgentDescentReplayWorld(to_agentdescent(_worlds(tmp_path, n=1)[0]),
                                    max_parallelism=2, max_decision_rounds=12)
    world.reset()
    with pytest.raises(IllegalBatch, match="duplicate"):
        world.probe_batch(["b0a0", "b0a0"])
    with pytest.raises(IllegalBatch, match="max_parallelism"):
        world.probe_batch(["b0a0", "b1a0", "b2a0"])
    with pytest.raises(IllegalBatch, match="illegal cells"):
        world.probe_batch(["b0a1"])             # its parent is not revealed yet


def test_replay_costs_no_generation_on_the_agentdescent_world(tmp_path):
    from dreamrsi.backends.agentdescent_replay import AgentDescentReplayWorld
    from dreamrsi.backends.agentdescent_tree import to_agentdescent

    adt = to_agentdescent(_worlds(tmp_path, n=1)[0])
    before = len(adt)
    world = AgentDescentReplayWorld(adt, max_parallelism=3, max_decision_rounds=12)
    spec = load_policy_source(baseline.source(), name="pi_1", strict=False)
    spec.instantiate({"beta": 1.0}).solve(world, None)
    assert len(adt) == before


# -- the tree bridge ---------------------------------------------------------
def _grid_tree():
    tree = DiscoveryTree(world_id="w", baseline_score=2.5, task="synthetic",
                         meta={"plan": {"branch_count": 3, "refine_count": 1}})
    spec = [("b0a0", 3.0, "ok"), ("b0a1", float("-inf"), "timeout"),
            ("b1a0", float("-inf"), "correctness"), ("b2a0", 2.75, "ok")]
    for i, (cid, score, fail) in enumerate(spec):
        branch, attempt = int(cid[1]), int(cid[3])
        tree.add(Node(cell_id=cid, branch=branch, attempt=attempt,
                      parent_id=ROOT_ID if attempt == 0 else cell_id(branch, attempt - 1),
                      seq=i, score=score, evaluated=True, valid=(fail == "ok"),
                      fail_class=fail, error=None if fail == "ok" else f"{fail}!",
                      diagnostics={"seconds": float(i)}))
    return tree


def test_a_world_survives_the_trip_through_the_other_model():
    from dreamrsi.backends.agentdescent_tree import from_agentdescent, to_agentdescent

    original = _grid_tree()
    back = from_agentdescent(to_agentdescent(original), task=original.task)

    assert back.baseline_score == original.baseline_score
    assert back.ceiling == original.ceiling
    assert sorted(back.nodes) == sorted(original.nodes)
    for cid, node in original.nodes.items():
        got = back.nodes[cid]
        for field in ("branch", "attempt", "parent_id", "score", "evaluated",
                      "valid", "fail_class", "error", "seq", "cost"):
            assert getattr(got, field) == getattr(node, field), (cid, field)
        assert got.diagnostics == node.diagnostics


def test_the_cell_bijection_holds_both_ways():
    from dreamrsi.backends.agentdescent_tree import cell_of, index_of, to_agentdescent

    adt = to_agentdescent(_grid_tree())
    assert cell_of(adt, 0) == ROOT_ID
    for node in adt.nodes:
        assert index_of(adt, cell_of(adt, node.index)) == node.index
    with pytest.raises(KeyError):
        index_of(adt, "b9a9")


def test_a_tree_with_no_grid_name_is_refused_rather_than_truncated():
    """agentdescent tolerates re-expanded interior nodes; dreamrsi cannot name them."""
    from dreamrsi.backends import require_agentdescent
    from dreamrsi.backends.agentdescent_tree import ShapeError, from_agentdescent

    dream = require_agentdescent()
    adt = dream.DiscoveryTree.rooted(0.0, name="wide")
    first = adt.add(0, dream.Attempt(score=1.0))
    adt.add(first.index, dream.Attempt(score=2.0))
    adt.add(first.index, dream.Attempt(score=3.0))      # a second child: no cell id
    assert adt.unreachable == 1
    with pytest.raises(ShapeError, match="re-expands"):
        from_agentdescent(adt)


def test_the_engine_still_measures_the_batch_wall_clock(tmp_path):
    """The engine owns the thread pool now, so the world has to time the wait.

    Without this the manifest reports `seconds: 0.0` for every round on the
    agentdescent engine -- a field that silently stops meaning anything.
    """
    root = str(tmp_path / "clock")
    history = os.path.join(root, "history")
    os.makedirs(history, exist_ok=True)
    task = make_task("synthetic", {})
    agent = make_discovery_agent("mock", task, {"seed": 0})
    paths = task.prepare(history)
    spec = load_policy_source(baseline.source(), name="pi_1", strict=False)
    out = online_engine("agentdescent")(
        task=task, agent=agent, policy=spec.instantiate({"beta": 1.0}),
        plan=GridPlan(branch_count=3, refine_count=2, reason="clock"),
        run_dir=os.path.join(history, "r0001"), history_dir=history,
        baseline_dir=paths["baseline_dir"], problem_file=paths["problem_file"],
        baseline_score=task.baseline_score(history), max_parallelism=3,
        max_decision_rounds=8, world_id="w", cycle=1)
    assert out["world"].wall_clock > 0.0
    assert out["manifest"]["seconds"] > 0.0
