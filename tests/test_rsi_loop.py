"""End-to-end: the three stages and the outer loop."""

import json
import os

import pytest

from dreamrsi.agents import make_discovery_agent, make_policy_developer
from dreamrsi.dreaming import dream, evaluate_policy
from dreamrsi.policy.baseline import SOURCE_PATH, source as baseline_source
from dreamrsi.policy.loader import load_policy_module
from dreamrsi.rsi import DREAM, FIXED, RSIConfig, run_rsi
from dreamrsi.tasks import make_task
from dreamrsi.tree import DiscoveryTree
from dreamrsi.world import SimulatorPool


def config(tmp_path, **kw):
    base = dict(name="t", task="synthetic", mode=DREAM, rounds=2, revisions=1,
                max_parallelism=4, max_decision_rounds=20,
                branch_count=4, refine_count=5, out_dir=str(tmp_path))
    base.update(kw)
    return RSIConfig(**base)


def run(cfg):
    task = make_task(cfg.task, cfg.task_config)
    agent = make_discovery_agent("mock", task, {"seed": cfg.seed})
    dev = make_policy_developer("mock") if cfg.mode == DREAM else None
    return run_rsi(cfg, task=task, agent=agent, developer=dev, log=lambda *a, **k: None)


def test_loop_produces_the_documented_layout(tmp_path):
    cfg = config(tmp_path)
    report = run(cfg)
    root = os.path.join(str(tmp_path), "t")

    assert os.path.exists(os.path.join(root, "report.json"))
    assert os.path.exists(os.path.join(root, "history", "baseline", "eval", "score.json"))
    assert os.path.exists(os.path.join(root, "history", "r0000_init", "method.py"))
    for t in (1, 2):
        rd = os.path.join(root, "history", f"r{t:04d}_dream")
        assert os.path.exists(os.path.join(rd, "trace.json"))
        assert os.path.exists(os.path.join(rd, "live_cycle_manifest.json"))
        assert os.path.exists(os.path.join(rd, "policy_used.py"))
        assert os.path.exists(os.path.join(root, "trace_pool", f"iter{t}",
                                           "live_cycle_manifest.json"))
    assert os.path.exists(os.path.join(root, "trace_pool", "_current",
                                       "live_cycle_manifest.json"))
    # dreaming archives its between-round feedback
    pr = os.path.join(root, "history", "r0001_dream", "proposal_results")
    assert os.path.exists(os.path.join(pr, "beta_sweep.json"))
    assert os.path.exists(os.path.join(pr, "policy_execution_traces.jsonl"))
    assert len(report["rounds"]) == 2


def test_every_attempt_leaves_a_proposal_and_a_score(tmp_path):
    cfg = config(tmp_path, rounds=1, mode=FIXED)
    run(cfg)
    attempts = os.path.join(str(tmp_path), "t", "history", "r0001_fixed", "attempts")
    dirs = sorted(os.listdir(attempts))
    assert dirs
    for d in dirs:
        node = os.path.join(attempts, d)
        assert os.path.exists(os.path.join(node, "proposal.md"))
        assert os.path.exists(os.path.join(node, "eval", "score.json"))


def test_the_pool_grows_by_one_world_per_round(tmp_path):
    report = run(config(tmp_path, rounds=3, revisions=1))
    assert len(report["pool"]) == 3
    assert {w["world_id"] for w in report["pool"]} == {f"t-iter{t}" for t in (1, 2, 3)}


def test_selection_never_regresses_below_the_incumbent(tmp_path):
    """The candidate set contains pi^0, so V^{m*} >= V^0."""
    report = run(config(tmp_path, rounds=3, revisions=2))
    for rec in report["rounds"]:
        if "dream" in rec:
            assert rec["dream"]["selected_reward"] >= rec["dream"]["incumbent_reward"] - 1e-12


def test_fixed_mode_never_changes_its_policy(tmp_path):
    report = run(config(tmp_path, rounds=3, mode=FIXED))
    assert all("dream" not in rec for rec in report["rounds"])
    plans = {(r["plan"]["branch_count"], r["plan"]["refine_count"]) for r in report["rounds"]}
    assert len(plans) == 1


def test_dream_mode_reduces_discovery_cost(tmp_path):
    """Same agent, evaluator, initialisation and per-round grid budget."""
    fixed = run(config(tmp_path / "a", name="fx", mode=FIXED, rounds=5,
                       max_parallelism=6, branch_count=6, refine_count=12,
                       max_decision_rounds=40))
    dreamt = run(config(tmp_path / "b", name="dr", mode=DREAM, rounds=5, revisions=3,
                        max_parallelism=6, branch_count=6, refine_count=12,
                        max_decision_rounds=40))
    assert dreamt["cumulative_agent_calls"] <= fixed["cumulative_agent_calls"]
    assert dreamt["best_score"] >= fixed["best_score"] - 1e-9


def test_dream_alone_over_a_pool(tmp_path):
    cfg = config(tmp_path, rounds=2, mode=FIXED, max_parallelism=6,
                 branch_count=6, refine_count=10, max_decision_rounds=40)
    run(cfg)
    root = os.path.join(str(tmp_path), "t", "history")
    trees = [DiscoveryTree.load(os.path.join(root, f"r{t:04d}_fixed", "trace.json"))
             for t in (1, 2)]
    pool = SimulatorPool(trees, max_parallelism=6, max_decision_rounds=40)
    out = dream(baseline_source(), pool, make_policy_developer("mock"),
                revisions=2, out_dir=str(tmp_path / "dreamt"))
    assert out["best"].score >= out["incumbent"].score - 1e-12
    assert len(out["evaluations"]) == 3          # pi^0 plus two revisions
    traces = os.path.join(str(tmp_path / "dreamt"), "proposal_results",
                          "policy_execution_traces.jsonl")
    with open(traces, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows and {"world_id", "beta", "rounds"} <= set(rows[0])


def test_evaluate_policy_costs_no_agent_calls(tmp_path):
    cfg = config(tmp_path, rounds=1, mode=FIXED)
    run(cfg)
    tree = DiscoveryTree.load(os.path.join(str(tmp_path), "t", "history",
                                           "r0001_fixed", "trace.json"))
    pool = SimulatorPool([tree], max_parallelism=4, max_decision_rounds=20)
    ev = evaluate_policy(load_policy_module(SOURCE_PATH), pool, collect_traces=False)
    # a full beta sweep over the frozen tree: pure replay, nothing is executed
    assert len(ev.points) == 6
    assert ev.summary.attainment == pytest.approx(1.0)


def test_recovering_from_a_failed_parent_keeps_the_gain_shares_finite():
    """`plan_grid` reads these shares; NaN in them silently poisons the next grid.

    A refinement that succeeds where its parent failed has no finite gain over
    that parent -- `score - (-inf)` is `+inf` -- and two such branches made the
    shares `inf / inf`. The baseline is the reference in that case.
    """
    import math

    from dreamrsi.online import live_cycle_manifest
    from dreamrsi.tree import DiscoveryTree, Node
    from dreamrsi.types import ROOT_ID, GridPlan, SimResult, cell_id

    tree = DiscoveryTree(world_id="w", baseline_score=1.0, task="synthetic")
    for branch in (0, 1):
        tree.add(Node(cell_id=cell_id(branch, 0), branch=branch, attempt=0,
                      parent_id=ROOT_ID, seq=0, score=float("-inf"),
                      evaluated=True, valid=False, fail_class="timeout",
                      error="exceeded"))
        tree.add(Node(cell_id=cell_id(branch, 1), branch=branch, attempt=1,
                      parent_id=cell_id(branch, 0), seq=1, score=2.0 + branch,
                      evaluated=True, valid=True, fail_class="ok"))

    class _World:
        agent_calls, wall_clock = 4, 1.0

    manifest = live_cycle_manifest(
        tree, SimResult(total_probes=4, decision_rounds=2,
                        effective_sequential_rounds=2, best_score=3.0),
        GridPlan(branch_count=2, refine_count=1, reason="x"), _World())

    for key in ("early_gain_share", "late_gain_share"):
        assert math.isfinite(manifest[key]), (key, manifest[key])
    assert manifest["early_gain_share"] + manifest["late_gain_share"] == pytest.approx(1.0)
    assert manifest["late_gain_share"] == pytest.approx(1.0)   # all of it was depth
