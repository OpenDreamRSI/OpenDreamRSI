"""The evidence layer: replay fidelity, held-out generalization, seed statistics."""

import os

import pytest

from dreamrsi.agents import make_discovery_agent
from dreamrsi.experiments.benchmark import (
    SeedComparison,
    SeedRow,
    bootstrap_ci,
    compare_across_seeds,
    _through_calls,
    _through_round,
)
from dreamrsi.experiments.fidelity import (
    fidelity_probe,
    replay_fidelity,
    support_coverage,
)
from dreamrsi.experiments.generalization import (
    generalization_report,
    spearman,
    temporal_split,
)
from dreamrsi.policy import baseline, library
from dreamrsi.policy.loader import load_policy_source
from dreamrsi.rsi import DREAM, FIXED, RSIConfig
from dreamrsi.tasks import make_task
from dreamrsi.tree import DiscoveryTree, Node
from dreamrsi.types import GridPlan
from dreamrsi.world import ReplayWorld, SimulatorPool

CANDIDATES = [("pi_1", baseline.source())] + [(n, library.source_of(n))
                                              for n in library.available()]
BETAS = (0.2, 1.0)


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    """One live-then-replay probe per (shipped policy, beta), collected once."""
    root = tmp_path_factory.mktemp("fidelity")
    task = make_task("synthetic", {})
    agent = make_discovery_agent("mock", task, {"seed": 0})
    history = os.path.join(str(root), "history")
    os.makedirs(history, exist_ok=True)
    paths = task.prepare(history)
    baseline_score = task.baseline_score(history)
    plan = GridPlan(branch_count=4, refine_count=5, reason="test probe")

    out = []
    for name, source in CANDIDATES:
        for beta in BETAS:
            out.append(fidelity_probe(
                task, agent, source, run_dir=os.path.join(str(root), f"{name}_{beta}"),
                history_dir=history, baseline_dir=paths["baseline_dir"],
                problem_file=paths["problem_file"], baseline_score=baseline_score,
                plan=plan, beta=beta, name=name,
                max_parallelism=4, max_decision_rounds=20))
    return out


# -- exactness ---------------------------------------------------------------
def test_a_frozen_history_replays_the_live_run_exactly(probes):
    """The claim Dream-RSI's economy rests on, checked through serialisation."""
    for out in probes:
        report = out["fidelity"]
        assert report.exact, (
            f"{report.policy} b={report.beta} diverged at round "
            f"{report.divergence_round}: {report.divergence}"
        )
        assert report.live_rounds == report.replay_rounds
        assert report.live_probes == report.replay_probes
        assert report.live_best == report.replay_best
        assert report.score_mismatches == []


def test_replay_costs_no_generation(probes):
    """A replay reveals recorded outcomes; it must never grow the frozen tree."""
    for out in probes:
        tree = out["tree"]
        before = len(tree)
        world = ReplayWorld(tree, max_parallelism=4, max_decision_rounds=20)
        out["spec"].instantiate({"beta": 0.5}).solve(world, None)
        assert len(tree) == before


def test_divergence_is_reported_rather_than_hidden(probes):
    """A policy that behaves differently on replay must be caught, not averaged away."""
    out = probes[0]
    tree = out["tree"]
    fake_batches = [["b0a0"], ["b0a1"]]  # not what the live run actually did
    report = replay_fidelity(
        tree, out["spec"],
        live_batches=fake_batches,
        live_observed={},
        live_best=0.0,
        live_probes=2,
        config={"beta": BETAS[0]},
        max_parallelism=4, max_decision_rounds=20,
    )
    assert not report.identical
    assert report.divergence_round is not None
    assert report.divergence is not None


def test_a_tampered_trace_is_caught(probes):
    """If a stored outcome stops matching what the live run saw, fidelity fails."""
    out = probes[0]
    live = {cid: 0.0 for cid in out["tree"].nodes}
    report = replay_fidelity(
        out["tree"], out["spec"],
        live_batches=[],
        live_observed=live,
        live_best=0.0,
        live_probes=0,
        config={"beta": BETAS[0]},
        max_parallelism=4, max_decision_rounds=20,
    )
    assert report.score_mismatches
    assert not report.exact


# -- reach -------------------------------------------------------------------
def test_support_coverage_is_total_on_a_complete_grid():
    """A trace holding the whole planned grid restricts nothing."""
    tree = DiscoveryTree(world_id="full", baseline_score=0.0,
                         meta={"plan": {"branch_count": 2, "refine_count": 1}})
    for b in range(2):
        for a in range(2):
            tree.add(Node(cell_id=f"b{b}a{a}", branch=b, attempt=a,
                          parent_id="root" if a == 0 else f"b{b}a{a - 1}",
                          seq=len(tree), score=float(b + a), evaluated=True))
    spec = load_policy_source(baseline.source(), name="pi_1", strict=False)
    report = support_coverage(tree, spec, config={"beta": 1.0},
                              max_parallelism=2, max_decision_rounds=10)
    assert report.mean_action_coverage == 1.0
    assert report.unavailable_actions == 0
    assert not report.stopped_by_exhaustion


def test_a_partial_trace_reports_the_actions_it_cannot_serve():
    """The explored-region caveat, measured: one branch, a grid that planned two."""
    tree = DiscoveryTree(world_id="partial", baseline_score=0.0,
                         meta={"plan": {"branch_count": 2, "refine_count": 3}})
    for a in range(2):
        tree.add(Node(cell_id=f"b0a{a}", branch=0, attempt=a,
                      parent_id="root" if a == 0 else f"b0a{a - 1}",
                      seq=a, score=float(a), evaluated=True))
    spec = load_policy_source(baseline.source(), name="pi_1", strict=False)
    report = support_coverage(tree, spec, config={"beta": 1.0},
                              max_parallelism=2, max_decision_rounds=10)

    assert report.planned_grid_known
    assert report.mean_action_coverage < 1.0
    assert report.unavailable_roots >= 1          # b1a0 was planned, never opened
    assert report.stopped_by_exhaustion           # the trace ran out first
    assert report.terminal_live_actions > 0


def test_coverage_is_not_claimed_without_a_recorded_grid():
    """No stored plan means no evidence, and the report says so."""
    tree = DiscoveryTree(world_id="no-plan", baseline_score=0.0)
    tree.add(Node(cell_id="b0a0", branch=0, attempt=0, parent_id="root",
                  seq=0, score=1.0, evaluated=True))
    spec = load_policy_source(baseline.source(), name="pi_1", strict=False)
    report = support_coverage(tree, spec, config={"beta": 1.0},
                              max_parallelism=2, max_decision_rounds=10)
    assert not report.planned_grid_known
    assert report.mean_action_coverage == 1.0


# -- generalization ----------------------------------------------------------
def test_temporal_split_holds_out_the_newest_worlds(probes):
    trees = [o["tree"] for o in probes]
    pool, holdout = temporal_split(trees, holdout_last=2)
    assert len(pool) == len(trees) - 2
    assert holdout == trees[-2:]
    with pytest.raises(ValueError):
        temporal_split(trees[:1], holdout_last=1)


def test_every_candidate_is_scored_on_both_sides_of_the_split(probes):
    pool, holdout = temporal_split([o["tree"] for o in probes], holdout_last=2)
    report = generalization_report(CANDIDATES, pool, holdout, betas=BETAS,
                                   base_config={"beta": 0.62},
                                   max_parallelism=4, max_decision_rounds=20)
    assert len(report.versions) == len(CANDIDATES)
    assert all(v.scored for v in report.versions)
    assert report.selected_by_pool is not None
    assert report.best_on_holdout is not None
    # Regret is measured against an oracle, so it can never be negative.
    assert report.selection_regret >= 0.0


def test_selection_regret_is_zero_when_the_pool_orders_like_the_holdout(probes):
    """A pool that ranks candidates correctly loses nothing by being used."""
    pool, holdout = temporal_split([o["tree"] for o in probes], holdout_last=2)
    report = generalization_report(CANDIDATES, pool, holdout, betas=BETAS,
                                   base_config={"beta": 0.62},
                                   max_parallelism=4, max_decision_rounds=20)
    if report.selected_by_pool == report.best_on_holdout:
        assert report.selection_regret == pytest.approx(0.0)
        assert report.selection_agrees


def test_a_broken_candidate_is_scored_as_unusable_not_crashed(probes):
    pool, holdout = temporal_split([o["tree"] for o in probes], holdout_last=2)
    report = generalization_report([("broken", "this is not python (")], pool, holdout,
                                   betas=BETAS, max_parallelism=4,
                                   max_decision_rounds=20)
    assert report.versions[0].error
    assert not report.versions[0].scored
    assert report.selected_by_pool is None


def test_spearman_matches_known_orderings():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3], [1, 1, 1]) is None      # constant: no ordering
    assert spearman([1.0], [1.0]) is None              # too few points
    assert spearman([1, 2], [1]) is None               # mismatched lengths


# -- seed statistics ---------------------------------------------------------
def test_paired_rows_expose_quality_and_cost_separately():
    rows = [SeedRow(seed=0, fixed_best=-10.0, dream_best=-12.0,
                    fixed_calls=400, dream_calls=200, matched_best=-9.0,
                    matched_calls=396, matched_rounds=9)]
    comparison = SeedComparison(rows=rows)
    assert rows[0].best_delta == pytest.approx(-2.0)      # worse after equal rounds
    assert rows[0].matched_delta == pytest.approx(1.0)    # better after equal calls
    assert rows[0].call_ratio == pytest.approx(2.0)
    summary = comparison.summary()
    assert summary["quality"]["record"] == {"win": 0, "tie": 0, "loss": 1}
    assert summary["quality_at_matched_budget"]["record"] == {"win": 1, "tie": 0, "loss": 0}
    assert summary["cost"]["cheaper_seeds"] == 1


def test_budget_matching_never_credits_an_overspent_round():
    report = {"rounds": [
        {"cumulative_agent_calls": 100, "manifest": {"best_score": -5.0}},
        {"cumulative_agent_calls": 200, "manifest": {"best_score": -3.0}},
        {"cumulative_agent_calls": 300, "manifest": {"best_score": -1.0}},
    ]}
    matched = _through_calls(report, 250)
    assert matched["rounds"] == 2
    assert matched["calls"] == 200
    assert matched["best_score"] == -3.0               # the -1.0 round overran

    assert _through_round(report, 1)["best_score"] == -5.0
    assert _through_calls(report, 10)["rounds"] == 0   # nothing affordable


def test_bootstrap_interval_is_seeded_and_brackets_the_mean():
    values = [1.0, 1.2, 0.9, 1.1, 1.05]
    first = bootstrap_ci(values)
    assert first == bootstrap_ci(values)               # reproducible
    assert first[0] <= sum(values) / len(values) <= first[1]
    assert bootstrap_ci([1.0]) is None


def test_the_comparison_runs_both_arms_under_one_budget(tmp_path):
    base = RSIConfig(name="bench", task="synthetic", mode=DREAM, rounds=2,
                     revisions=1, max_parallelism=4, max_decision_rounds=20,
                     branch_count=3, refine_count=3, out_dir=str(tmp_path))
    comparison = compare_across_seeds(base, [0], equal_budget_rounds=4,
                                      out_dir=str(tmp_path), log=lambda *a, **k: None)
    assert len(comparison.rows) == 1
    row = comparison.rows[0]
    assert row.fixed_calls > 0 and row.dream_calls > 0
    assert row.matched_calls is not None
    # The matched read-off must stay inside the fixed arm's budget.
    assert row.matched_calls <= row.fixed_calls


def test_pool_window_restricts_what_dreaming_may_see(tmp_path):
    """The evolving-worlds ablation: a window of 1 never accumulates history."""
    from dreamrsi.rsi import _dream_pool

    base = RSIConfig(name="w", task="synthetic", out_dir=str(tmp_path))
    trees = [DiscoveryTree(world_id=f"w{i}") for i in range(4)]
    pool = SimulatorPool(trees)

    assert len(_dream_pool(pool, base)) == 4                       # default: all of H_t
    windowed = _dream_pool(pool, RSIConfig(name="w", pool_window=1))
    assert len(windowed) == 1
    assert windowed.trees[0].world_id == "w3"                      # the newest one
    # A window wider than the pool is a no-op rather than an error.
    assert len(_dream_pool(pool, RSIConfig(name="w", pool_window=99))) == 4


def test_fixed_mode_gets_no_policy_developer():
    """The controlled baseline must not be a dreaming run with the result discarded."""
    from dreamrsi.rsi import build_components

    _, _, developer = build_components(RSIConfig(task="synthetic", mode=FIXED))
    assert developer is None
    _, _, developer = build_components(RSIConfig(task="synthetic", mode=DREAM))
    assert developer is not None
