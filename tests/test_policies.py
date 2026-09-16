"""Behaviour of the shipped policies on frozen worlds."""

import pytest

from dreamrsi.dreaming import evaluate_policy
from dreamrsi.policy import library
from dreamrsi.policy.baseline import SOURCE_PATH
from dreamrsi.policy.loader import load_policy_module, load_policy_source
from dreamrsi.tree import DiscoveryTree
from dreamrsi.types import ROOT_ID
from dreamrsi.world import ReplayWorld, SimulatorPool

POLICY_PATHS = [SOURCE_PATH] + [
    f"dreamrsi/policy/library/{n}.py" for n in library.available()
]


def make_tree(world_id, branches=6, depth=8, seed=0):
    """A grid where branch quality differs and one branch hides the ceiling."""
    import random

    rng = random.Random(seed)
    tree = DiscoveryTree(world_id, baseline_score=0.0)
    for b in range(branches):
        quality = (b % 3) + rng.random()
        parent, score = ROOT_ID, 0.0
        for a in range(depth):
            score += quality * (0.7 ** a) + rng.gauss(0, 0.05)
            fail = (a == 2 and b == 1)
            parent = tree.continue_from(
                parent, score=score, evaluated=True, valid=not fail,
                fail_class="compile_other" if fail else "ok",
                error="build error" if fail else None,
            ).cell_id
    return tree


def pool(n=3, **kw):
    return SimulatorPool([make_tree(f"w{i}", seed=i) for i in range(n)],
                         max_parallelism=kw.get("max_parallelism", 6),
                         max_decision_rounds=kw.get("max_decision_rounds", 40))


@pytest.mark.parametrize("path", POLICY_PATHS)
def test_every_shipped_policy_passes_the_static_checks(path):
    spec = load_policy_module(path)
    assert spec.warnings == []
    assert spec.overrides_plan_grid


@pytest.mark.parametrize("path", POLICY_PATHS)
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_policies_terminate_and_stay_legal(path, beta):
    spec = load_policy_module(path)
    w = ReplayWorld(make_tree("w"), max_parallelism=6, max_decision_rounds=40)
    res = spec.instantiate({"beta": beta}).solve(w, None)   # replay: budget is None
    assert res.decision_rounds <= w.max_decision_rounds
    assert res.total_probes <= len(w.tree)
    assert len(res.revealed) == len(set(res.revealed))
    assert all(len(b) <= w.max_parallelism for b in res.batches)


@pytest.mark.parametrize("path", POLICY_PATHS)
def test_plan_grid_always_returns_a_plan(path):
    from dreamrsi.types import GridPlanningContext

    spec = load_policy_module(path)
    policy = spec.instantiate({"beta": 0.6})
    empty = GridPlanningContext(hard_max_branch_count=8, hard_max_refine_count=8)
    plan = policy.plan_grid(empty)
    assert plan is not None and plan.reason
    assert 1 <= plan.branch_count <= 8 and 0 <= plan.refine_count <= 8

    with_history = GridPlanningContext(
        history=({"planned_branch_count": 4, "planned_refine_count": 6,
                  "effective_branch_count": 4, "effective_refine_count": 6,
                  "early_gain_share": 0.1, "late_gain_share": 0.9, "plateaued": False,
                  "unopened_roots": 0, "hard_failures": 0, "probes": 28,
                  "best_score": 1.0},),
        hard_max_branch_count=16, hard_max_refine_count=16)
    plan = policy.plan_grid(with_history)
    if path != SOURCE_PATH:
        # the fixed baseline holds its grid by design; a proposed policy must
        # respond to the evidence -- late gains mean deepen or hold.
        assert plan.refine_count >= 6
    assert plan.reason


def test_the_editable_template_is_runnable_but_flagged():
    """The stub handed to the developer must load, run and terminate ...

    ... while still being flagged: the improvement prompt forbids inheriting the
    template's ``plan_grid``, so a proposed policy that does not override it is
    not a valid deliverable.
    """
    from dreamrsi.policy import template

    spec = load_policy_source(template.source(), name="template")
    assert any("delegated to the runner's fallback" in w for w in spec.warnings)

    w = ReplayWorld(make_tree("w"), max_parallelism=6, max_decision_rounds=40)
    res = spec.instantiate({"beta": 0.6}).solve(w, None)
    assert res.decision_rounds <= w.max_decision_rounds

    # and it is comfortably beaten by the policies in the library
    p = pool()
    stub = evaluate_policy(spec, p, collect_traces=False)
    better = evaluate_policy(
        load_policy_module("dreamrsi/policy/library/adaptive_portfolio.py"), p,
        collect_traces=False)
    assert better.summary.reward > stub.summary.reward


def test_baseline_reveals_the_whole_frozen_grid():
    spec = load_policy_module(SOURCE_PATH)
    w = ReplayWorld(make_tree("w"), max_parallelism=6, max_decision_rounds=40)
    res = spec.instantiate({"beta": 1.0}).solve(w, None)
    # parallel refine opens W workspaces and refines each to exhaustion
    assert res.total_probes == len(w.tree)
    assert res.best_score == pytest.approx(w.tree.ceiling)


def test_adaptive_policy_spends_fewer_probes_than_the_baseline():
    p = pool()
    base = evaluate_policy(load_policy_module(SOURCE_PATH), p, collect_traces=False)
    adaptive = evaluate_policy(
        load_policy_module("dreamrsi/policy/library/adaptive_portfolio.py"), p,
        collect_traces=False)
    assert adaptive.summary.mean_probes < base.summary.mean_probes


def test_beta_actually_moves_the_trade_off():
    """The sweep must be non-degenerate: beta has to change attainment or work."""
    adaptive = evaluate_policy(
        load_policy_module("dreamrsi/policy/library/adaptive_portfolio.py"), pool(),
        collect_traces=False)
    assert adaptive.summary.non_degenerate
    by_beta = adaptive.summary.per_beta
    lo, hi = by_beta["0.000"], by_beta["1.000"]
    assert hi["total_probes"] >= lo["total_probes"]   # high beta is more patient


def test_a_crashing_policy_fails_softly():
    src = '''
from see.policy.api import LLMDesignedMethod, GridPlan
NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def solve(self, question, budget=None):
        raise RuntimeError("boom")

    def plan_grid(self, context):
        return GridPlan(1, 1, "x")
'''
    ev = evaluate_policy(load_policy_source(src, name="bad"), pool(1), collect_traces=False)
    assert ev.summary.reward <= 0.0


def test_an_illegal_batch_is_caught_not_crashed():
    src = '''
from see.policy.api import LLMDesignedMethod, GridPlan, SimResult, finalize_result
NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def solve(self, question, budget=None):
        question.reset()
        question.probe_batch(["b0a0", "b0a0"])     # duplicate ids
        return finalize_result(question, SimResult())

    def plan_grid(self, context):
        return GridPlan(1, 1, "x")
'''
    ev = evaluate_policy(load_policy_source(src, name="illegal"), pool(1), collect_traces=False)
    assert ev.summary.attainment == 0.0


def test_a_crash_is_scored_on_what_the_policy_actually_did():
    """A revision that explored well and then hit a typo is not a revision that
    did nothing, and the reflector is asked to tell them apart.

    Measured: a Haiku-written revision treated `question.observed()` values as
    dicts, crashed on its second round, and was recorded as 0 probes / reward
    -0.25 -- the same floor as a policy that never selected a batch. It had in
    fact revealed a full batch and reached 0.915 attainment.
    """
    src = '''
from see.policy.api import LLMDesignedMethod, GridPlan, SimResult, _record_curve
NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        question.probe_batch(question.legal_roots()[:2],
                             on_reveal=lambda _: _record_curve(res, question))
        raise AttributeError("'Observation' object has no attribute 'get'")

    def plan_grid(self, context):
        return GridPlan(2, 1, "x")
'''
    ev = evaluate_policy(load_policy_source(src, name="partial"), pool(1),
                         collect_traces=True)
    point = ev.points[0]
    assert point.total_probes == 2          # the batch it did reveal is counted
    assert point.decision_rounds == 1
    assert ev.summary.attainment > 0.0      # and what it attained is measured
    assert "crash" in ev.traces[0]["stopped_reason"]


def test_a_crashing_policy_can_never_be_selected():
    """Scoring it low is not enough -- the winner gets redeployed online.

    `points` is appended for every episode, crashed or not, so the old
    "errors and not points" guard never fired; a crasher was held back only by
    the floor reward of an empty episode. Salvaging the trajectory removes that
    floor, so the disqualification has to be explicit.
    """
    src = '''
from see.policy.api import LLMDesignedMethod, GridPlan, SimResult, _record_curve
NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        question.probe_batch(question.legal_roots()[:2],
                             on_reveal=lambda _: _record_curve(res, question))
        raise RuntimeError("boom")

    def plan_grid(self, context):
        return GridPlan(2, 1, "x")
'''
    ev = evaluate_policy(load_policy_source(src, name="crasher"), pool(1),
                         collect_traces=False)
    assert ev.error and "crash" in ev.error
    assert ev.score == float("-inf")        # so max() over candidates skips it
