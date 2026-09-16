"""The policy surface and the constraints the loader enforces."""

import pytest

from dreamrsi.policy.api import _budget_done, _record_curve, finalize_result, legal_batch
from dreamrsi.policy.loader import PolicyRejected, load_policy_source
from dreamrsi.policy.observation_signal import (
    branch_failed_hard,
    branch_promising,
    probe_improved_vs_baseline,
    probe_improved_vs_parent,
    successful_anchor,
)
from dreamrsi.tree import DiscoveryTree
from dreamrsi.types import ROOT_ID, Observation, SimResult, cell_id
from dreamrsi.world import ReplayWorld

GOOD = '''
from see.policy.api import (
    GridPlan, GridPlanningContext, LLMDesignedMethod, SimResult,
    _budget_done, _record_curve, finalize_result,
)

NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def __init__(self, config=None):
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.6))

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        while not _budget_done(question, budget):
            legal = question.legal_actions()
            if not legal:
                break
            batch = legal[: question.max_parallelism]
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context):
        return GridPlan(2, 2, "test")
'''


def world(branches=3, depth=3, **kw):
    tree = DiscoveryTree("w", baseline_score=0.0)
    for b in range(branches):
        parent = ROOT_ID
        for a in range(depth):
            parent = tree.continue_from(parent, score=float(b) + 0.1 * a,
                                        evaluated=True, valid=True).cell_id
    return ReplayWorld(tree, **kw)


def test_loads_and_runs_a_well_formed_policy():
    spec = load_policy_source(GOOD, name="good")
    assert spec.warnings == [] and spec.overrides_plan_grid
    w = world(max_parallelism=3)
    res = spec.instantiate({"beta": 0.5}).solve(w, None)
    assert res.total_probes == len(w.tree)
    assert res.best_score == pytest.approx(w.tree.ceiling)


def test_missing_plan_grid_is_reported():
    src = GOOD.replace("    def plan_grid(self, context):\n        return GridPlan(2, 2, \"test\")\n", "")
    spec = load_policy_source(src, name="nogrid")
    assert any("plan_grid" in w for w in spec.warnings)
    with pytest.raises(PolicyRejected):
        load_policy_source(src, name="nogrid", strict=True)


def test_reading_bookkeeping_inside_solve_is_flagged():
    src = GOOD.replace("            legal = question.legal_actions()",
                       "            _ = question.best_so_far\n"
                       "            legal = question.legal_actions()")
    spec = load_policy_source(src, name="peek")
    assert any("best_so_far" in w for w in spec.warnings)


def test_a_module_without_optimalpolicy_is_rejected():
    with pytest.raises(PolicyRejected):
        load_policy_source("NAME = 'x'\n", name="empty")


def test_plan_grid_returning_none_is_flagged():
    src = GOOD.replace('        return GridPlan(2, 2, "test")', "        return None")
    spec = load_policy_source(src, name="none")
    assert any("non-None" in w for w in spec.warnings)


def test_budget_done_respects_both_caps():
    w = world(max_parallelism=2)
    assert not _budget_done(w, None)
    w.max_decision_rounds = 0
    assert _budget_done(w, None)
    w.max_decision_rounds = 99
    w.probe_batch([cell_id(0, 0), cell_id(1, 0)])
    assert _budget_done(w, budget=2)
    assert not _budget_done(w, budget=3)


def test_record_curve_and_finalize_track_the_environment():
    w = world(max_parallelism=2)
    res = SimResult()
    w.probe_batch([cell_id(0, 0)], on_reveal=lambda _: _record_curve(res, w))
    out = finalize_result(w, res)
    assert out.curve[0][0] == 1
    assert out.revealed == [cell_id(0, 0)]
    assert out.stopped_reason == "empty_batch"


def test_legal_batch_filters_to_something_the_world_accepts():
    w = world(max_parallelism=2)
    batch = legal_batch(w, [cell_id(0, 1), cell_id(0, 0), cell_id(1, 0), cell_id(2, 0)])
    assert batch == [cell_id(0, 0), cell_id(1, 0)]
    assert w.probe_batch(batch)


# -- observation signals -----------------------------------------------------
def obs(**kw):
    base = dict(cell_id="b0a0", branch=0, attempt=0, score=1.0, evaluated=True,
                valid=True, fail_class="ok", error=None)
    base.update(kw)
    return Observation(**base)


def test_success_semantics_ignore_valid_and_n_valid():
    o = obs(valid=False, n_valid=None, n_total=None)
    assert o.is_success                      # error is None and fail_class == "ok"
    assert not obs(fail_class="runtime", error="boom").is_success


def test_improvement_helpers():
    assert probe_improved_vs_parent(obs(delta_vs_parent=0.5))
    assert not probe_improved_vs_parent(obs(delta_vs_parent=-0.5))
    assert probe_improved_vs_baseline(obs(delta_vs_baseline=0.2))
    assert not probe_improved_vs_baseline(obs(fail_class="runtime", delta_vs_baseline=9.0))


def test_a_later_success_cancels_an_earlier_closure():
    prefix = {
        "b0a0": obs(cell_id="b0a0", attempt=0, fail_class="timeout", error="t", valid=False),
        "b0a1": obs(cell_id="b0a1", attempt=1, score=3.0),
    }
    assert not branch_failed_hard(prefix, 0)
    assert successful_anchor(prefix, 0).cell_id == "b0a1"


def test_repairable_failures_do_not_close_a_branch():
    prefix = {"b0a0": obs(fail_class="compile_other", error="nope", valid=False)}
    assert not branch_failed_hard(prefix, 0)
    assert branch_failed_hard(obs(fail_class="timeout", error="t")) is True


def test_shallow_weak_scores_do_not_discard_a_branch():
    prefix = {"b0a0": obs(score=-5.0, delta_vs_baseline=-5.0)}
    assert branch_promising(prefix, 0, baseline=0.0)


def test_an_observation_also_answers_to_mapping_access():
    """`observed()` is a dict *of* Observations, and agents read it as a dict of dicts.

    Measured: four of six agent-written revisions in one four-round run crashed
    on `obs.get("branch")`, each after a full batch had already been revealed.
    Supporting the spelling costs four lines and turns a fatal misreading into a
    harmless one.
    """
    obs = Observation(cell_id="b1a2", branch=1, attempt=2, score=-3.5,
                      fail_class="runtime", error="boom")
    assert obs.get("branch") == 1 == obs["branch"]
    assert obs.get("fail_class") == "runtime"
    assert obs.get("nope") is None
    assert obs.get("nope", "fallback") == "fallback"
    with pytest.raises(KeyError):
        obs["nope"]


def test_mapping_access_is_a_pure_alias_not_a_second_semantics():
    """`get("score")` must be `.score`, not `to_dict()`'s JSON-safe substitute."""
    missing = Observation(cell_id="b0a0", branch=0, attempt=0)
    assert missing.score == float("-inf")
    assert missing.get("score") == float("-inf")
    assert missing["score"] == float("-inf")
    assert missing.to_dict()["score"] is None        # only JSON gets the None


def test_a_policy_that_treats_observations_as_dicts_now_survives():
    """The exact shape of the revisions that died: read the prefix, then branch."""
    src = '''
from see.policy.api import (LLMDesignedMethod, GridPlan, SimResult,
                            _budget_done, _record_curve, finalize_result)
NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        while not _budget_done(question, budget):
            prefix = question.observed()
            for cell, obs in prefix.items():          # the dict-of-dicts reading
                if obs.get("branch") is None or obs.get("fail_class") != "ok":
                    continue
            batch = question.legal_actions()[:question.max_parallelism]
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context):
        return GridPlan(2, 2, "x")
'''
    spec = load_policy_source(src, name="dictish")
    replay = world(branches=2, depth=2, max_parallelism=2, max_decision_rounds=8)
    result = spec.instantiate({"beta": 0.5}).solve(replay, None)
    assert result.total_probes > 0
    assert "crash" not in result.stopped_reason
