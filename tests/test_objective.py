"""The two replay objectives."""

import pytest

from dreamrsi.objective import (
    attainment,
    attainment_auc,
    envelope_auc,
    parallel_penalty,
    replay_value,
    summarize_sweep,
)
from dreamrsi.types import SimResult, SweepPoint


def result(curve, probes, rounds, seq, best):
    return SimResult(curve=curve, total_probes=probes, decision_rounds=rounds,
                     effective_sequential_rounds=seq, best_score=best)


def test_equation_one_decomposition():
    """V = max score - beta1 * N + beta2 * N / max(1, k)."""
    r = result([(4, 1.0)], probes=8, rounds=2, seq=2, best=1.0)
    v = replay_value(r, beta1=0.01, beta2=0.1, quality=1.0)
    assert v.discovery_quality == pytest.approx(1.0)
    assert v.execution_cost == pytest.approx(0.08)
    assert v.parallelism_bonus == pytest.approx(0.1 * 8 / 2)
    assert v.value == pytest.approx(1.0 - 0.08 + 0.4)


def test_parallelism_bonus_rewards_batching():
    serial = replay_value(result([], 8, 8, 8, 1.0), quality=1.0)
    batched = replay_value(result([], 8, 2, 2, 1.0), quality=1.0)
    assert batched.value > serial.value


def test_attainment_is_normalised_between_baseline_and_ceiling():
    assert attainment(0.0, 0.0, 1.0) == pytest.approx(0.0)
    assert attainment(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert attainment(0.25, 0.0, 1.0) == pytest.approx(0.25)
    assert attainment(2.0, 0.0, 1.0) == pytest.approx(1.0)      # clipped
    assert attainment(float("-inf"), 0.0, 1.0) == 0.0
    assert attainment(5.0, 5.0, 5.0) == pytest.approx(1.0)      # degenerate span


def test_auc_prefers_reaching_attainment_early():
    early = result([(2, 1.0)], 2, 1, 1, 1.0)
    late = result([(9, 1.0)], 9, 9, 9, 1.0)
    a = attainment_auc(early, baseline=0.0, ceiling=1.0, horizon=10)
    b = attainment_auc(late, baseline=0.0, ceiling=1.0, horizon=10)
    assert a > b


def test_auc_holds_flat_after_the_policy_stops():
    r = result([(2, 0.5)], 2, 1, 1, 0.5)
    assert attainment_auc(r, baseline=0.0, ceiling=1.0, horizon=10) == pytest.approx(0.5 * 9 / 10)


def test_parallel_penalty_spans_serial_to_full_batches():
    assert parallel_penalty(result([], 8, 8, 8, 1.0)) == pytest.approx(1.0)
    assert parallel_penalty(result([], 8, 2, 2, 1.0)) == pytest.approx(0.25)
    assert parallel_penalty(result([], 0, 0, 0, 1.0)) == pytest.approx(1.0)


def test_envelope_is_the_upper_hull_of_the_sweep():
    cheap = [(2, 0.6)]        # low beta: a little attainment, very few probes
    thorough = [(8, 1.0)]     # high beta: full attainment, more probes
    env = envelope_auc([cheap, thorough], horizon=10)
    assert env > attainment_auc(result(cheap, 2, 1, 1, 0.6), baseline=0.0, ceiling=1.0, horizon=10)
    assert env > attainment_auc(result(thorough, 8, 1, 1, 1.0), baseline=0.0, ceiling=1.0, horizon=10)


def test_envelope_of_a_single_curve_equals_its_own_auc():
    curve = [(3, 0.4), (6, 0.9)]
    r = result(curve, 6, 2, 2, 0.9)
    assert envelope_auc([curve], 10) == pytest.approx(
        attainment_auc(r, baseline=0.0, ceiling=1.0, horizon=10))


def _point(beta, auc, att, probes, pen):
    return SweepPoint(world_id="w", beta=beta, auc=auc, attainment=att, total_probes=probes,
                      decision_rounds=1, effective_sequential_rounds=1,
                      parallel_penalty=pen, best_score=att, replay_value=auc)


def test_reward_subtracts_lambda_times_the_mean_penalty():
    pts = [_point(0.0, 0.5, 0.5, 10, 0.5), _point(1.0, 0.9, 1.0, 20, 0.3)]
    s = summarize_sweep(pts, lam=0.25, frontier_auc=0.8)
    assert s.auc == pytest.approx(0.8)
    assert s.parallel_penalty == pytest.approx(0.4)
    assert s.reward == pytest.approx(0.8 - 0.25 * 0.4)


def test_degenerate_sweeps_are_flagged():
    flat = [_point(b, 0.5, 0.5, 10, 0.5) for b in (0.0, 0.5, 1.0)]
    assert summarize_sweep(flat).non_degenerate is False
    varied = [_point(0.0, 0.4, 0.4, 5, 0.5), _point(1.0, 0.9, 0.9, 20, 0.3)]
    assert summarize_sweep(varied).non_degenerate is True


def test_empty_sweep_is_the_worst_possible_score():
    s = summarize_sweep([])
    assert s.reward < 0
