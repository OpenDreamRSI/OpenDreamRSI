"""Reading a finished comparison back off disk.

The interesting cases are the ones where a bare score delta would mislead: an
arm that ran fewer rounds, and a gap smaller than the noise the two arms show
on the round where they are running the *same* policy.
"""

import json
import os

import pytest

from dreamrsi.experiments.arms import (
    ArmComparison,
    compare_arms,
    fail_classes,
    load_arm,
    render,
    round_dirs,
)


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def make_run(root, name, *, mode, bests, calls_per_round=9, phases=None,
             fails=None):
    """A minimal run directory of the shape ``run_rsi`` leaves behind."""
    run_dir = os.path.join(str(root), name)
    rounds = []
    for i, best in enumerate(bests):
        rd = f"r{i + 1:04d}_{mode}"
        rounds.append({
            "round": i + 1,
            "manifest": {"best_score": best, "probes": 9, "decision_rounds": 3},
            "cumulative_agent_calls": calls_per_round * (i + 1),
            "dream": (phases or {}).get(i + 1, {}).get("dream"),
        })
        nodes = [{"fail_class": fc}
                 for fc, n in ((fails or {}).get(i + 1, {"ok": 9})).items()
                 for _ in range(n)]
        _write(os.path.join(run_dir, "history", rd, "trace.json"), {"nodes": nodes})
        sweep = (phases or {}).get(i + 1, {}).get("sweep")
        if sweep:
            _write(os.path.join(run_dir, "history", rd, "proposal_results",
                                "beta_sweep.json"), sweep)
    _write(os.path.join(run_dir, "report.json"), {
        "config": {"mode": mode, "task": "synthetic", "rounds": len(bests),
                   "revisions": 2, "branch_count": 3, "refine_count": 2,
                   "max_parallelism": 3, "max_decision_rounds": 8,
                   "engine": "native", "seed": 0},
        "baseline_score": 1.0,
        "rounds": rounds,
        "best_score": max(b for b in bests if b is not None),
        "cumulative_agent_calls": calls_per_round * len(bests),
        "seconds": 60.0 * len(bests),
    })
    return run_dir


def test_an_arm_is_reconstructed_from_its_run_directory(tmp_path):
    run = make_run(tmp_path, "d", mode="dream", bests=[1.5, 1.8, 1.9])
    arm = load_arm(run)
    assert arm.mode == "dream"
    assert [r.best_score for r in arm.rounds] == [1.5, 1.8, 1.9]
    assert [r.cumulative_agent_calls for r in arm.rounds] == [9, 18, 27]
    assert arm.best_score == 1.9


def test_round_dirs_skips_the_seed_directory(tmp_path):
    run = make_run(tmp_path, "d", mode="dream", bests=[1.5])
    os.makedirs(os.path.join(run, "history", "r0000_init"))
    assert round_dirs(run) == ["r0001_dream"]


def test_fail_classes_counts_what_the_evaluator_recorded(tmp_path):
    run = make_run(tmp_path, "d", mode="dream", bests=[1.5],
                   fails={1: {"ok": 6, "timeout": 2, "correctness": 1}})
    assert fail_classes(run, "r0001_dream") == {"ok": 6, "timeout": 2,
                                                "correctness": 1}


def test_a_missing_report_is_an_error_not_an_empty_arm(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_arm(str(tmp_path / "nothing"))


# -- the two things a score delta hides --------------------------------------
def test_an_unmatched_budget_is_reported_not_averaged_over(tmp_path):
    dream = make_run(tmp_path, "d", mode="dream", bests=[1.5, 1.8, 1.9])
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.4])
    cmp = compare_arms(dream, fixed)
    assert cmp.budget_matched is False
    assert "WARNING: budgets differ" in render(cmp)


def test_round_one_is_the_noise_floor_because_both_arms_run_one_policy(tmp_path):
    # Round 1 differs by 0.30; the final gap is 0.10.  The arms cannot be
    # distinguished, and saying so is the whole point of the column.
    dream = make_run(tmp_path, "d", mode="dream", bests=[1.5, 1.6])
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.2, 1.5])
    cmp = compare_arms(dream, fixed)
    assert cmp.noise_floor == pytest.approx(0.30)
    assert cmp.final_gap == pytest.approx(0.10)
    assert cmp.exceeds_noise is False
    assert "does not clear the noise floor" in render(cmp)


def test_a_gap_that_clears_the_noise_floor_is_said_to_clear_it(tmp_path):
    dream = make_run(tmp_path, "d", mode="dream", bests=[1.5, 2.5])
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.49, 1.5])
    cmp = compare_arms(dream, fixed)
    assert cmp.exceeds_noise is True
    out = render(cmp)
    assert "exceeds the round-1 noise" in out
    assert "still n=1 per arm" in out


def test_a_missing_round_one_score_leaves_the_floor_unknown(tmp_path):
    dream = make_run(tmp_path, "d", mode="dream", bests=[None, 1.6])
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.2, 1.5])
    cmp = compare_arms(dream, fixed)
    assert cmp.noise_floor is None
    assert cmp.exceeds_noise is None
    render(cmp)  # must not raise


# -- dreaming phases ---------------------------------------------------------
def _phase(*, selected, improved, incumbent, versions):
    return {"dream": {"selected_version": selected, "improved": improved,
                      "incumbent_reward": incumbent,
                      "selected_reward": versions[selected].get("score")},
            "sweep": {"selected_version": selected, "versions": versions}}


def test_a_phase_that_selected_the_incumbent_says_why_it_could(tmp_path):
    crashed = {"version": 1, "score": -0.25, "error": "KeyError: 'branch'",
               "pareto": {"auc": 0.0, "attainment": 0.0, "mean_probes": 0.0,
                          "non_degenerate": False}}
    incumbent = {"version": 0, "score": 0.64, "error": None,
                 "pareto": {"auc": 0.72, "attainment": 1.0, "mean_probes": 9.0,
                            "non_degenerate": False}}
    dream = make_run(tmp_path, "d", mode="dream", bests=[1.5],
                     phases={1: _phase(selected=0, improved=False, incumbent=0.64,
                                       versions=[incumbent, crashed])})
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.4])
    cmp = compare_arms(dream, fixed)
    assert len(cmp.dream.phases) == 1
    assert cmp.dream.phases[0].crashed == 1
    out = render(cmp)
    assert "(1 crashed)" in out
    assert "CRASHED: KeyError" in out
    # every version was beta-insensitive, so the note has to fire
    assert "no version responded to beta" in out


def test_a_beta_sensitive_version_suppresses_the_degeneracy_note(tmp_path):
    versions = [
        {"version": 0, "score": 0.64, "error": None,
         "pareto": {"auc": 0.72, "attainment": 1.0, "mean_probes": 9.0,
                    "non_degenerate": False}},
        {"version": 1, "score": 0.70, "error": None,
         "pareto": {"auc": 0.80, "attainment": 0.9, "mean_probes": 5.0,
                    "non_degenerate": True}},
    ]
    dream = make_run(tmp_path, "d", mode="dream", bests=[1.5],
                     phases={1: _phase(selected=1, improved=True, incumbent=0.64,
                                       versions=versions)})
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.4])
    out = render(compare_arms(dream, fixed))
    assert "improved=True" in out
    assert "no version responded to beta" not in out


def test_an_arm_with_no_dreaming_phase_renders(tmp_path):
    dream = make_run(tmp_path, "d", mode="fixed", bests=[1.5, 1.6])
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.4, 1.5])
    out = render(compare_arms(dream, fixed))
    assert "none recorded" in out


def test_the_comparison_round_trips_through_json(tmp_path):
    dream = make_run(tmp_path, "d", mode="dream", bests=[1.5, 1.6])
    fixed = make_run(tmp_path, "f", mode="fixed", bests=[1.2, 1.5])
    payload = compare_arms(dream, fixed).to_dict()
    assert json.loads(json.dumps(payload))["noise_floor"] == pytest.approx(0.30)
