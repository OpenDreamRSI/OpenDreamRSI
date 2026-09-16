"""Task evaluators: scoring protocol, failure classification, correctness checks."""

import json
import os
import shutil

import pytest

from dreamrsi.tasks import available, make_task
from dreamrsi.tasks.autocorrelation import evaluate_functional
from dreamrsi.tasks.sum_difference import gamma

DISCOVERED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "discovered", "lasso_path", "method.py")


def candidate(tmp_path, task, source, name=None):
    node = tmp_path / (name or "attempt")
    node.mkdir(parents=True, exist_ok=True)
    (node / task.eval_program).write_text(source)
    return str(node)


def test_registry_lists_every_task():
    assert set(available()) == {"synthetic", "sum_difference", "circle_packing",
                                "autocorrelation", "lasso", "kernelbench"}


# -- synthetic ---------------------------------------------------------------
def test_synthetic_prepare_writes_problem_and_scored_baseline(tmp_path):
    task = make_task("synthetic")
    paths = task.prepare(str(tmp_path))
    assert os.path.exists(paths["problem_file"])
    score = json.loads((tmp_path / "baseline" / "eval" / "score.json").read_text())
    assert score["fail_class"] == "ok" and score["score"] is not None
    assert task.baseline_score(str(tmp_path)) == pytest.approx(score["score"])


@pytest.mark.parametrize("source,fail_class", [
    ('{"x": [broken', "compile_other"),
    ('{"x": [0, 0]}', "runtime"),                 # wrong dimension
    ('{"x": [99.0, 0, 0, 0]}', "correctness"),    # out of bounds
])
def test_synthetic_classifies_failures(tmp_path, source, fail_class):
    task = make_task("synthetic")
    res = task.evaluate(candidate(tmp_path, task, source, name=fail_class))
    assert res.fail_class == fail_class
    assert res.error and not res.valid


def test_synthetic_scores_the_optimum_highest(tmp_path):
    task = make_task("synthetic")
    good = task.evaluate(candidate(tmp_path, task, json.dumps({"x": [task.shift] * 4}), "g"))
    worse = task.evaluate(candidate(tmp_path, task, json.dumps({"x": [3.0] * 4}), "w"))
    assert good.valid and good.score > worse.score
    assert good.score == pytest.approx(0.0)      # the global minimum of Rastrigin


# -- sum-difference ----------------------------------------------------------
def test_gamma_is_one_for_a_symmetric_set():
    assert gamma(list(range(12))) == pytest.approx(1.0)
    assert gamma([0]) == 0.0


def test_sum_difference_scores_a_program(tmp_path):
    task = make_task("sum_difference")
    src = 'import json\nprint(json.dumps({"A": [0, 1, 2, 4, 5, 9, 12, 13, 17, 20, 21, 22]}))\n'
    res = task.evaluate(candidate(tmp_path, task, src))
    assert res.valid and res.fail_class == "ok"
    assert res.score == pytest.approx(gamma([0, 1, 2, 4, 5, 9, 12, 13, 17, 20, 21, 22]))


def test_sum_difference_rejects_a_silent_program(tmp_path):
    task = make_task("sum_difference")
    res = task.evaluate(candidate(tmp_path, task, "pass\n"))
    assert res.fail_class == "correctness" and not res.valid


# -- circle packing ----------------------------------------------------------
def test_circle_packing_accepts_a_feasible_grid(tmp_path):
    task = make_task("circle_packing", {"n": 4})
    src = ('import json\n'
           'print(json.dumps({"n": 4, "centers": [[0.25,0.25],[0.25,0.75],[0.75,0.25],[0.75,0.75]],'
           ' "radii": [0.25,0.25,0.25,0.25]}))\n')
    res = task.evaluate(candidate(tmp_path, task, src))
    assert res.valid and res.score == pytest.approx(1.0)


def test_circle_packing_rejects_overlaps(tmp_path):
    task = make_task("circle_packing", {"n": 2})
    src = ('import json\n'
           'print(json.dumps({"n": 2, "centers": [[0.5,0.5],[0.5,0.5]], "radii": [0.4,0.4]}))\n')
    res = task.evaluate(candidate(tmp_path, task, src))
    assert res.fail_class == "correctness" and not res.valid
    assert res.n_valid < res.n_total


# -- autocorrelation ---------------------------------------------------------
def test_uniform_density_matches_the_analytic_autoconvolution():
    import numpy as np

    v, h = np.full(64, 2.0), 0.5 / 64
    assert evaluate_functional(v, 1, h) == pytest.approx(2.0, abs=1e-6)
    assert evaluate_functional(v, 2, h) == pytest.approx(2.0 / 3.0, abs=1e-2)


def test_autocorrelation_enforces_the_mass_constraint(tmp_path):
    task = make_task("autocorrelation", {"problem": 1})
    bad = 'import json\nprint(json.dumps({"values": [1.0] * 50}))\n'   # integral = 0.5
    res = task.evaluate(candidate(tmp_path, task, bad, "bad"))
    assert res.fail_class == "correctness" and "mass" in res.error

    ok = 'import json\nprint(json.dumps({"values": [2.0] * 50}))\n'
    res = task.evaluate(candidate(tmp_path, task, ok, "ok"))
    assert res.valid and res.score == pytest.approx(-2.0, abs=1e-6)   # minimisation


def test_autocorrelation_rejects_negative_values_when_f_must_be_nonnegative(tmp_path):
    task = make_task("autocorrelation", {"problem": 1})
    src = ('import json\n'
           'v = [2.0] * 50\nv[0] = -1.0\nv[1] = 5.0\n'
           'print(json.dumps({"values": v}))\n')
    res = task.evaluate(candidate(tmp_path, task, src))
    assert res.fail_class == "correctness" and "non-negative" in res.error


# -- lasso -------------------------------------------------------------------
LASSO_CFG = {"n_instances": 4, "n_timing": 2, "n_lambda": 10, "repeats": 1}


def test_lasso_python_route_scores_the_sklearn_baseline(tmp_path):
    task = make_task("lasso", LASSO_CFG)
    task.prepare(str(tmp_path))
    score = json.loads((tmp_path / "baseline" / "eval" / "score.json").read_text())
    assert score["fail_class"] == "ok" and score["score"] > 0
    assert score["diagnostics"]["route"] == "python"


def test_lasso_rejects_a_solver_that_fails_the_objective_check(tmp_path):
    task = make_task("lasso", LASSO_CFG)
    src = ("import numpy as np\n\n"
           "def solve_path(X, y, lambdas):\n"
           "    return np.zeros((len(lambdas), X.shape[1]))\n")   # all-zero is suboptimal
    res = task.evaluate(candidate(tmp_path, task, src))
    assert res.fail_class == "correctness" and res.score == 0.0
    assert "objective check failed" in res.error


def test_lasso_rejects_a_wrongly_shaped_answer(tmp_path):
    task = make_task("lasso", LASSO_CFG)
    src = ("import numpy as np\n\n"
           "def solve_path(X, y, lambdas):\n"
           "    return np.zeros((3, 3))\n")
    res = task.evaluate(candidate(tmp_path, task, src))
    assert res.fail_class == "correctness" and "shape" in res.error


def test_lasso_reports_a_missing_program(tmp_path):
    task = make_task("lasso", LASSO_CFG)
    (tmp_path / "empty").mkdir()
    res = task.evaluate(str(tmp_path / "empty"))
    assert res.fail_class == "compile_other"


# -- the program Dream-RSI discovered ---------------------------------------
def test_discovered_solver_is_a_well_formed_candidate():
    """The Appendix C listing, as extracted from the paper."""
    import ast

    source = open(DISCOVERED, encoding="utf-8").read()
    ast.parse(source)
    ns = {}
    exec(compile(source, DISCOVERED, "exec"), ns)          # noqa: S102 - our own file
    assert ns["COMPILE_FLAGS"] == ["-fopenmp", "-ffast-math"]
    cpp = ns["CPP_CODE"]
    assert cpp.count("{") == cpp.count("}")
    assert cpp.count("(") == cpp.count(")")
    assert "solve_active_set" in cpp and "int main()" in cpp
    assert len(source.splitlines()) == 847                 # the listing's own line count


@pytest.mark.skipif(not shutil.which("g++"), reason="needs a C++ compiler")
def test_discovered_solver_compiles_and_passes_the_objective_check(tmp_path):
    from dreamrsi.tasks.lasso import find_eigen

    if find_eigen() is None:
        pytest.skip("needs Eigen headers (brew install eigen / apt install libeigen3-dev)")
    task = make_task("lasso", {"n_instances": 4, "n_timing": 1, "n_lambda": 10, "repeats": 1})
    node = tmp_path / "discovered"
    node.mkdir()
    shutil.copyfile(DISCOVERED, node / task.eval_program)
    res = task.evaluate(str(node))
    assert res.fail_class == "ok", res.error
    assert res.valid and res.score > 0
    assert res.diagnostics["route"] == "native"
