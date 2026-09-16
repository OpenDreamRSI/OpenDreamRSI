"""Problem 4 of Appendix A -- the three autocorrelation inequalities.

For an integrable ``f : R -> R`` the autoconvolution is

    (f * f)(t) = integral f(t - x) f(x) dx,   t in [-1/2, 1/2].

All three variants constrain ``integral_{-1/4}^{1/4} f = 1`` and differ in the
functional they optimise:

``1``  f >= 0 on [-1/4, 1/4];  minimise ``Phi_1(f) = max_t (f * f)(t)``
``2``  f >= 0 on [-1/4, 1/4];  maximise ``Phi_2(f) = ||f*f||_2^2 / (||f*f||_1 ||f*f||_inf)``
``3``  f possibly signed;      minimise ``Phi_3(f) = max_t |(f * f)(t)|``

A candidate discretises ``f`` as a piecewise-constant function on a uniform
partition of ``[-1/4, 1/4]``, which is the standard relaxation used by the
automated-discovery systems the paper compares against.  The task score negates
the two minimisation variants so that larger is always better.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from ._runner import run_python, split_failure
from .base import EvalResult, Task
from . import register

BASELINE = """import json

# Baseline: the uniform density on [-1/4, 1/4].
N = 50
values = [2.0] * N          # integral over an interval of length 1/2 equals 1
print(json.dumps({"problem": 1, "values": values}))
"""


def autoconvolution(values: np.ndarray, h: float) -> np.ndarray:
    """Piecewise-constant autoconvolution sampled on the induced grid."""
    return np.convolve(values, values) * h


def evaluate_functional(values: np.ndarray, problem: int, h: float) -> float:
    conv = autoconvolution(values, h)
    if problem == 1:
        return float(np.max(conv))
    if problem == 2:
        l1 = float(np.sum(np.abs(conv)) * h)
        l2sq = float(np.sum(conv ** 2) * h)
        linf = float(np.max(np.abs(conv)))
        if l1 <= 0 or linf <= 0:
            return 0.0
        return l2sq / (l1 * linf)
    if problem == 3:
        return float(np.max(np.abs(conv)))
    raise ValueError(f"problem must be 1, 2 or 3 (got {problem})")


class AutocorrelationTask(Task):
    name = "autocorrelation"
    eval_program = "method.py"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.problem = int(self.config.get("problem", 1))
        self.max_grid = int(self.config.get("max_grid", 4000))
        self.timeout = float(self.config.get("timeout", 120.0))
        self.mass_tol = float(self.config.get("mass_tol", 1e-6))

    @property
    def maximise(self) -> bool:
        return self.problem == 2

    def problem_statement(self) -> str:
        variant = {
            1: "minimise Phi_1(f) = max_{t in [-1/2,1/2]} (f * f)(t), with f >= 0",
            2: "maximise Phi_2(f) = ||f*f||_2^2 / (||f*f||_1 ||f*f||_inf), with f >= 0",
            3: "minimise Phi_3(f) = max_{t in [-1/2,1/2]} |(f * f)(t)|, f possibly signed",
        }[self.problem]
        return (
            f"# Autocorrelation inequality {self.problem}\n\n"
            "Let f be supported on [-1/4, 1/4] with integral_{-1/4}^{1/4} f(x) dx = 1.\n"
            f"Objective: {variant}.\n\n"
            f"Write `{self.eval_program}`; running it must print JSON to stdout:\n"
            '    {"problem": %d, "values": [v_1, ..., v_N]}\n\n'
            "`values` are the heights of a piecewise-constant f on a uniform partition of\n"
            f"[-1/4, 1/4] into N <= {self.max_grid} cells, so the mass constraint reads\n"
            "sum_i v_i * (1/2) / N = 1.\n" % self.problem
        )

    def baseline_source(self) -> str:
        return BASELINE.replace('"problem": 1', f'"problem": {self.problem}')

    def evaluate(self, node_dir: str) -> EvalResult:
        path = os.path.join(node_dir, self.eval_program)
        if not os.path.exists(path):
            return EvalResult(fail_class="compile_other", error=f"missing {self.eval_program}")
        payload, tag, seconds = run_python(path, timeout=self.timeout)
        cls, msg = split_failure(tag)
        if payload is None:
            return EvalResult(fail_class=cls, error=msg, diagnostics={"seconds": seconds})
        if not isinstance(payload, dict) or "values" not in payload:
            return EvalResult(fail_class="correctness", error='expected {"values": [...]}')

        raw = payload["values"]
        if not isinstance(raw, list) or not raw:
            return EvalResult(fail_class="correctness", error="`values` must be a non-empty list")
        if len(raw) > self.max_grid:
            return EvalResult(fail_class="correctness",
                              error=f"N = {len(raw)} exceeds max_grid={self.max_grid}")
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            return EvalResult(fail_class="correctness", error=f"non-numeric value: {exc}")
        if not np.all(np.isfinite(values)):
            return EvalResult(fail_class="correctness", error="non-finite value in `values`")

        n = values.size
        h = 0.5 / n
        mass = float(values.sum() * h)
        n_total = n + 1
        n_valid = n_total
        if abs(mass - 1.0) > self.mass_tol:
            return EvalResult(fail_class="correctness", valid=False,
                              n_valid=n_total - 1, n_total=n_total,
                              error=f"mass constraint violated: integral f = {mass:.9f}",
                              diagnostics={"mass": mass, "seconds": seconds})
        if self.problem in (1, 2):
            negative = int(np.sum(values < -1e-12))
            if negative:
                return EvalResult(fail_class="correctness", valid=False,
                                  n_valid=n_total - negative, n_total=n_total,
                                  error=f"{negative} negative value(s) but f must be non-negative",
                                  diagnostics={"seconds": seconds})

        phi = evaluate_functional(values, self.problem, h)
        score = phi if self.maximise else -phi
        return EvalResult(score=float(score), valid=True, fail_class="ok",
                          n_valid=n_valid, n_total=n_total,
                          diagnostics={"phi": phi, "grid": n, "mass": mass, "seconds": seconds})


register("autocorrelation", AutocorrelationTask)
