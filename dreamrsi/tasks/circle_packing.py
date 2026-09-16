"""Problem 3 of Appendix A -- circle packing in a unit square.

For ``n`` in ``{26, 32}``, find centers ``(x_i, y_i)`` in ``[0, 1]^2`` and radii
``r_i >= 0`` with

    r_i <= x_i <= 1 - r_i,   r_i <= y_i <= 1 - r_i,
    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2   for i < j,

maximising ``sum_i r_i``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from ._runner import run_python, split_failure
from .base import EvalResult, Task
from . import register

BASELINE = """import json
import math

# Baseline: a regular k x k grid of equal circles (k^2 >= n), dropping extras.
n = 26
k = math.ceil(math.sqrt(n))
r = 0.5 / k
centers, radii = [], []
for i in range(k):
    for j in range(k):
        if len(centers) == n:
            break
        centers.append([(2 * i + 1) * r, (2 * j + 1) * r])
        radii.append(r)
print(json.dumps({"n": n, "centers": centers, "radii": radii}))
"""

#: Feasibility slack: candidates are allowed a tiny numerical margin.
TOL = 1e-9


class CirclePackingTask(Task):
    name = "circle_packing"
    eval_program = "method.py"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.n = int(self.config.get("n", 26))
        self.timeout = float(self.config.get("timeout", 120.0))
        self.tol = float(self.config.get("tol", TOL))

    def problem_statement(self) -> str:
        return (
            "# Circle packing in a unit square\n\n"
            f"Place n = {self.n} circles inside [0, 1]^2 with centers (x_i, y_i) and radii r_i >= 0\n"
            "such that every circle lies inside the square and no two circles overlap:\n\n"
            "    r_i <= x_i <= 1 - r_i,  r_i <= y_i <= 1 - r_i\n"
            "    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2,  i < j\n\n"
            "Maximise sum_i r_i.\n\n"
            f"Write `{self.eval_program}`; running it must print JSON to stdout:\n"
            '    {"n": 26, "centers": [[x, y], ...], "radii": [r, ...]}\n'
        )

    def baseline_source(self) -> str:
        return BASELINE.replace("n = 26", f"n = {self.n}")

    def _feasible(self, centers: List[Tuple[float, float]], radii: List[float]):
        violations = 0
        for (x, y), r in zip(centers, radii):
            if r < -self.tol:
                violations += 1
            elif not (r - self.tol <= x <= 1 - r + self.tol and r - self.tol <= y <= 1 - r + self.tol):
                violations += 1
        n = len(radii)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx * dx + dy * dy < (radii[i] + radii[j]) ** 2 - self.tol:
                    violations += 1
        return violations

    def evaluate(self, node_dir: str) -> EvalResult:
        path = os.path.join(node_dir, self.eval_program)
        if not os.path.exists(path):
            return EvalResult(fail_class="compile_other", error=f"missing {self.eval_program}")
        payload, tag, seconds = run_python(path, timeout=self.timeout)
        cls, msg = split_failure(tag)
        if payload is None:
            return EvalResult(fail_class=cls, error=msg, diagnostics={"seconds": seconds})
        if not isinstance(payload, dict):
            return EvalResult(fail_class="correctness", error="expected a JSON object")

        centers = payload.get("centers")
        radii = payload.get("radii")
        if (not isinstance(centers, list) or not isinstance(radii, list)
                or len(centers) != self.n or len(radii) != self.n):
            return EvalResult(fail_class="correctness",
                              error=f"expected exactly n = {self.n} centers and radii")
        try:
            centers = [(float(c[0]), float(c[1])) for c in centers]
            radii = [float(r) for r in radii]
        except (TypeError, ValueError, IndexError) as exc:
            return EvalResult(fail_class="correctness", error=f"malformed geometry: {exc}")
        if any(not math.isfinite(v) for v in radii) or any(
                not math.isfinite(c) for xy in centers for c in xy):
            return EvalResult(fail_class="correctness", error="non-finite coordinate or radius")

        violations = self._feasible(centers, radii)
        total_checks = self.n + self.n * (self.n - 1) // 2
        if violations:
            return EvalResult(fail_class="correctness", valid=False,
                              n_valid=total_checks - violations, n_total=total_checks,
                              error=f"{violations} constraint violation(s)",
                              diagnostics={"seconds": seconds})
        return EvalResult(score=float(sum(radii)), valid=True, fail_class="ok",
                          n_valid=total_checks, n_total=total_checks,
                          diagnostics={"sum_radii": sum(radii), "n": self.n, "seconds": seconds})


register("circle_packing", CirclePackingTask)
