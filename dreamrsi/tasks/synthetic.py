"""A cheap, fully offline discovery task.

The recursive loop in the paper is expensive because each node costs one coding
agent call plus one evaluation.  To make the *orchestration* reproducible
without any model access, this task poses a genuine multi-modal search problem
(shifted Rastrigin) whose structure has the same shape as the real ones:

* different roots land in different basins, so *width* matters;
* refining inside a basin pays off with diminishing returns, so *depth* matters;
* some basins are traps, so pruning on trajectory evidence matters.

A candidate is a JSON program ``{"x": [...]}``; the evaluator scores ``-f(x)``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from .base import EvalResult, Task

DIM = 4
BOUND = 5.12


def rastrigin(x: List[float]) -> float:
    return 10.0 * len(x) + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in x)


class SyntheticTask(Task):
    """Minimise a shifted Rastrigin function; score is the negated objective."""

    name = "synthetic"
    eval_program = "method.json"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.dim = int(self.config.get("dim", DIM))
        self.shift = float(self.config.get("shift", 0.35))

    def problem_statement(self) -> str:
        return (
            "# Synthetic discovery task\n\n"
            f"Find `x` in R^{self.dim}, each coordinate in [-{BOUND}, {BOUND}], that minimises\n"
            f"the shifted Rastrigin function f(x) = rastrigin(x - {self.shift}).\n\n"
            f"Write `{self.eval_program}` containing a JSON object with key `x`:\n"
            '    {"x": [0.1, -0.2, 0.0, 0.3]}\n\n'
            "The score reported by the evaluator is `-f(x)`; larger is better.\n"
        )

    def baseline_source(self) -> str:
        return json.dumps({"x": [2.0] * self.dim, "note": "baseline: fixed start point"}, indent=2) + "\n"

    def objective(self, x: List[float]) -> float:
        return rastrigin([v - self.shift for v in x])

    def evaluate(self, node_dir: str) -> EvalResult:
        path = os.path.join(node_dir, self.eval_program)
        if not os.path.exists(path):
            return EvalResult(evaluated=True, fail_class="compile_other",
                              error=f"missing {self.eval_program}")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            return EvalResult(evaluated=True, fail_class="compile_other",
                              error=f"program is not valid JSON: {exc}")
        x = payload.get("x")
        if not isinstance(x, list) or len(x) != self.dim:
            return EvalResult(evaluated=True, fail_class="runtime",
                              error=f"`x` must be a list of {self.dim} numbers")
        try:
            x = [float(v) for v in x]
        except (TypeError, ValueError) as exc:
            return EvalResult(evaluated=True, fail_class="runtime",
                              error=f"non-numeric coordinate: {exc}")
        n_total = self.dim
        n_valid = sum(1 for v in x if -BOUND <= v <= BOUND)
        if n_valid != n_total:
            return EvalResult(evaluated=True, fail_class="correctness", valid=False,
                              n_valid=n_valid, n_total=n_total,
                              error="one or more coordinates are out of bounds")
        return EvalResult(
            score=-self.objective(x),
            valid=True,
            fail_class="ok",
            n_valid=n_valid,
            n_total=n_total,
            diagnostics={"x": x, "objective": self.objective(x)},
        )
