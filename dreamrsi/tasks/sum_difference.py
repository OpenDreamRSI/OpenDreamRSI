"""Problem 2 of Appendix A -- the Sum-Difference Problem.

Find a finite set ``A`` in Z whose normalised sumset is large relative to its
normalised difference set::

    Gamma(A) = log(|A + A| / |A|) / log(|A - A| / |A|)

with ``A + A = {a + a' }`` and ``A - A = {a - a'}``.  The goal is to construct a
finite ``A`` maximising ``Gamma(A)``.  Larger is better, so the task score is
``Gamma(A)`` directly.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

from ._runner import run_python, split_failure
from .base import EvalResult, Task
from . import register

BASELINE = """import json

# Baseline: a short arithmetic-progression-like set.
A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(json.dumps({"A": A}))
"""


def gamma(a: List[int]) -> float:
    s = set(a)
    if len(s) < 2:
        return 0.0
    sums = {x + y for x in s for y in s}
    diffs = {x - y for x in s for y in s}
    n = len(s)
    num = math.log(len(sums) / n)
    den = math.log(len(diffs) / n)
    if den <= 0:
        return 0.0
    return num / den


class SumDifferenceTask(Task):
    name = "sum_difference"
    eval_program = "method.py"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.max_size = int(self.config.get("max_size", 400))
        self.timeout = float(self.config.get("timeout", 60.0))

    def problem_statement(self) -> str:
        return (
            "# Sum-Difference Problem\n\n"
            "Construct a finite set A subset of Z maximising\n\n"
            "    Gamma(A) = log(|A + A| / |A|) / log(|A - A| / |A|)\n\n"
            "where A + A = {a + a' : a, a' in A} and A - A = {a - a' : a, a' in A}.\n\n"
            f"Write `{self.eval_program}`; running it must print a JSON object to stdout:\n"
            '    {"A": [0, 1, 3, 7, ...]}\n\n'
            f"At most {self.max_size} distinct integers. Larger Gamma(A) is better.\n"
        )

    def baseline_source(self) -> str:
        return BASELINE

    def evaluate(self, node_dir: str) -> EvalResult:
        path = os.path.join(node_dir, self.eval_program)
        if not os.path.exists(path):
            return EvalResult(fail_class="compile_other", error=f"missing {self.eval_program}")
        payload, tag, seconds = run_python(path, timeout=self.timeout)
        cls, msg = split_failure(tag)
        if payload is None:
            return EvalResult(fail_class=cls, error=msg, diagnostics={"seconds": seconds})

        a = payload.get("A") if isinstance(payload, dict) else payload
        if not isinstance(a, list) or not a:
            return EvalResult(fail_class="correctness", error="`A` must be a non-empty list")
        try:
            a = [int(v) for v in a]
        except (TypeError, ValueError) as exc:
            return EvalResult(fail_class="correctness", error=f"non-integer element: {exc}")
        distinct = sorted(set(a))
        if len(distinct) > self.max_size:
            return EvalResult(fail_class="correctness", n_valid=self.max_size, n_total=len(distinct),
                              error=f"|A| = {len(distinct)} exceeds max_size={self.max_size}")
        value = gamma(distinct)
        return EvalResult(score=value, valid=True, fail_class="ok",
                          n_valid=len(distinct), n_total=len(distinct),
                          diagnostics={"gamma": value, "size": len(distinct), "seconds": seconds})


register("sum_difference", SumDifferenceTask)
