"""Task interface.

A task supplies three things the orchestration layer never looks inside:

* a **problem file** describing what to discover,
* a **baseline** artifact and its score (the floor a candidate must beat),
* an **evaluator** that turns a candidate directory into a score plus
  diagnostics, following a fixed task-scoring protocol where larger is better.

Dream-RSI leaves the discovery agent and the evaluator unchanged: only the
exploration-policy code is updated across recursive rounds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvalResult:
    """The evaluator's verdict on one attempt."""

    score: float = float("-inf")
    valid: bool = False
    evaluated: bool = True
    fail_class: str = "ok"
    error: Optional[str] = None
    n_valid: Optional[int] = None
    n_total: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": None if self.score == float("-inf") else self.score,
            "valid": self.valid,
            "evaluated": self.evaluated,
            "fail_class": self.fail_class,
            "error": self.error,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "diagnostics": self.diagnostics,
        }

    def write(self, node_dir: str) -> None:
        """Write ``eval/score.json`` (and ``error.txt``) as the prompts expect."""
        eval_dir = os.path.join(node_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "score.json"), "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
        if self.error:
            with open(os.path.join(node_dir, "error.txt"), "w", encoding="utf-8") as fh:
                fh.write(self.error.rstrip() + "\n")


class Task:
    """Base class for a discovery task."""

    name = "task"
    #: File name the discovery agent must write inside its own attempt dir.
    eval_program = "method.py"
    #: Higher is better after this transform; tasks that minimise negate inside
    #: :meth:`evaluate`.
    larger_is_better = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})

    # -- setup ---------------------------------------------------------------
    def problem_statement(self) -> str:
        raise NotImplementedError

    def baseline_source(self) -> str:
        """Source of the reference implementation placed in ``baseline/``."""
        raise NotImplementedError

    def prepare(self, run_dir: str) -> Dict[str, str]:
        """Materialise ``problem.md`` and ``baseline/`` under ``run_dir``."""
        os.makedirs(run_dir, exist_ok=True)
        problem_file = os.path.join(run_dir, "problem.md")
        with open(problem_file, "w", encoding="utf-8") as fh:
            fh.write(self.problem_statement())
        baseline_dir = os.path.join(run_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        with open(os.path.join(baseline_dir, self.eval_program), "w", encoding="utf-8") as fh:
            fh.write(self.baseline_source())
        result = self.evaluate(baseline_dir)
        result.write(baseline_dir)
        self._baseline_score = result.score
        return {"problem_file": problem_file, "baseline_dir": baseline_dir}

    def baseline_score(self, run_dir: Optional[str] = None) -> float:
        cached = getattr(self, "_baseline_score", None)
        if cached is not None:
            return float(cached)
        if run_dir is not None:
            path = os.path.join(run_dir, "baseline", "eval", "score.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    v = json.load(fh).get("score")
                if v is not None:
                    self._baseline_score = float(v)
                    return float(v)
        return 0.0

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, node_dir: str) -> EvalResult:
        raise NotImplementedError


def read_candidate(node_dir: str, filename: str) -> str:
    path = os.path.join(node_dir, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()
