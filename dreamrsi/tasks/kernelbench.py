"""GPU kernel engineering (KernelBench).

Candidate implementations are evaluated by execution performance, measured as
inverse runtime ``1 / ms``, subject to correctness checks against the reference
implementation -- the protocol used for the VGG16, LayerNorm, ConvDiv and
ConvMax results in Section 4.3.

This adapter deliberately does not vendor KernelBench.  It shells out to a
user-supplied evaluation command so the orchestration layer stays independent of
the CUDA toolchain, and it fails with an actionable message when the environment
cannot run kernels at all.

Config keys::

    level, problem      KernelBench task selector, passed to the command
    command             argv template; ``{node_dir}``, ``{program}``, ``{level}``
                        and ``{problem}`` are substituted
    timeout             seconds

The command must print a JSON object on stdout with at least
``{"correct": bool, "runtime_ms": float}``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from .base import EvalResult, Task
from . import register

BASELINE = '''"""Reference kernel: hand this to the evaluation command unchanged."""

REFERENCE = True
'''


class KernelBenchTask(Task):
    name = "kernelbench"
    eval_program = "kernel.py"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.level = self.config.get("level", 1)
        self.problem = self.config.get("problem", "LayerNorm")
        self.command = self.config.get("command")
        self.timeout = float(self.config.get("timeout", 900.0))

    def problem_statement(self) -> str:
        return (
            f"# KernelBench: level {self.level}, problem {self.problem}\n\n"
            "Write a numerically correct, high-performance GPU kernel. Candidates are\n"
            "scored by execution performance (1 / ms) subject to a correctness check\n"
            "against the reference implementation.\n\n"
            f"Write `{self.eval_program}`. It is evaluated by the configured KernelBench\n"
            "command; the command reports {\"correct\": bool, \"runtime_ms\": float}.\n"
        )

    def baseline_source(self) -> str:
        return BASELINE

    def _argv(self, node_dir: str) -> List[str]:
        if not self.command:
            raise RuntimeError(
                "kernelbench task needs task_config.command, e.g.\n"
                "  command: python -m kernelbench.eval --level {level} "
                "--problem {problem} --candidate {program} --json"
            )
        raw = self.command if isinstance(self.command, str) else " ".join(self.command)
        filled = raw.format(node_dir=node_dir,
                            program=os.path.join(node_dir, self.eval_program),
                            level=self.level, problem=self.problem)
        return shlex.split(filled)

    def evaluate(self, node_dir: str) -> EvalResult:
        program = os.path.join(node_dir, self.eval_program)
        if not os.path.exists(program):
            return EvalResult(fail_class="compile_other", error=f"missing {self.eval_program}")
        try:
            argv = self._argv(node_dir)
        except RuntimeError as exc:
            return EvalResult(evaluated=False, fail_class="env", error=str(exc))
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return EvalResult(fail_class="timeout", error=f"exceeded {self.timeout:g}s")
        except FileNotFoundError as exc:
            return EvalResult(evaluated=False, fail_class="env",
                              error=f"evaluation command not found: {exc}")
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-25:]
            cls = "compile_other" if "error:" in (proc.stderr or "").lower() else "runtime"
            return EvalResult(fail_class=cls, error="\n".join(tail))
        try:
            payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            return EvalResult(fail_class="runtime",
                              error=f"evaluation command did not print JSON: {exc}")

        if not payload.get("correct", False):
            return EvalResult(fail_class="correctness", valid=False, score=0.0,
                              n_valid=int(payload.get("n_valid", 0) or 0),
                              n_total=int(payload.get("n_total", 0) or 0),
                              error=payload.get("error") or "correctness check failed",
                              diagnostics=payload)
        ms = float(payload.get("runtime_ms", 0.0))
        if ms <= 0:
            return EvalResult(fail_class="runtime", error="non-positive runtime reported",
                              diagnostics=payload)
        return EvalResult(score=1000.0 / ms if payload.get("score_unit") == "per_s" else 1.0 / ms,
                          valid=True, fail_class="ok",
                          n_valid=int(payload.get("n_valid", 1) or 1),
                          n_total=int(payload.get("n_total", 1) or 1),
                          diagnostics=payload)


register("kernelbench", KernelBenchTask)
