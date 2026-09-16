"""Run a candidate program in a subprocess and parse its JSON output.

Candidate programs are written by a coding agent, so they are treated as
untrusted: separate process, hard wall-clock limit, no stdin, and the failure is
classified rather than raised.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple


def run_python(path: str, *, timeout: float = 60.0, cwd: Optional[str] = None,
               env: Optional[Dict[str, str]] = None) -> Tuple[Optional[Any], str, float]:
    """Execute ``path`` and parse its stdout as JSON.

    Returns ``(payload, fail_class, seconds)``; ``payload`` is ``None`` unless
    ``fail_class == "ok"``.  The error text is carried in the second element's
    companion by the caller, so this helper keeps it on the exception-free path.
    """
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=cwd or os.path.dirname(os.path.abspath(path)),
            env={**os.environ, **(env or {})},
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout::exceeded {timeout:.0f}s", time.time() - started
    except OSError as exc:
        return None, f"env::cannot launch interpreter: {exc}", time.time() - started

    elapsed = time.time() - started
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-20:]
        return None, "runtime::" + "\n".join(tail), elapsed
    out = (proc.stdout or "").strip()
    if not out:
        return None, "correctness::program produced no output on stdout", elapsed
    try:
        return json.loads(out), "ok::", elapsed
    except json.JSONDecodeError as exc:
        return None, f"correctness::stdout is not valid JSON: {exc}", elapsed


def split_failure(tag: str) -> Tuple[str, str]:
    cls, _, msg = tag.partition("::")
    return cls, msg
