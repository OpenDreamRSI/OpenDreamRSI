"""On-disk layout of a run.

The directory names follow the paths the paper's prompts reference, so an agent
written against Appendix B can be pointed straight at a run::

    runs/<name>/
      history/
        baseline/                  the parallel-refine floor to beat
        r0000_init/method.py       pi_1
        r0001_dream/               round 1: attempts/, trace.json,
          attempts/attempt_b0a0/     proposal.md, method.*, eval/score.json
          live_cycle_manifest.json
          proposal_results/        beta_sweep.json, policy_execution_traces.jsonl
      trace_pool/
        iter1/live_cycle_manifest.json
        _current/live_cycle_manifest.json
      report.json
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict


def write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_default)
        fh.write("\n")
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default(obj: Any) -> Any:
    if obj == float("inf") or obj == float("-inf"):
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


@dataclass
class RunLayout:
    root: str

    @property
    def history_dir(self) -> str:
        return os.path.join(self.root, "history")

    @property
    def trace_pool_dir(self) -> str:
        return os.path.join(self.root, "trace_pool")

    def create(self) -> "RunLayout":
        for d in (self.root, self.history_dir, self.trace_pool_dir):
            os.makedirs(d, exist_ok=True)
        return self

    def round_dir(self, t: int, mode: str = "dream") -> str:
        d = os.path.join(self.history_dir, f"r{t:04d}_{mode}")
        os.makedirs(d, exist_ok=True)
        return d

    def publish_manifest(self, t: int, manifest: Dict[str, Any]) -> None:
        """Mirror the live manifest into ``trace_pool/`` for ``plan_grid``."""
        it = os.path.join(self.trace_pool_dir, f"iter{t}")
        os.makedirs(it, exist_ok=True)
        write_json(os.path.join(it, "live_cycle_manifest.json"), manifest)
        cur = os.path.join(self.trace_pool_dir, "_current")
        os.makedirs(cur, exist_ok=True)
        shutil.copyfile(os.path.join(it, "live_cycle_manifest.json"),
                        os.path.join(cur, "live_cycle_manifest.json"))
