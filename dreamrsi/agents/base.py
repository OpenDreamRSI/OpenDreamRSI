"""Agent interfaces.

Dream-RSI deliberately leaves two components fixed across the whole recursive
loop and changes only the exploration-policy code:

``DiscoveryAgent``
    the coding agent that, given a resumed workspace and the accumulated
    history, writes one new candidate (Appendix B.1 is its prompt).

``PolicyDeveloper``
    the LLM that rewrites the exploration policy from replay trajectories and
    scores (Appendix B.2 is its prompt).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass
class AttemptContext:
    """Everything a discovery agent may read for one ``CONTINUE`` call."""

    node_dir: str
    run_dir: str
    history_dir: str
    baseline_dir: str
    problem_file: str
    eval_program: str
    branch: int
    attempt: int
    cycle: int = 1
    parent_dir: Optional[str] = None
    sibling_dirs: Sequence[str] = ()
    direction_guidance: str = ""
    seed: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    def substitutions(self) -> Dict[str, str]:
        """Values for the ``$name`` variables in the exploration prompt."""
        return {
            "node_dir": self.node_dir,
            "history_dir": self.history_dir,
            "baseline_dir": self.baseline_dir,
            "eval_program": self.eval_program,
            "problem_file": self.problem_file,
            "direction_guidance": self.direction_guidance,
        }


class DiscoveryAgent:
    """Writes ``proposal.md`` and the eval program into ``ctx.node_dir``."""

    name = "discovery-agent"

    def propose(self, ctx: AttemptContext) -> None:
        raise NotImplementedError


@dataclass
class RevisionRequest:
    """One offline revision step for the policy-development agent."""

    method_file: str
    current_source: str
    history_dir: str
    trace_pool_dir: str
    feedback: Dict[str, Any] = field(default_factory=dict)
    revision_index: int = 0


class PolicyDeveloper:
    """Returns the full source of the next policy version."""

    name = "policy-developer"

    def revise(self, request: RevisionRequest) -> str:
        raise NotImplementedError


def write_proposal(node_dir: str, text: str) -> str:
    os.makedirs(node_dir, exist_ok=True)
    path = os.path.join(node_dir, "proposal.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")
    return path
