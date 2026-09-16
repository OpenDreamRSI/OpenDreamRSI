"""The editable template handed to the policy-development agent.

The agent is told to edit *only* this file and to implement
``OptimalPolicy.solve`` and ``OptimalPolicy.plan_grid``.  The stub below is
deliberately weak: it is a legal, terminating policy, so a round always
produces something runnable, but it inherits the ``plan_grid`` warning and is
easily beaten.
"""

from __future__ import annotations

from typing import Optional

from see.policy.api import (  # the import path named in the paper's prompt
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    _budget_done,
    _record_curve,
    finalize_result,
)

NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    """Serial depth-first stub: open one root, refine it to exhaustion."""

    NAME = "OptimalPolicy"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.6))

    def solve(self, question, budget: Optional[int] = None) -> SimResult:
        question.reset()
        res = SimResult()
        while not _budget_done(question, budget):
            legal = question.legal_actions()
            if not legal:
                break
            roots = set(question.legal_roots())
            frontier = [c for c in legal if c not in roots]
            batch = frontier[:1] or legal[:1]
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        return context.bootstrap_plan("template stub: replace with an evidence-based plan")


SOURCE_PATH = __file__


def source() -> str:
    """The exact text a policy-development agent is handed to edit."""
    with open(SOURCE_PATH, "r", encoding="utf-8") as fh:
        return fh.read()
