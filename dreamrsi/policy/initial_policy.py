"""pi_1: the manually designed exploration policy both systems start from.

Section 4 of the paper: "This exploration policy follows a simple *parallel
refining* strategy: it launches multiple independent exploration workspaces in
parallel, with each workspace maintaining its own local discovery trajectory and
repeatedly refining its current candidate based on the history accumulated
within that workspace."

Recursive Fixed Exploration keeps this policy for every recursive round.
Dream-RSI deploys the same policy in round 1 and then improves it by dreaming,
which is why the two methods have identical round-1 search behaviour.

It is written in the same shape as an agent-authored revision -- module-level
``NAME``, ``class OptimalPolicy(LLMDesignedMethod)``, imports from
``see.policy.api`` -- so it travels through exactly the same load-and-check path.
"""

from __future__ import annotations

from typing import Optional

from see.policy.api import (
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
    """Open ``W`` workspaces, then refine all of them every round until exhausted.

    It never prunes, never re-allocates effort between workspaces and never stops
    early -- the fixed strategy the paper contrasts against.  ``beta`` is read
    for interface parity but, by construction, routes no threshold.
    """

    NAME = "OptimalPolicy"

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.beta = float(self.config.get("beta", 1.0))
        self.workers = self.config.get("workers")

    def solve(self, question, budget: Optional[int] = None) -> SimResult:
        question.reset()
        res = SimResult()
        width = int(self.workers or question.max_parallelism)

        while not _budget_done(question, budget):
            opened = question.opened_branches()
            batch = []
            if len(opened) < width:
                # launch the remaining independent workspaces
                batch = question.legal_roots()[: width - len(opened)]
            if not batch:
                # refine every open workspace once, in parallel
                roots = set(question.legal_roots())
                batch = [c for c in sorted(question.legal_actions()) if c not in roots]
            batch = batch[: question.max_parallelism]
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))

        return finalize_result(question, res)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        """A fixed policy also keeps a fixed grid: always the configured default."""
        return GridPlan(
            branch_count=max(1, min(context.fallback_branch_count, context.hard_max_branch_count)),
            refine_count=max(0, min(context.fallback_refine_count, context.hard_max_refine_count)),
            reason="fixed exploration: grid held at the round-1 configuration",
        )
