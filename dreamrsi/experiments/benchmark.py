"""Does the effect survive more than one seed?

The single run in the README is an anecdote.  Both arms share a discovery agent,
an evaluator, an initialisation and a per-round budget, and differ only in
whether the exploration policy may improve -- but they also share a seed, and a
mock hill-climber on a multi-modal landscape is noisy enough that one seed can
show almost anything.  This module runs the controlled comparison of Section 4
across many seeds and reports it as a *paired* experiment, because the pairing
is the point: for a given seed the two arms see the same task, the same agent
and the same round-1 policy, so the within-pair difference isolates dreaming.

Reported per metric: the mean and median difference, a bootstrap confidence
interval over seeds, and the win/tie/loss count.  Two metrics matter and they
are not interchangeable:

* **best score** -- the quality actually discovered.  Dream-RSI's claim is that
  it is *not* paid for, so this should come out a tie.
* **discovery-agent calls** -- what quality cost.  This is where an improved
  exploration policy is supposed to show up.

A method that wins on calls while tying on quality is doing what the paper says.
One that wins on calls by losing quality is just a smaller budget.

:func:`ablate_pool_window` runs the same harness with the simulator pool pinned
to a sliding window, which turns off the "evolving worlds" half of the paper's
title: the pool stops growing, and dreaming only ever sees recent history.

No statistical test is applied.  With a mock agent on a synthetic task the
sampling distribution is a property of this repository's stand-ins, not of
Dream-RSI, so a p-value here would be precision about the wrong population.  The
intervals are there to show spread, not to certify an effect.
"""

from __future__ import annotations

import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..rsi import DREAM, FIXED, RSIConfig, build_components, run_rsi

#: Resamples used for the bootstrap interval over seeds.
BOOTSTRAP_RESAMPLES = 2000


@dataclass
class SeedRow:
    """One seed's paired outcome, at equal rounds and (optionally) equal budget."""

    seed: int
    fixed_best: Optional[float]
    dream_best: Optional[float]
    fixed_calls: int
    dream_calls: int
    #: Dream-RSI run on until it has spent as many calls as the fixed arm.
    matched_best: Optional[float] = None
    matched_calls: Optional[int] = None
    matched_rounds: Optional[int] = None

    @property
    def best_delta(self) -> Optional[float]:
        """``dream - fixed`` after the same number of rounds; positive favours dreaming."""
        if self.fixed_best is None or self.dream_best is None:
            return None
        return self.dream_best - self.fixed_best

    @property
    def matched_delta(self) -> Optional[float]:
        """``dream - fixed`` after the same number of *calls*.

        This is the comparison that answers the paper's claim.  At equal rounds
        an improved policy can look worse simply by choosing to stop sooner, and
        spending less is not the same as exploring better.
        """
        if self.fixed_best is None or self.matched_best is None:
            return None
        return self.matched_best - self.fixed_best

    @property
    def call_ratio(self) -> Optional[float]:
        """``fixed / dream``; ``1.6`` means Dream-RSI used 1.6x fewer calls."""
        if not self.dream_calls:
            return None
        return self.fixed_calls / self.dream_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "fixed_best": self.fixed_best,
            "dream_best": self.dream_best,
            "best_delta": self.best_delta,
            "fixed_calls": self.fixed_calls,
            "dream_calls": self.dream_calls,
            "call_ratio": self.call_ratio,
            "matched_best": self.matched_best,
            "matched_delta": self.matched_delta,
            "matched_calls": self.matched_calls,
            "matched_rounds": self.matched_rounds,
        }


@dataclass
class SeedComparison:
    """Paired fixed-vs-dream comparison aggregated over seeds."""

    rows: List[SeedRow] = field(default_factory=list)
    quality_tolerance: float = 1e-9

    # -- quality -------------------------------------------------------------
    @property
    def best_deltas(self) -> List[float]:
        return [r.best_delta for r in self.rows if r.best_delta is not None]

    @property
    def matched_deltas(self) -> List[float]:
        return [r.matched_delta for r in self.rows if r.matched_delta is not None]

    def record(self, deltas: Sequence[float]) -> Dict[str, int]:
        """Win / tie / loss on discovered quality, from Dream-RSI's side."""
        w = t = l = 0
        for d in deltas:
            if d > self.quality_tolerance:
                w += 1
            elif d < -self.quality_tolerance:
                l += 1
            else:
                t += 1
        return {"win": w, "tie": t, "loss": l}

    @property
    def quality_record(self) -> Dict[str, int]:
        return self.record(self.best_deltas)

    # -- cost ----------------------------------------------------------------
    @property
    def call_ratios(self) -> List[float]:
        return [r.call_ratio for r in self.rows if r.call_ratio is not None]

    @property
    def cheaper_seeds(self) -> int:
        return sum(1 for r in self.rows if r.dream_calls < r.fixed_calls)

    def summary(self) -> Dict[str, Any]:
        return {
            "seeds": len(self.rows),
            "quality": {
                "mean_delta": _mean(self.best_deltas),
                "median_delta": _median(self.best_deltas),
                "ci95_delta": bootstrap_ci(self.best_deltas),
                "record": self.quality_record,
            },
            "quality_at_matched_budget": {
                "seeds": len(self.matched_deltas),
                "mean_delta": _mean(self.matched_deltas),
                "median_delta": _median(self.matched_deltas),
                "ci95_delta": bootstrap_ci(self.matched_deltas),
                "record": self.record(self.matched_deltas),
                "mean_calls": _mean([float(r.matched_calls) for r in self.rows
                                     if r.matched_calls is not None]),
            },
            "cost": {
                "mean_call_ratio": _mean(self.call_ratios),
                "median_call_ratio": _median(self.call_ratios),
                "ci95_call_ratio": bootstrap_ci(self.call_ratios),
                "cheaper_seeds": self.cheaper_seeds,
                "mean_fixed_calls": _mean([float(r.fixed_calls) for r in self.rows]),
                "mean_dream_calls": _mean([float(r.dream_calls) for r in self.rows]),
            },
            "per_seed": [r.to_dict() for r in self.rows],
        }


@dataclass
class AblationRow:
    """One ablation arm, summarised the same way as the main comparison."""

    label: str
    pool_window: Optional[int]
    comparison: SeedComparison

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "pool_window": self.pool_window,
                **self.comparison.summary()}


def run_arm(config: RSIConfig, log: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    """Build the task, agents and developer for one arm and run the loop."""
    task, agent, developer = build_components(config)
    kwargs: Dict[str, Any] = {}
    if log is not None:
        kwargs["log"] = log
    return run_rsi(config, task=task, agent=agent, developer=developer, **kwargs)


def compare_across_seeds(base: RSIConfig, seeds: Sequence[int], *,
                         pool_window: Optional[int] = None,
                         equal_budget_rounds: Optional[int] = None,
                         out_dir: Optional[str] = None,
                         label: str = "",
                         log: Optional[Callable[..., None]] = None) -> SeedComparison:
    """Run both arms once per seed and pair the results.

    Everything except ``mode`` and ``seed`` is held fixed, so a row's difference
    is attributable to dreaming rather than to the budget or the agent.

    ``equal_budget_rounds`` adds the comparison that equal *rounds* cannot make.
    An improved exploration policy typically stops earlier, so after the same
    number of rounds it has spent fewer discovery-agent calls -- and a lower
    score then says nothing, because the two arms did different amounts of work.
    Setting this runs the dreaming arm for that many rounds instead and reads off
    its best score at the moment its cumulative calls reach the fixed arm's
    total, which puts both arms on the same budget.
    """
    rows: List[SeedRow] = []
    for seed in seeds:
        outcome: Dict[str, Dict[str, Any]] = {}
        for mode in (FIXED, DREAM):
            name = "-".join(p for p in (base.name, label, mode, f"s{seed}") if p)
            rounds = (int(equal_budget_rounds)
                      if mode == DREAM and equal_budget_rounds else base.rounds)
            config = RSIConfig.from_dict({
                **base.__dict__,
                "mode": mode,
                "seed": int(seed),
                "name": name,
                "rounds": rounds,
                "pool_window": pool_window,
                "out_dir": out_dir or base.out_dir,
            })
            outcome[mode] = run_arm(config, log=log)

        fixed_calls = int(outcome[FIXED]["cumulative_agent_calls"])
        # At equal rounds: read the dreaming arm at the fixed arm's round count,
        # so a longer dreaming run does not quietly become a bigger one.
        at_round = _through_round(outcome[DREAM], base.rounds)
        matched = (_through_calls(outcome[DREAM], fixed_calls)
                   if equal_budget_rounds else None)
        rows.append(SeedRow(
            seed=int(seed),
            fixed_best=_finite(outcome[FIXED]["best_score"]),
            dream_best=_finite(at_round["best_score"]),
            fixed_calls=fixed_calls,
            dream_calls=int(at_round["calls"]),
            matched_best=None if matched is None else _finite(matched["best_score"]),
            matched_calls=None if matched is None else int(matched["calls"]),
            matched_rounds=None if matched is None else int(matched["rounds"]),
        ))
    return SeedComparison(rows=rows)


def _through_round(report: Dict[str, Any], rounds: int) -> Dict[str, Any]:
    """Best score and cumulative calls through the first ``rounds`` rounds."""
    return _accumulate(report["rounds"][:max(0, int(rounds))])


def _through_calls(report: Dict[str, Any], budget: int) -> Dict[str, Any]:
    """Best score and calls through the last round that stays within ``budget``.

    Rounds are whole units of work: a round that would overrun the budget is not
    counted, so the dreaming arm is never credited with a discovery it had to
    overspend to reach.
    """
    kept = []
    for record in report["rounds"]:
        if int(record["cumulative_agent_calls"]) > budget:
            break
        kept.append(record)
    return _accumulate(kept)


def _accumulate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best = max((r["manifest"].get("best_score") for r in records
                if r["manifest"].get("best_score") is not None), default=None)
    calls = int(records[-1]["cumulative_agent_calls"]) if records else 0
    return {"best_score": best, "calls": calls, "rounds": len(records)}


def ablate_pool_window(base: RSIConfig, seeds: Sequence[int],
                       windows: Sequence[Optional[int]] = (None, 1), *,
                       equal_budget_rounds: Optional[int] = None,
                       out_dir: Optional[str] = None,
                       log: Optional[Callable[..., None]] = None) -> List[AblationRow]:
    """Repeat the comparison with the simulator pool pinned to a window.

    ``None`` is the paper's setting -- dream over all of ``H_t``.  ``1`` keeps
    only the newest world, so the offline phase never accumulates evidence; any
    advantage that disappears between the two is an advantage that came from the
    history *growing*, not merely from replaying one.
    """
    out: List[AblationRow] = []
    for window in windows:
        label = "all" if window is None else f"w{window}"
        out.append(AblationRow(
            label=label,
            pool_window=window,
            comparison=compare_across_seeds(
                base, seeds, pool_window=window,
                equal_budget_rounds=equal_budget_rounds,
                out_dir=os.path.join(out_dir or base.out_dir, f"ablation-{label}"),
                label=label, log=log),
        ))
    return out


def bootstrap_ci(values: Sequence[float], *, alpha: float = 0.05,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = 12345) -> Optional[List[float]]:
    """Percentile bootstrap interval for the mean.  ``None`` below two points.

    Seeded so a reported interval is reproducible.  With a handful of seeds this
    interval is wide and should be read as spread, not as a test.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(resamples):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return [lo, hi]


def _finite(v: Any) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return None if v == float("-inf") else v


def _mean(xs: Sequence[float]) -> Optional[float]:
    return statistics.fmean(xs) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    return statistics.median(xs) if xs else None
