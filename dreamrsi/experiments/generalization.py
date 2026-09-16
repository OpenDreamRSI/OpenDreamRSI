"""Does a policy selected by dreaming survive contact with a world it never saw?

Dreaming maximises ``V^m``, the average replay score on the frozen pool ``H_t``.
Because the incumbent ``pi_t^0`` is always in the candidate set, the selected
``pi_{t+1}`` satisfies ``V^{m*} >= V^0`` -- but that guarantee is *in-pool* and
tautological: it says the winner won on the worlds it was scored on.  The
property the outer loop actually needs is different and is nowhere guaranteed:
``pi_{t+1}`` is deployed on ``T_{t+1}``, a world that did not exist when it was
selected.

Selecting on a frozen pool is a model-selection problem, so it has a model
selection problem's failure mode.  The pool is small -- one world per completed
round -- and every candidate is scored on all of it, so a revision can win by
suiting the recorded traces rather than by exploring better.  This module
measures that the way model selection is normally audited: hold worlds out.

``H_t`` is split *temporally* -- dream on the earlier worlds, score on the later
ones -- because that is the split the loop itself faces, and a random split
would leak the future into the pool.  Three numbers come out:

* **generalization gap** -- pool reward minus held-out reward, per version.  A
  large positive gap means replay scores are optimistic about live behaviour.
* **selection regret** -- how much held-out reward was lost by selecting on the
  pool instead of an oracle that could see the held-out worlds.  This is the
  quantity that matters: a big gap is harmless if it is uniform across
  candidates, because selection only depends on their *order*.
* **rank correlation** -- whether the pool orders candidates the way the
  held-out worlds do, which is precisely the assumption dreaming relies on.

Also reported is whether non-regression *transfers*: the selected version beats
the incumbent in-pool by construction, but on the held-out worlds it need not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..dreaming import DEFAULT_BETA_GRID, evaluate_policy
from ..objective import DEFAULT_LAMBDA
from ..policy.loader import load_policy_source
from ..tree import DiscoveryTree
from ..world import SimulatorPool

#: Two rewards closer than this are treated as a tie.
TIE = 1e-12


@dataclass
class VersionScore:
    """One candidate policy version scored on both sides of the split."""

    version: int
    name: str
    pool_reward: Optional[float]
    holdout_reward: Optional[float]
    pool_attainment: Optional[float] = None
    holdout_attainment: Optional[float] = None
    pool_probes: Optional[float] = None
    holdout_probes: Optional[float] = None
    error: Optional[str] = None

    @property
    def gap(self) -> Optional[float]:
        """``pool_reward - holdout_reward``: how optimistic the pool was."""
        if self.pool_reward is None or self.holdout_reward is None:
            return None
        return self.pool_reward - self.holdout_reward

    @property
    def scored(self) -> bool:
        return self.pool_reward is not None and self.holdout_reward is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "pool_reward": self.pool_reward,
            "holdout_reward": self.holdout_reward,
            "gap": self.gap,
            "pool_attainment": self.pool_attainment,
            "holdout_attainment": self.holdout_attainment,
            "pool_probes": self.pool_probes,
            "holdout_probes": self.holdout_probes,
            "error": self.error,
        }


@dataclass
class GeneralizationReport:
    """The audit of one selection decision against held-out worlds."""

    pool_worlds: List[str]
    holdout_worlds: List[str]
    versions: List[VersionScore] = field(default_factory=list)
    selected_by_pool: Optional[int] = None
    best_on_holdout: Optional[int] = None
    selection_regret: Optional[float] = None
    mean_gap: Optional[float] = None
    rank_correlation: Optional[float] = None
    holdout_non_regression: Optional[bool] = None

    @property
    def selection_agrees(self) -> bool:
        """Whether selecting on the pool picked an oracle-optimal version."""
        return (self.selection_regret is not None
                and self.selection_regret <= TIE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_worlds": list(self.pool_worlds),
            "holdout_worlds": list(self.holdout_worlds),
            "versions": [v.to_dict() for v in self.versions],
            "selected_by_pool": self.selected_by_pool,
            "best_on_holdout": self.best_on_holdout,
            "selection_regret": self.selection_regret,
            "selection_agrees": self.selection_agrees,
            "mean_gap": self.mean_gap,
            "rank_correlation": self.rank_correlation,
            "holdout_non_regression": self.holdout_non_regression,
        }


def temporal_split(trees: Sequence[DiscoveryTree], holdout_last: int = 1
                   ) -> Tuple[List[DiscoveryTree], List[DiscoveryTree]]:
    """Split a history into ``(dream pool, held-out worlds)``, latest held out.

    Worlds arrive in round order, so holding out the tail reproduces the loop's
    own situation: select on what has happened, deploy on what has not.
    """
    trees = list(trees)
    k = int(holdout_last)
    if k < 1:
        raise ValueError("holdout_last must be >= 1")
    if len(trees) <= k:
        raise ValueError(
            f"need more than {k} world(s) to hold {k} out; the pool has {len(trees)}"
        )
    return trees[:-k], trees[-k:]


def generalization_report(candidates: Sequence[Tuple[str, str]],
                          pool_trees: Sequence[DiscoveryTree],
                          holdout_trees: Sequence[DiscoveryTree], *,
                          betas: Sequence[float] = DEFAULT_BETA_GRID,
                          lam: float = DEFAULT_LAMBDA,
                          base_config: Optional[Dict[str, Any]] = None,
                          max_parallelism: int = 6,
                          max_decision_rounds: int = 40) -> GeneralizationReport:
    """Score every candidate on both sides of the split and audit the choice.

    ``candidates`` is ``[(name, source), ...]`` in dreaming order, so index 0 is
    the incumbent ``pi_t^0`` -- the same convention
    :func:`dreamrsi.dreaming.dream` uses.
    """
    pool = SimulatorPool(list(pool_trees), max_parallelism=max_parallelism,
                         max_decision_rounds=max_decision_rounds)
    holdout = SimulatorPool(list(holdout_trees), max_parallelism=max_parallelism,
                            max_decision_rounds=max_decision_rounds)

    scores: List[VersionScore] = []
    for version, (name, source) in enumerate(candidates):
        try:
            spec = load_policy_source(source, name=name or f"v{version:03d}", strict=False)
        except Exception as exc:  # noqa: BLE001 - candidates are agent-written code
            scores.append(VersionScore(version=version, name=name, pool_reward=None,
                                       holdout_reward=None,
                                       error=f"{type(exc).__name__}: {exc}"))
            continue
        on_pool = evaluate_policy(spec, pool, betas=betas, lam=lam,
                                  base_config=base_config, version=version,
                                  collect_traces=False)
        on_holdout = evaluate_policy(spec, holdout, betas=betas, lam=lam,
                                     base_config=base_config, version=version,
                                     collect_traces=False)
        scores.append(VersionScore(
            version=version,
            name=spec.name,
            pool_reward=None if on_pool.error else on_pool.summary.reward,
            holdout_reward=None if on_holdout.error else on_holdout.summary.reward,
            pool_attainment=on_pool.summary.attainment,
            holdout_attainment=on_holdout.summary.attainment,
            pool_probes=on_pool.summary.mean_probes,
            holdout_probes=on_holdout.summary.mean_probes,
            error=on_pool.error or on_holdout.error,
        ))

    report = GeneralizationReport(
        pool_worlds=[t.world_id for t in pool_trees],
        holdout_worlds=[t.world_id for t in holdout_trees],
        versions=scores,
    )

    usable = [s for s in scores if s.scored]
    if not usable:
        return report

    selected = max(usable, key=lambda s: s.pool_reward)          # type: ignore[arg-type]
    oracle = max(usable, key=lambda s: s.holdout_reward)         # type: ignore[arg-type]
    report.selected_by_pool = selected.version
    report.best_on_holdout = oracle.version
    report.selection_regret = float(oracle.holdout_reward - selected.holdout_reward)  # type: ignore[operator]
    report.mean_gap = _mean([s.gap for s in usable if s.gap is not None])
    report.rank_correlation = spearman([s.pool_reward for s in usable],      # type: ignore[misc]
                                       [s.holdout_reward for s in usable])   # type: ignore[misc]

    incumbent = next((s for s in usable if s.version == 0), None)
    if incumbent is not None:
        report.holdout_non_regression = bool(
            selected.holdout_reward >= incumbent.holdout_reward - TIE  # type: ignore[operator]
        )
    return report


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation with average ranks for ties; ``None`` below two points.

    Returns ``None`` when either side is constant, because a constant ordering
    has no direction to agree or disagree with.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return None
    return num / ((dx * dy) ** 0.5)


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = shared
        i = j + 1
    return out


def _mean(xs: Sequence[float]) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else 0.0
