"""Read Section 4's controlled comparison back off two finished run directories.

``dreamrsi compare`` runs both arms itself and prints a table at the end.  That
is fine when both arms fit in one invocation, and useless otherwise: a real
comparison on a real model takes hours per arm, gets run separately so the two
arms do not contend for the same cores, and is analysed long after the
processes are gone.  Everything the comparison needs is already on disk in
``report.json``, ``trace.json`` and ``beta_sweep.json``; this module reads it.

It reports three things that a bare best-score delta hides, and which decided
how an earlier run on this repository had to be read:

* **The round-1 confound.**  Both arms deploy the *same* seed policy in round 1
  over the same grid.  Whatever gap appears there is sampling noise by
  construction, so it is the natural scale against which the final gap has to
  be judged.  Measured once at 0.354 against a final gap of 0.208 -- that is,
  the "effect" was smaller than the noise.
* **Budget.**  An arm that ran fewer rounds is not a control.  The columns
  carry cumulative agent calls so an unmatched comparison is visible rather
  than averaged over.
* **What dreaming chose, and against what.**  Per revision: the reward, the
  AUC, the attainment, and whether the beta sweep found the policy responsive
  to beta at all.  A phase that selected the incumbent because every candidate
  crashed and a phase that selected it because every candidate was genuinely
  worse are the same line in a score table and different findings.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

INIT_DIR = "r0000_init"


# -- reading -----------------------------------------------------------------
def _read(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def round_dirs(run_dir: str) -> List[str]:
    """Live-cycle directories of a run, in round order, excluding the seed dir."""
    history = os.path.join(run_dir, "history")
    if not os.path.isdir(history):
        return []
    return sorted(d for d in os.listdir(history)
                  if d.startswith("r0") and d != INIT_DIR)


def fail_classes(run_dir: str, round_dir: str) -> Counter:
    """How the evaluator classed every attempt of one live cycle."""
    trace = _read(os.path.join(run_dir, "history", round_dir, "trace.json"))
    if not trace:
        return Counter()
    return Counter(n.get("fail_class", "?") for n in trace.get("nodes", ()))


def beta_sweep(run_dir: str, round_dir: str) -> Optional[Dict[str, Any]]:
    return _read(os.path.join(run_dir, "history", round_dir,
                              "proposal_results", "beta_sweep.json"))


# -- rows --------------------------------------------------------------------
@dataclass
class ArmRound:
    round: int
    best_score: Optional[float]
    cumulative_agent_calls: int
    probes: int
    decision_rounds: int
    fail_classes: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"round": self.round, "best_score": self.best_score,
                "cumulative_agent_calls": self.cumulative_agent_calls,
                "probes": self.probes, "decision_rounds": self.decision_rounds,
                "fail_classes": dict(self.fail_classes)}


@dataclass
class DreamPhase:
    """One dreaming phase: what was on offer and what was taken."""

    round: int
    selected_version: int
    improved: bool
    incumbent_reward: Optional[float]
    selected_reward: Optional[float]
    versions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def crashed(self) -> int:
        return sum(1 for v in self.versions if v.get("error"))

    def to_dict(self) -> Dict[str, Any]:
        d = {"round": self.round, "selected_version": self.selected_version,
             "improved": self.improved, "incumbent_reward": self.incumbent_reward,
             "selected_reward": self.selected_reward, "crashed": self.crashed,
             "versions": self.versions}
        return d


@dataclass
class ArmReport:
    run_dir: str
    mode: str
    best_score: Optional[float]
    cumulative_agent_calls: int
    seconds: float
    baseline_score: float
    rounds: List[ArmRound] = field(default_factory=list)
    phases: List[DreamPhase] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"run_dir": self.run_dir, "mode": self.mode,
                "best_score": self.best_score,
                "cumulative_agent_calls": self.cumulative_agent_calls,
                "seconds": self.seconds, "baseline_score": self.baseline_score,
                "rounds": [r.to_dict() for r in self.rounds],
                "phases": [p.to_dict() for p in self.phases]}


def load_arm(run_dir: str) -> ArmReport:
    """Reconstruct one arm from its run directory."""
    report = _read(os.path.join(run_dir, "report.json"))
    if report is None:
        raise FileNotFoundError(os.path.join(run_dir, "report.json"))
    dirs = round_dirs(run_dir)

    rounds: List[ArmRound] = []
    phases: List[DreamPhase] = []
    for i, entry in enumerate(report.get("rounds", ())):
        manifest = entry.get("manifest", {})
        rd = dirs[i] if i < len(dirs) else None
        rounds.append(ArmRound(
            round=entry.get("round", i + 1),
            best_score=manifest.get("best_score"),
            cumulative_agent_calls=entry.get("cumulative_agent_calls", 0),
            probes=manifest.get("probes", 0),
            decision_rounds=manifest.get("decision_rounds", 0),
            fail_classes=dict(fail_classes(run_dir, rd)) if rd else {},
        ))
        sweep = beta_sweep(run_dir, rd) if rd else None
        dream = entry.get("dream")
        if sweep and dream:
            phases.append(DreamPhase(
                round=entry.get("round", i + 1),
                selected_version=sweep.get("selected_version", 0),
                improved=bool(dream.get("improved")),
                incumbent_reward=dream.get("incumbent_reward"),
                selected_reward=dream.get("selected_reward"),
                versions=list(sweep.get("versions", ())),
            ))

    return ArmReport(
        run_dir=run_dir,
        mode=report.get("config", {}).get("mode", "?"),
        best_score=report.get("best_score"),
        cumulative_agent_calls=report.get("cumulative_agent_calls", 0),
        seconds=float(report.get("seconds", 0.0)),
        baseline_score=float(report.get("baseline_score", 0.0)),
        rounds=rounds, phases=phases, config=report.get("config", {}),
    )


# -- the comparison ----------------------------------------------------------
@dataclass
class ArmComparison:
    dream: ArmReport
    fixed: ArmReport

    @property
    def noise_floor(self) -> Optional[float]:
        """The round-1 gap, which is noise by construction.

        Both arms run the identical seed policy over the identical grid in
        round 1, so any gap there is what two samples of the same process look
        like -- and therefore the smallest difference a final gap has to clear
        before it means anything.
        """
        if not self.dream.rounds or not self.fixed.rounds:
            return None
        a, b = self.dream.rounds[0].best_score, self.fixed.rounds[0].best_score
        return None if a is None or b is None else abs(a - b)

    @property
    def final_gap(self) -> Optional[float]:
        a, b = self.dream.best_score, self.fixed.best_score
        return None if a is None or b is None else a - b

    @property
    def budget_matched(self) -> bool:
        return (self.dream.cumulative_agent_calls
                == self.fixed.cumulative_agent_calls)

    @property
    def exceeds_noise(self) -> Optional[bool]:
        gap, floor = self.final_gap, self.noise_floor
        return None if gap is None or floor is None else abs(gap) > floor

    def to_dict(self) -> Dict[str, Any]:
        return {"dream": self.dream.to_dict(), "fixed": self.fixed.to_dict(),
                "noise_floor": self.noise_floor, "final_gap": self.final_gap,
                "budget_matched": self.budget_matched,
                "exceeds_noise": self.exceeds_noise}


def compare_arms(dream_dir: str, fixed_dir: str) -> ArmComparison:
    return ArmComparison(dream=load_arm(dream_dir), fixed=load_arm(fixed_dir))


# -- rendering ---------------------------------------------------------------
def _num(x: Optional[float], width: int = 12, places: int = 6) -> str:
    return f"{'--':>{width}}" if x is None else f"{x:>{width}.{places}f}"


def render(cmp: ArmComparison) -> str:
    """The table, then the caveats that decide how to read it."""
    out: List[str] = []
    cfg = cmp.dream.config
    grid = cfg.get("branch_count", 0) * (cfg.get("refine_count", 0) + 1)
    ceiling = cfg.get("max_parallelism", 0) * cfg.get("max_decision_rounds", 0)

    out.append("=" * 96)
    out.append(f"Dream-RSI vs Recursive Fixed Exploration -- {cfg.get('task', '?')}")
    out.append("=" * 96)
    out.append(f"grid      {cfg.get('branch_count')} x {cfg.get('refine_count', 0) + 1}"
               f" = {grid} cells   W={cfg.get('max_parallelism')}"
               f"  K1={cfg.get('max_decision_rounds')}  ceiling={ceiling} probes")
    out.append(f"loop      T={cfg.get('rounds')}  M={cfg.get('revisions')}"
               f"  engine={cfg.get('engine')}  seed={cfg.get('seed')}")
    out.append(f"baseline  {cmp.dream.baseline_score:.6f}")
    out.append("")

    out.append(f"{'round':>5} | {'dream best':>12} {'calls':>6} {'probes':>7} "
               f"| {'fixed best':>12} {'calls':>6} {'probes':>7} | {'delta':>11}")
    out.append("-" * 96)
    for i in range(max(len(cmp.dream.rounds), len(cmp.fixed.rounds))):
        cells = [f"{i + 1:>5} |"]
        got: Dict[str, Optional[float]] = {}
        for tag, arm in (("d", cmp.dream), ("f", cmp.fixed)):
            if i < len(arm.rounds):
                r = arm.rounds[i]
                got[tag] = r.best_score
                cells.append(f" {_num(r.best_score)} {r.cumulative_agent_calls:>6}"
                             f" {r.probes:>7} |")
            else:
                got[tag] = None
                cells.append(f" {'--':>12} {'--':>6} {'--':>7} |")
        d, f = got.get("d"), got.get("f")
        cells.append(f" {'--':>11}" if d is None or f is None else f" {d - f:>+11.6f}")
        out.append("".join(cells))
    out.append("-" * 96)
    for tag, arm in (("dream", cmp.dream), ("fixed", cmp.fixed)):
        out.append(f"{tag:>10}: best={_num(arm.best_score, 0)}"
                   f"  calls={arm.cumulative_agent_calls}"
                   f"  wall={arm.seconds / 60:.1f} min")
    out.append("")

    # -- how to read it ------------------------------------------------------
    if not cmp.budget_matched:
        out.append(f"WARNING: budgets differ ({cmp.dream.cumulative_agent_calls} vs "
                   f"{cmp.fixed.cumulative_agent_calls} agent calls). The arms are not")
        out.append("         a control on each other; any gap includes the extra budget.")
        out.append("")

    floor, gap = cmp.noise_floor, cmp.final_gap
    out.append("Round 1 is the control on itself: both arms deploy the same seed policy")
    out.append("over the same grid, so the round-1 gap is sampling noise by construction.")
    if floor is not None and gap is not None:
        out.append(f"  round-1 gap (noise) : {floor:.6f}")
        out.append(f"  final gap           : {gap:+.6f}")
        out.append("  -> " + ("final gap exceeds the round-1 noise; still n=1 per arm."
                              if abs(gap) > floor else
                              "the final gap does not clear the noise floor. n=1 shows nothing."))
    out.append("")

    out.append("Failure classes per round (dream / fixed):")
    for i in range(max(len(cmp.dream.rounds), len(cmp.fixed.rounds))):
        a = cmp.dream.rounds[i].fail_classes if i < len(cmp.dream.rounds) else {}
        b = cmp.fixed.rounds[i].fail_classes if i < len(cmp.fixed.rounds) else {}
        keys = sorted(set(a) | set(b))
        out.append(f"  round {i + 1}: " +
                   ("  ".join(f"{k}={a.get(k, 0)}/{b.get(k, 0)}" for k in keys) or "--"))
    out.append("")

    out.append("Dreaming phases:")
    if not cmp.dream.phases:
        out.append("  none recorded")
    degenerate: List[bool] = []
    for ph in cmp.dream.phases:
        out.append(f"  round {ph.round} -> pi_{ph.round + 1}: selected v{ph.selected_version}"
                   f"  incumbent={_num(ph.incumbent_reward, 0)}"
                   f" selected={_num(ph.selected_reward, 0)}"
                   f"  improved={ph.improved}"
                   + (f"  ({ph.crashed} crashed)" if ph.crashed else ""))
        for v in ph.versions:
            p = v.get("pareto") or {}
            degenerate.append(bool(p.get("non_degenerate")))
            note = f"  CRASHED: {str(v['error'])[:60]}" if v.get("error") else ""
            out.append(f"      v{v.get('version')!s:<3} reward={_num(v.get('score'), 10)}"
                       f"  auc={_num(p.get('auc'), 8, 4)}"
                       f"  attain={_num(p.get('attainment'), 6, 3)}"
                       f"  probes={_num(p.get('mean_probes'), 6, 1)}"
                       f"  beta_sensitive={p.get('non_degenerate')}{note}")
    out.append("")
    if degenerate and not any(degenerate):
        out.append("NOTE: no version responded to beta anywhere in the sweep. The policies")
        out.append("      on offer differ in code but not in behaviour, so the objective is")
        out.append("      ranking near-identical curves and the incumbent wins by default.")
    return "\n".join(out)
