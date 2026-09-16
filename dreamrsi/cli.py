"""``dreamrsi`` command line.

    dreamrsi run       --config configs/synthetic.yaml
    dreamrsi compare   --config configs/synthetic.yaml     # Dream-RSI vs fixed
    dreamrsi eval      --policy P.py --pool runs/x/history # beta sweep on a pool
    dreamrsi dream     --policy P.py --pool runs/x/history # offline improvement
    dreamrsi fidelity  --config configs/synthetic.yaml     # is replay exact?
    dreamrsi generalize --config configs/synthetic.yaml    # held-out worlds
    dreamrsi benchmark --config configs/synthetic.yaml     # across seeds
    dreamrsi inspect   runs/x/history/r0001_dream/trace.json
    dreamrsi tasks | dreamrsi prompt exploration
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__, prompts
from .agents import make_policy_developer
from .dreaming import DEFAULT_BETA_GRID, dream, evaluate_policy
from .policy.loader import load_policy_module, load_policy_source
from .rsi import DREAM, FIXED, RSIConfig, build_components, run_rsi
from .storage import write_json
from .tasks import available as available_tasks
from .tree import DiscoveryTree
from .types import GridPlan
from .world import SimulatorPool


def load_config(path: Optional[str], overrides: Dict[str, Any]) -> RSIConfig:
    data: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith((".yaml", ".yml")):
                import yaml

                data = yaml.safe_load(fh) or {}
            else:
                data = json.load(fh)
    data.update({k: v for k, v in overrides.items() if v is not None})
    return RSIConfig.from_dict(data)


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, {"mode": args.mode, "rounds": args.rounds,
                                       "name": args.name, "seed": args.seed,
                                       "task": args.task, "out_dir": args.out_dir})
    if args.task and not args.name:
        config.name = f"{config.name}-{args.task}"   # keep run dirs distinct
    task, agent, developer = build_components(config)
    report = run_rsi(config, task=task, agent=agent, developer=developer)
    print(json.dumps({"name": config.name, "mode": config.mode,
                      "best_score": report["best_score"],
                      "cumulative_agent_calls": report["cumulative_agent_calls"],
                      "seconds": report["seconds"]}, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run Recursive Fixed Exploration and Dream-RSI under identical budgets."""
    base = load_config(args.config, {"rounds": args.rounds, "seed": args.seed,
                                     "out_dir": args.out_dir})
    results = {}
    for mode in (FIXED, DREAM):
        config = RSIConfig.from_dict({**base.__dict__, "mode": mode,
                                      "name": f"{base.name}-{mode}"})
        task, agent, developer = build_components(config)
        results[mode] = run_rsi(config, task=task, agent=agent, developer=developer)

    print()
    print(f"{'round':>5}  {'fixed best':>14} {'fixed calls':>12}   "
          f"{'dream best':>14} {'dream calls':>12}")
    print("-" * 68)
    rounds = max(len(results[FIXED]["rounds"]), len(results[DREAM]["rounds"]))
    for i in range(rounds):
        row = [f"{i + 1:>5}"]
        for mode in (FIXED, DREAM):
            rs = results[mode]["rounds"]
            if i < len(rs):
                m = rs[i]["manifest"]
                best = m.get("best_score")
                row.append(f"  {('--' if best is None else f'{best:.6f}'):>14} "
                           f"{rs[i]['cumulative_agent_calls']:>12}")
            else:
                row.append(f"  {'--':>14} {'--':>12}")
        print("".join(row))
    print("-" * 68)
    for mode in (FIXED, DREAM):
        r = results[mode]
        print(f"{mode:>10}: best={r['best_score']:.6f}  "
              f"calls={r['cumulative_agent_calls']}  seconds={r['seconds']}")
    write_json(os.path.join(base.out_dir, f"{base.name}-comparison.json"),
               {m: {"best_score": r["best_score"],
                    "cumulative_agent_calls": r["cumulative_agent_calls"],
                    "rounds": [{"round": x["round"],
                                "best_score": x["manifest"].get("best_score"),
                                "calls": x["cumulative_agent_calls"],
                                "plan": x["plan"]} for x in r["rounds"]]}
                for m, r in results.items()})
    return 0


def _pool(paths: Sequence[str], max_parallelism: int, max_decision_rounds: int) -> SimulatorPool:
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "trace.json"), recursive=True)))
        else:
            files.append(p)
    if not files:
        raise SystemExit(f"no traces found under {list(paths)}")
    return SimulatorPool([DiscoveryTree.load(f) for f in files],
                         max_parallelism=max_parallelism,
                         max_decision_rounds=max_decision_rounds)


def cmd_eval(args: argparse.Namespace) -> int:
    spec = load_policy_module(args.policy, strict=False)
    pool = _pool(args.pool, args.max_parallelism, args.max_decision_rounds)
    ev = evaluate_policy(spec, pool, betas=args.betas or DEFAULT_BETA_GRID,
                         base_config={"beta": args.beta}, collect_traces=False)
    print(json.dumps({"policy": spec.name, "warnings": spec.warnings,
                      "worlds": len(pool), "pareto": ev.summary.to_dict()},
                     indent=2, default=str))
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    with open(args.policy, "r", encoding="utf-8") as fh:
        source = fh.read()
    pool = _pool(args.pool, args.max_parallelism, args.max_decision_rounds)
    developer = make_policy_developer(args.developer)
    out = dream(source, pool, developer, revisions=args.revisions,
                out_dir=args.out_dir, betas=args.betas or DEFAULT_BETA_GRID,
                base_config={"beta": args.beta})
    best = out["best"]
    print(json.dumps({"worlds": len(pool),
                      "selected_version": best.version,
                      "improved": out["improved"],
                      "incumbent_reward": out["incumbent"].summary.reward,
                      "selected_reward": best.summary.reward,
                      "artifacts": os.path.join(args.out_dir, "proposal_results")},
                     indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    tree = DiscoveryTree.load(args.trace)
    print(json.dumps(tree.summary(), indent=2))
    if args.verbose:
        for node in tree:
            score = "--" if node.score == float("-inf") else f"{node.score:.6f}"
            print(f"  {node.cell_id:>8}  seq={node.seq:>3}  score={score:>12}  "
                  f"{node.fail_class:<14} {'ok' if node.error is None else node.error[:60]}")
    return 0


def _candidate_policies() -> List[Tuple[str, str]]:
    """The candidate set dreaming chooses from: the incumbent, then the library.

    Index 0 is ``pi_1``, matching :func:`dreamrsi.dreaming.dream`'s convention
    that version 0 is the incumbent.  The mock developer proposes these same
    sources with a re-baked default beta, so auditing them directly audits the
    decision the offline phase actually makes.
    """
    from .policy import baseline, library

    return ([("pi_1", baseline.source())]
            + [(n, library.source_of(n)) for n in library.available()])


def cmd_fidelity(args: argparse.Namespace) -> int:
    """Check that a frozen history replays exactly, and measure what it omits."""
    from .experiments.fidelity import fidelity_probe, support_coverage

    config = load_config(args.config, {"seed": args.seed, "out_dir": args.out_dir})
    task, agent, _ = build_components(RSIConfig.from_dict({**config.__dict__,
                                                          "mode": FIXED}))
    root = os.path.join(config.out_dir, f"{config.name}-fidelity")
    history_dir = os.path.join(root, "history")
    os.makedirs(history_dir, exist_ok=True)
    paths = task.prepare(history_dir)
    baseline_score = task.baseline_score(history_dir)
    plan = GridPlan(branch_count=config.branch_count, refine_count=config.refine_count,
                    reason="fidelity probe")
    betas = list(args.betas or (0.2, config.beta, 1.0))

    probes, trees = [], []
    for name, source in _candidate_policies():
        for beta in betas:
            out = fidelity_probe(
                task, agent, source,
                run_dir=os.path.join(root, f"{name}_b{beta}"),
                history_dir=history_dir, baseline_dir=paths["baseline_dir"],
                problem_file=paths["problem_file"], baseline_score=baseline_score,
                plan=plan, beta=beta, name=name,
                max_parallelism=config.max_parallelism,
                max_decision_rounds=config.max_decision_rounds)
            probes.append(out)
            trees.append(out["tree"])

    grid = plan.branch_count * (plan.refine_count + 1)
    print(f"{'policy':>20} {'beta':>6} {'exact':>7} {'rounds live/replay':>20} "
          f"{'probes live/replay':>20} {'trace':>10}")
    print("-" * 89)
    for out in probes:
        f = out["fidelity"]
        size = f"{len(out['tree'])}/{grid}"
        print(f"{f.policy:>20} {f.beta:>6.2f} {str(f.exact):>7} "
              f"{f'{f.live_rounds}/{f.replay_rounds}':>20} "
              f"{f'{f.live_probes}/{f.replay_probes}':>20} {size:>10}")
    exact = sum(1 for o in probes if o["fidelity"].exact)
    print("-" * 89)
    print(f"exact replays: {exact}/{len(probes)}   "
          f"(trace = recorded cells out of the {grid}-cell live grid)")

    # Off-policy reach: every candidate replayed over every frozen trace.
    print()
    print("off-policy support: each candidate replayed over every frozen trace")
    print(f"{'policy':>20} {'mean cov':>9} {'worst cov':>10} {'narrowed':>9} "
          f"{'cut off':>9} {'actions left':>13}")
    print("-" * 74)
    matrix = []
    for name, source in _candidate_policies():
        spec = load_policy_source(source, name=name, strict=False)
        row = [support_coverage(tree, spec, config={"beta": config.beta}, label=name,
                                max_parallelism=config.max_parallelism,
                                max_decision_rounds=config.max_decision_rounds)
               for tree in trees]
        matrix.append((name, row))
        covs = [r.mean_action_coverage for r in row]
        narrowed = sum(1 for r in row if r.restricted_rounds)
        cut = [r for r in row if r.stopped_by_exhaustion]
        left = sum(r.terminal_live_actions for r in cut)
        print(f"{name:>20} {sum(covs) / len(covs):>9.3f} {min(covs):>10.3f} "
              f"{f'{narrowed}/{len(row)}':>9} {f'{len(cut)}/{len(row)}':>9} {left:>13}")
    print("-" * 74)
    print("cov      mean fraction of the live grid's legal actions the trace could serve")
    print("narrowed traces on which at least one decision round offered fewer actions")
    print("cut off  replays that ended because the trace ran out, not because the")
    print("         policy chose to stop -- with that many live actions still open")

    write_json(os.path.join(root, "fidelity.json"), {
        "betas": betas,
        "probes": [{"fidelity": o["fidelity"].to_dict(),
                    "self_support": o["support"].to_dict()} for o in probes],
        "off_policy_support": {name: [r.to_dict() for r in row] for name, row in matrix},
    })
    return 0 if exact == len(probes) else 1


def cmd_generalize(args: argparse.Namespace) -> int:
    """Audit a selection decision against worlds the pool never contained."""
    from .experiments.generalization import generalization_report, temporal_split

    config = load_config(args.config, {"rounds": args.rounds, "seed": args.seed,
                                       "out_dir": args.out_dir})
    if args.pool:
        trees = _pool(args.pool, config.max_parallelism,
                      config.max_decision_rounds).trees
    else:
        # Collect a history the honest way: actually run the loop, in the mode
        # the config asks for, so the pool is the one dreaming would really see.
        run_config = RSIConfig.from_dict({**config.__dict__,
                                          "name": f"{config.name}-generalize"})
        task, agent, developer = build_components(run_config)
        run_rsi(run_config, task=task, agent=agent, developer=developer)
        trees = _pool([os.path.join(run_config.out_dir, run_config.name, "history")],
                      config.max_parallelism, config.max_decision_rounds).trees

    pool, holdout = temporal_split(trees, holdout_last=args.holdout_last)
    report = generalization_report(
        _candidate_policies(), pool, holdout,
        betas=config.betas, lam=config.lam, base_config={"beta": config.beta},
        max_parallelism=config.max_parallelism,
        max_decision_rounds=config.max_decision_rounds)

    print(f"pool: {len(pool)} world(s)   held out: {len(holdout)} world(s)")
    print(f"{'version':>8} {'pool reward':>13} {'holdout reward':>15} {'gap':>10}")
    print("-" * 50)
    for v in report.versions:
        print(f"{v.version:>8} "
              f"{'--' if v.pool_reward is None else f'{v.pool_reward:.6f}':>13} "
              f"{'--' if v.holdout_reward is None else f'{v.holdout_reward:.6f}':>15} "
              f"{'--' if v.gap is None else f'{v.gap:+.6f}':>10}")
    print("-" * 50)
    print(f"selected on pool: v{report.selected_by_pool}   "
          f"best on holdout: v{report.best_on_holdout}")
    print(f"selection regret: "
          f"{'--' if report.selection_regret is None else f'{report.selection_regret:.6f}'}   "
          f"mean gap: {'--' if report.mean_gap is None else f'{report.mean_gap:+.6f}'}   "
          f"rank corr: "
          f"{'--' if report.rank_correlation is None else f'{report.rank_correlation:+.3f}'}")
    print(f"non-regression transfers to held-out worlds: {report.holdout_non_regression}")

    write_json(os.path.join(config.out_dir, f"{config.name}-generalization.json"),
               report.to_dict())
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the Section 4 comparison across seeds, optionally with the ablation."""
    from .experiments.benchmark import ablate_pool_window, compare_across_seeds

    config = load_config(args.config, {"rounds": args.rounds, "out_dir": args.out_dir})
    seeds = list(args.seeds)
    out_root = os.path.join(config.out_dir, f"{config.name}-benchmark")

    matched_rounds = (args.equal_budget_rounds if args.equal_budget_rounds
                      else 3 * config.rounds)
    if args.ablate:
        rows = ablate_pool_window(config, seeds, windows=(None, 1),
                                  equal_budget_rounds=matched_rounds,
                                  out_dir=out_root, log=_quiet)
        payload: Dict[str, Any] = {"seeds": seeds,
                                   "arms": [r.to_dict() for r in rows]}
        print(f"{'pool':>12} {'W/T/L @rounds':>14} {'W/T/L @budget':>14} "
              f"{'mean call ratio':>16} {'cheaper seeds':>14}")
        print("-" * 76)
        for row in rows:
            s = row.comparison.summary()
            label = "all of H_t" if row.pool_window is None else f"last {row.pool_window}"
            cheaper = f"{s['cost']['cheaper_seeds']}/{len(seeds)}"
            print(f"{label:>12} {_record(s['quality']['record']):>14} "
                  f"{_record(s['quality_at_matched_budget']['record']):>14} "
                  f"{_fmt(s['cost']['mean_call_ratio'], '.3f'):>16} {cheaper:>14}")
        print("-" * 76)
        print("W/T/L is Dream-RSI's record on discovered quality, at equal rounds")
        print("and at equal discovery-agent calls.")
    else:
        comparison = compare_across_seeds(config, seeds, out_dir=out_root,
                                          equal_budget_rounds=matched_rounds,
                                          log=_quiet)
        payload = {"seeds": seeds, **comparison.summary()}
        _print_seed_table(comparison)

    write_json(os.path.join(out_root, "benchmark.json"), payload)
    return 0


def _print_seed_table(comparison) -> None:
    print(f"{'seed':>5} {'fixed best':>14} {'dream best':>14} {'dream @budget':>14} "
          f"{'fixed calls':>12} {'dream calls':>12} {'ratio':>7}")
    print("-" * 85)
    for row in comparison.rows:
        print(f"{row.seed:>5} "
              f"{_fmt(row.fixed_best):>14} {_fmt(row.dream_best):>14} "
              f"{_fmt(row.matched_best):>14} "
              f"{row.fixed_calls:>12} {row.dream_calls:>12} "
              f"{_fmt(row.call_ratio, '.3f'):>7}")
    s = comparison.summary()
    n = len(comparison.rows)
    print("-" * 85)
    print(f"quality @equal rounds (dream - fixed): "
          f"mean {_fmt(s['quality']['mean_delta'], '+.6f')}  "
          f"median {_fmt(s['quality']['median_delta'], '+.6f')}  "
          f"95% CI {_fmt_ci(s['quality']['ci95_delta'])}  "
          f"W/T/L {_record(s['quality']['record'])}")
    m = s["quality_at_matched_budget"]
    if m["seeds"]:
        print(f"quality @equal calls  (dream - fixed): "
              f"mean {_fmt(m['mean_delta'], '+.6f')}  "
              f"median {_fmt(m['median_delta'], '+.6f')}  "
              f"95% CI {_fmt_ci(m['ci95_delta'])}  "
              f"W/T/L {_record(m['record'])}")
    print(f"calls (fixed / dream):                mean "
          f"{_fmt(s['cost']['mean_call_ratio'], '.3f')}x  "
          f"median {_fmt(s['cost']['median_call_ratio'], '.3f')}x  "
          f"95% CI {_fmt_ci(s['cost']['ci95_call_ratio'], '.3f')}  "
          f"cheaper on {s['cost']['cheaper_seeds']}/{n}")


def _record(rec: Dict[str, int]) -> str:
    return f"{rec['win']}/{rec['tie']}/{rec['loss']}"


def _fmt(v: Optional[float], spec: str = ".6f") -> str:
    return "--" if v is None else format(v, spec)


def _fmt_ci(ci: Optional[Sequence[float]], spec: str = "+.6f") -> str:
    return "--" if not ci else f"[{format(ci[0], spec)}, {format(ci[1], spec)}]"


def _quiet(*_args: Any) -> None:
    """Silence per-round progress: a benchmark runs the loop dozens of times."""


def cmd_tasks(args: argparse.Namespace) -> int:
    for name in available_tasks():
        print(name)
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    name = {"exploration": prompts.EXPLORATION,
            "policy": prompts.POLICY_IMPROVEMENT}[args.which]
    sys.stdout.write(prompts.load(name))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dreamrsi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"opendreamrsi {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--config", help="YAML/JSON run config")
        sp.add_argument("--rounds", type=int, help="outer iterations T")
        sp.add_argument("--seed", type=int)
        sp.add_argument("--out-dir", dest="out_dir", default=None)
        return sp

    r = common(sub.add_parser("run", help="run one recursive self-improvement loop"))
    r.add_argument("--mode", choices=[DREAM, FIXED], default=None)
    r.add_argument("--name", default=None)
    r.add_argument("--task", choices=available_tasks(), default=None,
                   help="override the config's task; every task ignores knobs it "
                        "does not read, so one agent config can drive several")
    r.set_defaults(func=cmd_run)

    c = common(sub.add_parser("compare", help="Dream-RSI vs Recursive Fixed Exploration"))
    c.set_defaults(func=cmd_compare)

    e = sub.add_parser("eval", help="beta-sweep a policy over a pool of frozen traces")
    e.add_argument("--policy", required=True)
    e.add_argument("--pool", nargs="+", required=True)
    e.add_argument("--beta", type=float, default=0.62)
    e.add_argument("--betas", type=float, nargs="*")
    e.add_argument("--max-parallelism", dest="max_parallelism", type=int, default=4)
    e.add_argument("--max-decision-rounds", dest="max_decision_rounds", type=int, default=32)
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("dream", help="offline policy improvement over a pool")
    d.add_argument("--policy", required=True)
    d.add_argument("--pool", nargs="+", required=True)
    d.add_argument("--revisions", type=int, default=3)
    d.add_argument("--developer", default="mock")
    d.add_argument("--beta", type=float, default=0.62)
    d.add_argument("--betas", type=float, nargs="*")
    d.add_argument("--out-dir", dest="out_dir", default="runs/dream")
    d.add_argument("--max-parallelism", dest="max_parallelism", type=int, default=4)
    d.add_argument("--max-decision-rounds", dest="max_decision_rounds", type=int, default=32)
    d.set_defaults(func=cmd_dream)

    f = common(sub.add_parser(
        "fidelity", help="check that a frozen history replays exactly, and what it omits"))
    f.add_argument("--betas", type=float, nargs="*",
                   help="betas to probe (default: 0.2, the config beta, 1.0)")
    f.set_defaults(func=cmd_fidelity)

    g = common(sub.add_parser(
        "generalize", help="score dreaming's candidates on held-out worlds"))
    g.add_argument("--pool", nargs="*",
                   help="trace files or directories; omit to collect a fresh history")
    g.add_argument("--holdout-last", dest="holdout_last", type=int, default=1,
                   help="how many of the newest worlds to hold out (default 1)")
    g.set_defaults(func=cmd_generalize)

    b = sub.add_parser("benchmark", help="fixed vs dream across seeds, paired")
    b.add_argument("--config", help="YAML/JSON run config")
    b.add_argument("--rounds", type=int, help="outer iterations T")
    b.add_argument("--out-dir", dest="out_dir", default=None)
    b.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    b.add_argument("--equal-budget-rounds", dest="equal_budget_rounds", type=int,
                   default=None,
                   help="rounds to give the dreaming arm so it can spend the fixed "
                        "arm's full call budget (default: 3x --rounds)")
    b.add_argument("--ablate", action="store_true",
                   help="also run with the simulator pool pinned to the newest world")
    b.set_defaults(func=cmd_benchmark)

    i = sub.add_parser("inspect", help="summarise a frozen discovery tree")
    i.add_argument("trace")
    i.add_argument("-v", "--verbose", action="store_true")
    i.set_defaults(func=cmd_inspect)

    sub.add_parser("tasks", help="list registered tasks").set_defaults(func=cmd_tasks)

    pr = sub.add_parser("prompt", help="print a prompt from Appendix B")
    pr.add_argument("which", choices=["exploration", "policy"])
    pr.set_defaults(func=cmd_prompt)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
