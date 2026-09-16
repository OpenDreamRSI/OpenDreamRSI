"""``dreamrsi`` command line.

    dreamrsi run       --config configs/synthetic.yaml
    dreamrsi compare   --config configs/synthetic.yaml     # Dream-RSI vs fixed
    dreamrsi eval      --policy P.py --pool runs/x/history # beta sweep on a pool
    dreamrsi dream     --policy P.py --pool runs/x/history # offline improvement
    dreamrsi inspect   runs/x/history/r0001_dream/trace.json
    dreamrsi tasks | dreamrsi prompt exploration
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, prompts
from .agents import make_discovery_agent, make_policy_developer
from .dreaming import DEFAULT_BETA_GRID, dream, evaluate_policy
from .policy.loader import load_policy_module
from .rsi import DREAM, FIXED, RSIConfig, run_rsi
from .storage import write_json
from .tasks import available as available_tasks, make_task
from .tree import DiscoveryTree
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


def _build(config: RSIConfig):
    task = make_task(config.task, config.task_config)
    agent = make_discovery_agent(config.discovery_agent, task,
                                 {"seed": config.seed, **config.task_config.get("agent", {})})
    developer = (make_policy_developer(config.policy_developer)
                 if config.mode == DREAM else None)
    return task, agent, developer


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, {"mode": args.mode, "rounds": args.rounds,
                                       "name": args.name, "seed": args.seed,
                                       "out_dir": args.out_dir})
    task, agent, developer = _build(config)
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
        task, agent, developer = _build(config)
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
