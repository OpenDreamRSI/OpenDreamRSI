#!/usr/bin/env python3
"""Is a frozen history an exact simulator when the environment is a real model?

``dreamrsi fidelity`` answers this with the mock discovery agent, which is
seeded -- so the result proves less than it looks, because a recording of a
deterministic process reproduces it trivially.  The claim only earns its keep
when the environment *cannot* be re-run, which is the only situation in which a
replay simulator is worth having.

This script deploys the same policy live ``--runs`` times through a real coding
agent, which will not produce the same candidates twice, then freezes each tree,
writes it to disk, loads it back, and replays the same policy over it.  What it
reports is the pair:

* the live runs disagree with each other -- the environment is not reproducible;
* each frozen history replays its own live run exactly -- the recording is.

Every cell costs one real discovery-agent call, so keep the grid small.

    python3 scripts/real_replay_fidelity.py --config configs/circle_packing_claude.yaml

Exits non-zero if any replay is inexact, so it works as a regression gate for
whoever has model access.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dreamrsi.agents import make_discovery_agent          # noqa: E402
from dreamrsi.cli import load_config                      # noqa: E402
from dreamrsi.experiments.fidelity import fidelity_probe  # noqa: E402
from dreamrsi.policy import baseline                      # noqa: E402
from dreamrsi.storage import write_json                   # noqa: E402
from dreamrsi.tasks import make_task                      # noqa: E402
from dreamrsi.types import GridPlan                       # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="a run config with a real agent")
    p.add_argument("--runs", type=int, default=2, help="live deployments to compare")
    p.add_argument("--branch-count", dest="branch_count", type=int, default=2)
    p.add_argument("--refine-count", dest="refine_count", type=int, default=1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    args = p.parse_args(argv)

    config = load_config(args.config, {})
    root = args.out_dir or os.path.join(config.out_dir, f"{config.name}-real-fidelity")
    history = os.path.join(root, "history")
    os.makedirs(history, exist_ok=True)

    task = make_task(config.task, config.task_config)
    agent = make_discovery_agent(config.discovery_agent, task,
                                 {"seed": config.seed, **config.task_config.get("agent", {})})
    paths = task.prepare(history)
    base = task.baseline_score(history)
    plan = GridPlan(branch_count=args.branch_count, refine_count=args.refine_count,
                    reason="real replay-fidelity probe")
    cells = plan.branch_count * (plan.refine_count + 1)
    print(f"task {task.name}  baseline {base:.6f}  "
          f"{args.runs} live run(s) of up to {cells} real call(s) each",
          file=sys.stderr, flush=True)

    probes, elapsed = [], []
    for i in range(1, args.runs + 1):
        started = time.time()
        out = fidelity_probe(
            task, agent, baseline.source(),
            run_dir=os.path.join(history, f"r{i:04d}_probe"),
            history_dir=history, baseline_dir=paths["baseline_dir"],
            problem_file=paths["problem_file"], baseline_score=base, plan=plan,
            beta=args.beta, name=f"pi_1_run{i}",
            max_parallelism=config.max_parallelism,
            max_decision_rounds=config.max_decision_rounds, cycle=i)
        probes.append(out)
        elapsed.append(time.time() - started)
        report = out["fidelity"]
        print(f"  live run {i}: {elapsed[-1]:.0f}s, "
              f"{len(out['tree'])} cell(s), exact replay {report.exact}",
              file=sys.stderr, flush=True)

    scored = [{c: n.score for c, n in sorted(o["tree"].nodes.items())} for o in probes]
    reproducible = all(s == scored[0] for s in scored[1:])

    print()
    print(f"{'run':>5} {'seconds':>8} {'cells':>6} {'best':>12} {'exact replay':>13}")
    print("-" * 48)
    for i, (out, secs) in enumerate(zip(probes, elapsed), 1):
        report = out["fidelity"]
        best = "--" if report.live_best is None else f"{report.live_best:.6f}"
        print(f"{i:>5} {secs:>8.0f} {len(out['tree']):>6} {best:>12} {str(report.exact):>13}")
    print("-" * 48)
    print(f"the live runs reproduce each other: {reproducible}")
    print(f"every frozen history replays exactly: "
          f"{all(o['fidelity'].exact for o in probes)}")
    if not reproducible:
        print("\n-> the environment is not reproducible; the history it leaves behind is.")

    payload = {
        "task": task.name,
        "baseline_score": base,
        "live_runs_reproduce_each_other": reproducible,
        "runs": [{"seconds": round(secs, 1),
                  "scores": {c: (None if v == float("-inf") else v)
                             for c, v in s.items()},
                  "fidelity": o["fidelity"].to_dict(),
                  "support": o["support"].to_dict()}
                 for o, s, secs in zip(probes, scored, elapsed)],
    }
    write_json(os.path.join(root, "real_fidelity.json"), payload)
    print(f"\nwrote {os.path.join(root, 'real_fidelity.json')}")
    return 0 if all(o["fidelity"].exact for o in probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
