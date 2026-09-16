"""The three stages of Dream-RSI, one at a time, with no model access.

Run with::

    python3 examples/quickstart.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dreamrsi.agents import make_discovery_agent, make_policy_developer
from dreamrsi.dreaming import dream
from dreamrsi.online import run_online_cycle
from dreamrsi.policy.baseline import SOURCE_PATH, source as baseline_source
from dreamrsi.policy.loader import load_policy_module
from dreamrsi.storage import RunLayout
from dreamrsi.tasks import make_task
from dreamrsi.types import GridPlan
from dreamrsi.world import SimulatorPool

W, ROUNDS = 6, 40


def main() -> None:
    root = tempfile.mkdtemp(prefix="dreamrsi-quickstart-")
    layout = RunLayout(os.path.join(root, "demo")).create()

    task = make_task("synthetic")
    paths = task.prepare(layout.history_dir)
    baseline = task.baseline_score(layout.history_dir)
    agent = make_discovery_agent("mock", task, {"seed": 0})

    spec = load_policy_module(SOURCE_PATH)
    policy = spec.instantiate({"beta": 0.62})
    plan = GridPlan(branch_count=6, refine_count=12, reason="quickstart")

    # ---- Stage 1: online exploration ---------------------------------------
    print("1. Online exploration")
    pool = SimulatorPool(max_parallelism=W, max_decision_rounds=ROUNDS)
    for t in (1, 2):
        cycle = run_online_cycle(
            task=task, agent=agent, policy=policy, plan=plan,
            run_dir=layout.round_dir(t, "demo"),
            history_dir=layout.history_dir,
            baseline_dir=paths["baseline_dir"],
            problem_file=paths["problem_file"],
            baseline_score=baseline,
            max_parallelism=W, max_decision_rounds=ROUNDS,
            world_id=f"demo-iter{t}", cycle=t,
        )
        m = cycle["manifest"]
        print(f"   round {t}: {m['probes']} probes over {m['decision_rounds']} decision rounds, "
              f"best={m['best_score']:.4f}")
        # ---- Stage 2: the completed tree becomes a replay world ------------
        pool.append(cycle["tree"])

    print(f"\n2. Simulator pool: {len(pool)} world(s)")
    for summary in pool.summary():
        print(f"   {summary['world_id']}: {summary['nodes']} recorded cells, "
              f"{summary['branch_count']} branches, ceiling={summary['ceiling']:.4f}")

    # ---- Stage 3: dreaming --------------------------------------------------
    print("\n3. Dreaming: replaying candidate policies at zero execution cost")
    out = dream(baseline_source(), pool, make_policy_developer("mock"),
                revisions=3, out_dir=os.path.join(root, "dreamt"))
    for ev in out["evaluations"]:
        tag = "  <- selected" if ev.version == out["best"].version else ""
        print(f"   v{ev.version}: reward={ev.score:+.4f} "
              f"auc={ev.summary.auc:.4f} penalty={ev.summary.parallel_penalty:.3f} "
              f"probes={ev.summary.mean_probes:.1f}{tag}")
    print(f"\n   incumbent reward {out['incumbent'].summary.reward:.4f} -> "
          f"selected {out['best'].summary.reward:.4f} "
          f"(never worse: pi^0 is always a candidate)")
    print(f"\n   artifacts: {os.path.join(root, 'dreamt', 'proposal_results')}")


if __name__ == "__main__":
    main()
