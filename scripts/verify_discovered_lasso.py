#!/usr/bin/env python3
"""Compile and check the Lasso solver Dream-RSI discovered (Appendix C).

Requires a C++ compiler and Eigen.  Point at Eigen with ``--eigen`` or
``$EIGEN_INCLUDE`` if it is not in a standard prefix.

    python3 scripts/verify_discovered_lasso.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dreamrsi.tasks import make_task              # noqa: E402
from dreamrsi.tasks.lasso import find_eigen       # noqa: E402

DISCOVERED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "discovered", "lasso_path", "method.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eigen", default=None, help="path to the Eigen include dir")
    ap.add_argument("--instances", type=int, default=9)
    ap.add_argument("--timing", type=int, default=3)
    ap.add_argument("--lambdas", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    eigen = find_eigen(args.eigen)
    if eigen is None:
        print("Eigen not found. Install it (brew install eigen / apt install libeigen3-dev)\n"
              "or pass --eigen /path/to/eigen3.", file=sys.stderr)
        return 2
    print(f"Eigen: {eigen}")

    task = make_task("lasso", {"n_instances": args.instances, "n_timing": args.timing,
                               "n_lambda": args.lambdas, "repeats": args.repeats,
                               "eigen_include": eigen})
    root = tempfile.mkdtemp(prefix="dreamrsi-lasso-")
    task.prepare(root)
    print(f"sklearn baseline R_search = {task.baseline_score(root):.3f}")

    node = os.path.join(root, "discovered")
    os.makedirs(node, exist_ok=True)
    shutil.copyfile(DISCOVERED, os.path.join(node, task.eval_program))
    result = task.evaluate(node)

    if result.fail_class != "ok":
        print(f"FAILED ({result.fail_class}):\n{result.error}", file=sys.stderr)
        return 1
    print(f"discovered solver  R_search = {result.score:.3f}")
    print(json.dumps(result.diagnostics, indent=2))
    print(f"\nobjective check passed on {result.n_valid}/{result.n_total} held-out instances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
