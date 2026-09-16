"""The prompts of Appendix B and the command line."""

import json
import os

from dreamrsi import prompts
from dreamrsi.cli import main


# -- prompts -----------------------------------------------------------------
def test_exploration_prompt_matches_listing_1():
    text = prompts.load(prompts.EXPLORATION)
    assert text.startswith("You must read every historical proposal before proposing")
    for heading in ("## 1. Read the complete history first",
                    "## 2. Learn from both successes and failures",
                    "## 3. Don't converge into a local optimum",
                    "## 4. Propose and implement",
                    "## Files", "## Note:"):
        assert heading in text
    assert "Never execute pkill, kill, killall" in text


def test_exploration_prompt_substitutes_its_dollar_variables():
    out = prompts.exploration_prompt(node_dir="/n", history_dir="/h", baseline_dir="/b",
                                     eval_program="method.py", problem_file="/p.md",
                                     direction_guidance="WIDEN")
    assert "$node_dir" not in out and "$history_dir" not in out
    assert "/n/proposal.md" in out and "WIDEN" in out


def test_policy_prompt_matches_listing_2():
    text = prompts.load(prompts.POLICY_IMPROVEMENT)
    assert "prefix-only exploration policy" in text
    assert "pareto.reward = pareto.auc - lambda * parallel_penalty" in text
    for heading in ("## API", "## Hard constraints",
                    "## Beta: fixed per run, adaptive across cycles",
                    "## Required next-cycle grid planning",
                    "## Learn from history without leaking outcomes",
                    "## Deliverable"):
        assert heading in text
    # the API block the policy is written against
    for name in ("question.reset()", "question.observed()", "question.legal_actions()",
                 "question.legal_roots()", "question.opened_branches()",
                 "question.meta(cell_id)", "question.probe_batch(",
                 "question.baseline_score", "question.max_parallelism"):
        assert name in text


def test_policy_prompt_substitutes_its_brace_variables():
    out = prompts.policy_improvement_prompt(method_file="M.py", history_dir="/h",
                                            trace_pool="/tp")
    assert "{method_file}" not in out and "M.py" in out
    assert "/h/r####_*/" in out


def test_the_api_the_prompt_names_actually_exists():
    """Every symbol Listing 2 tells the agent to import must be importable."""
    from see.policy.api import (  # noqa: F401
        GridPlan,
        GridPlanningContext,
        LLMDesignedMethod,
        SimResult,
        _budget_done,
        _record_curve,
        finalize_result,
    )
    from see.policy.observation_signal import (  # noqa: F401
        branch_failed_hard,
        branch_promising,
        probe_improved_vs_baseline,
        probe_improved_vs_parent,
    )


def test_the_question_api_the_prompt_names_actually_exists():
    """Both phases implement the same tree-based decision interface."""
    from dreamrsi.online import LiveWorld
    from dreamrsi.tree import DiscoveryTree
    from dreamrsi.types import ROOT_ID
    from dreamrsi.world import ReplayWorld

    methods = ("reset", "observed", "legal_actions", "legal_roots",
               "opened_branches", "meta", "probe_batch")
    for cls in (ReplayWorld, LiveWorld):
        for name in methods:
            assert callable(getattr(cls, name))

    tree = DiscoveryTree("w", baseline_score=0.0)
    tree.continue_from(ROOT_ID, score=1.0, evaluated=True, valid=True)
    question = ReplayWorld(tree, max_parallelism=2)
    for name in ("baseline_score", "max_parallelism", "best_so_far", "budget_spent",
                 "decision_rounds", "effective_sequential_rounds", "max_decision_rounds"):
        assert hasattr(question, name)
    meta = question.meta("b0a0")
    for name in ("branch", "attempt", "parent_id", "seq", "tags"):
        assert hasattr(meta, name)


# -- cli ---------------------------------------------------------------------
def test_cli_lists_tasks(capsys):
    assert main(["tasks"]) == 0
    assert "synthetic" in capsys.readouterr().out


def test_cli_prints_prompts(capsys):
    assert main(["prompt", "exploration"]) == 0
    assert "read every historical proposal" in capsys.readouterr().out
    assert main(["prompt", "policy"]) == 0
    assert "prefix-only exploration policy" in capsys.readouterr().out


def test_cli_run_eval_and_inspect(tmp_path, capsys):
    cfg = {"name": "cli", "task": "synthetic", "mode": "fixed", "rounds": 1,
           "max_parallelism": 4, "branch_count": 4, "refine_count": 4,
           "out_dir": str(tmp_path), "discovery_agent": "mock"}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    assert main(["run", "--config", str(cfg_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "fixed" and out["cumulative_agent_calls"] > 0

    pool = str(tmp_path / "cli" / "history")
    policy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "dreamrsi", "policy", "library", "adaptive_portfolio.py")
    assert main(["eval", "--policy", policy, "--pool", pool]) == 0
    ev = json.loads(capsys.readouterr().out)
    assert ev["worlds"] == 1 and "reward" in ev["pareto"]

    trace = str(tmp_path / "cli" / "history" / "r0001_fixed" / "trace.json")
    assert main(["inspect", trace, "-v"]) == 0
    text = capsys.readouterr().out
    assert '"nodes"' in text and "b0a0" in text


def test_cli_dream_over_a_pool(tmp_path, capsys):
    cfg = {"name": "d", "task": "synthetic", "mode": "fixed", "rounds": 2,
           "max_parallelism": 6, "branch_count": 6, "refine_count": 8,
           "max_decision_rounds": 40, "out_dir": str(tmp_path)}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    main(["run", "--config", str(cfg_path)])
    capsys.readouterr()

    policy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "dreamrsi", "policy", "initial_policy.py")
    assert main(["dream", "--policy", policy, "--pool", str(tmp_path / "d" / "history"),
                 "--revisions", "2", "--out-dir", str(tmp_path / "dreamt"),
                 "--max-parallelism", "6", "--max-decision-rounds", "40"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["worlds"] == 2
    assert out["selected_reward"] >= out["incumbent_reward"] - 1e-12
