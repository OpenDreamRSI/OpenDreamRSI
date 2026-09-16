"""The CLI agent: what a coding-agent CLI is actually handed.

A CLI agent is launched *inside* the attempt directory, which makes two things
matter that no in-process agent ever notices: the paths in its prompt have to
resolve from there, and a sandboxed CLI has to be told which directories it may
read. Both are invisible when they break -- the prompt tells the agent to read
the history first, and an agent that cannot find the history just proposes
without one.
"""

import os
import sys

import pytest

from dreamrsi.agents.base import AttemptContext
from dreamrsi.agents.cli_agent import CLIDiscoveryAgent, CLIPolicyDeveloper, _argv
from dreamrsi.online import LiveWorld
from dreamrsi.tasks import make_task
from dreamrsi.types import GridPlan


class _RecordingAgent:
    """Stands in for a discovery agent, capturing the context it was handed."""

    def __init__(self):
        self.contexts = []

    def propose(self, ctx):
        self.contexts.append(ctx)
        with open(os.path.join(ctx.node_dir, ctx.eval_program), "w", encoding="utf-8") as fh:
            fh.write("import json\nprint(json.dumps({'x': [0.0, 0.0, 0.0, 0.0]}))\n")


def test_every_path_handed_to_an_agent_is_absolute(tmp_path, monkeypatch):
    """A relative run dir must still produce paths a subprocess can resolve."""
    monkeypatch.chdir(tmp_path)
    task = make_task("synthetic", {})
    os.makedirs("rel/history", exist_ok=True)
    paths = task.prepare("rel/history")          # deliberately relative
    agent = _RecordingAgent()

    world = LiveWorld(task, agent, GridPlan(branch_count=1, refine_count=1),
                      run_dir="rel/history/r0001", history_dir="rel/history",
                      baseline_dir=paths["baseline_dir"],
                      problem_file=paths["problem_file"],
                      max_parallelism=1, max_decision_rounds=4)
    world.probe_batch(["b0a0"])
    world.probe_batch(["b0a1"])

    assert agent.contexts
    for ctx in agent.contexts:
        for name in ("node_dir", "run_dir", "history_dir", "baseline_dir", "problem_file"):
            value = getattr(ctx, name)
            assert os.path.isabs(value), f"{name} is relative: {value}"
            assert os.path.exists(value), f"{name} does not exist: {value}"
        for sib in ctx.sibling_dirs:
            assert os.path.isabs(sib)
    # A refinement must point at its parent, absolutely.
    refine = agent.contexts[-1]
    assert refine.parent_dir and os.path.isabs(refine.parent_dir)
    assert os.path.isdir(refine.parent_dir)


def test_the_prompt_the_cli_sees_carries_resolvable_paths(tmp_path, monkeypatch):
    """Listing 1's variables must survive into a prompt an agent can act on."""
    monkeypatch.chdir(tmp_path)
    task = make_task("synthetic", {})
    os.makedirs("rel/history", exist_ok=True)
    paths = task.prepare("rel/history")
    agent = _RecordingAgent()
    world = LiveWorld(task, agent, GridPlan(branch_count=1, refine_count=0),
                      run_dir="rel/history/r0001", history_dir="rel/history",
                      baseline_dir=paths["baseline_dir"],
                      problem_file=paths["problem_file"],
                      max_parallelism=1, max_decision_rounds=4)
    world.probe_batch(["b0a0"])

    from dreamrsi import prompts

    ctx = agent.contexts[0]
    text = prompts.exploration_prompt(**ctx.substitutions())
    assert "$node_dir" not in text and "$history_dir" not in text

    # Every path the prompt names must be absolute, and must live somewhere the
    # agent can reach: the ones it reads already exist, and the ones it is told
    # to write (proposal.md, the eval program) have an existing parent.
    named = [t for line in text.splitlines() for t in line.split("`")
             if t.startswith(str(tmp_path))]
    assert named
    for token in named:
        token = token.rstrip("/")
        assert os.path.isabs(token), token
        assert os.path.exists(token) or os.path.isdir(os.path.dirname(token)), token


# -- the command template ----------------------------------------------------
def test_the_template_exposes_the_paths_the_prompt_names():
    """A sandboxed CLI needs the history directory to honour the prompt."""
    argv = _argv("agent --add-dir {history_dir} --problem {problem_file} -o {program}",
                 prompt_file="/p/PROMPT.md", node_dir="/p", run_dir="/r",
                 history_dir="/h", baseline_dir="/h/baseline",
                 problem_file="/h/problem.md", program="/p/method.py")
    assert argv == ["agent", "--add-dir", "/h", "--problem", "/h/problem.md",
                    "-o", "/p/method.py"]


def test_an_unknown_placeholder_names_itself_and_the_alternatives():
    """A typo in a command template must not surface as a bare KeyError."""
    with pytest.raises(KeyError) as excinfo:
        _argv("agent --add-dir {trace_pool_dir}", prompt_file="/p", node_dir="/p",
              run_dir="/r", history_dir="/h", baseline_dir="/b",
              problem_file="/f", program="/g")
    message = str(excinfo.value)
    assert "trace_pool_dir" in message
    assert "history_dir" in message      # lists what this agent does provide


def test_history_dir_is_the_placeholder_both_agents_share(tmp_path):
    """So one command template can drive discovery and policy development."""
    task = make_task("synthetic", {})
    paths = task.prepare(str(tmp_path))
    command = f"{sys.executable} -c pass --add-dir {{history_dir}}"

    discovery = CLIDiscoveryAgent(task, command=command)
    node_dir = os.path.join(str(tmp_path), "attempt_b0a0")
    os.makedirs(node_dir, exist_ok=True)
    discovery.propose(AttemptContext(
        node_dir=node_dir, run_dir=str(tmp_path), history_dir=str(tmp_path),
        baseline_dir=paths["baseline_dir"], problem_file=paths["problem_file"],
        eval_program=task.eval_program, branch=0, attempt=0))
    assert os.path.exists(os.path.join(node_dir, "PROMPT.md"))

    from dreamrsi.agents.base import RevisionRequest

    developer = CLIPolicyDeveloper(command=command, workdir=str(tmp_path / "dev"))
    source = developer.revise(RevisionRequest(
        method_file="method.py", current_source="NAME = 'x'\n",
        history_dir=str(tmp_path), trace_pool_dir=str(tmp_path / "pool")))
    assert source == "NAME = 'x'\n"      # the stub CLI wrote nothing, so it round-trips


def test_a_failing_cli_reports_its_exit_code(tmp_path):
    """Agent failures are data, but a broken command must say so loudly."""
    task = make_task("synthetic", {})
    paths = task.prepare(str(tmp_path))
    agent = CLIDiscoveryAgent(task, command=f"{sys.executable} -c \"raise SystemExit(3)\"")
    node_dir = os.path.join(str(tmp_path), "attempt_b0a0")
    os.makedirs(node_dir, exist_ok=True)
    with pytest.raises(RuntimeError, match="exited with 3"):
        agent.propose(AttemptContext(
            node_dir=node_dir, run_dir=str(tmp_path), history_dir=str(tmp_path),
            baseline_dir=paths["baseline_dir"], problem_file=paths["problem_file"],
            eval_program=task.eval_program, branch=0, attempt=0))
