"""Agents backed by a coding-agent CLI (Gemini CLI, Claude Code, Codex, ...).

The paper drives Gemini-3.1-Pro and Gemini-3.7-Flash through the Gemini CLI.
Nothing here is specific to one vendor: a CLI agent is any command that can be
launched inside a directory with a prompt and is trusted to edit files there.

The command template accepts ``{prompt_file}``, ``{node_dir}``, ``{run_dir}``
and ``{program}``.  The prompt is also piped on stdin, which is what most CLIs
accept by default.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, Optional

from .. import prompts
from .base import AttemptContext, DiscoveryAgent, PolicyDeveloper, RevisionRequest

DEFAULT_COMMAND = "gemini --yolo"


class CLIDiscoveryAgent(DiscoveryAgent):
    """Run a coding-agent CLI inside the attempt directory.

    The agent is handed Listing 1 verbatim with its ``$node_dir``-style
    variables filled in, and is expected to write ``proposal.md`` and the task's
    eval program.  Everything else in the run directory is read-only by
    convention -- the prompt says so, and the runner only ever reads back those
    two files.
    """

    name = "cli-discovery-agent"

    def __init__(self, task, *, command: str = DEFAULT_COMMAND, timeout: float = 1800.0,
                 env: Optional[Dict[str, str]] = None, **_: Any) -> None:
        self.task = task
        self.command = command
        self.timeout = float(timeout)
        self.env = dict(env or {})

    def propose(self, ctx: AttemptContext) -> None:
        prompt = prompts.exploration_prompt(**ctx.substitutions())
        prompt_file = os.path.join(ctx.node_dir, "PROMPT.md")
        os.makedirs(ctx.node_dir, exist_ok=True)
        with open(prompt_file, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        argv = shlex.split(self.command.format(
            prompt_file=prompt_file, node_dir=ctx.node_dir, run_dir=ctx.run_dir,
            program=os.path.join(ctx.node_dir, ctx.eval_program)))
        proc = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                              cwd=ctx.node_dir, timeout=self.timeout,
                              env={**os.environ, **self.env})
        with open(os.path.join(ctx.node_dir, "agent_stdout.txt"), "w", encoding="utf-8") as fh:
            fh.write(proc.stdout or "")
        if proc.returncode != 0:
            raise RuntimeError(
                f"{argv[0]} exited with {proc.returncode}: "
                + ((proc.stderr or "").strip().splitlines() or [""])[-1]
            )


class CLIPolicyDeveloper(PolicyDeveloper):
    """Improve the exploration policy with a coding-agent CLI.

    The CLI is pointed at ``method.py`` with Listing 2 and the archived replay
    feedback for the round.  It must leave a complete policy in ``method.py``.
    """

    name = "cli-policy-developer"

    def __init__(self, *, command: str = DEFAULT_COMMAND, timeout: float = 1800.0,
                 workdir: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                 **_: Any) -> None:
        self.command = command
        self.timeout = float(timeout)
        self.workdir = workdir
        self.env = dict(env or {})

    def revise(self, request: RevisionRequest) -> str:
        workdir = self.workdir or os.path.join(request.history_dir, "_policy_dev")
        os.makedirs(workdir, exist_ok=True)
        method_path = os.path.join(workdir, os.path.basename(request.method_file))
        with open(method_path, "w", encoding="utf-8") as fh:
            fh.write(request.current_source)

        prompt = prompts.policy_improvement_prompt(
            method_file=method_path,
            history_dir=request.history_dir,
            trace_pool=request.trace_pool_dir,
        ) + "\n\n## Replay feedback for this revision\n\n```json\n" + _json(request.feedback) + "\n```\n"
        prompt_file = os.path.join(workdir, f"PROMPT_r{request.revision_index}.md")
        with open(prompt_file, "w", encoding="utf-8") as fh:
            fh.write(prompt)

        argv = shlex.split(self.command.format(
            prompt_file=prompt_file, node_dir=workdir, run_dir=workdir, program=method_path))
        proc = subprocess.run(argv, input=prompt, capture_output=True, text=True,
                              cwd=workdir, timeout=self.timeout,
                              env={**os.environ, **self.env})
        if proc.returncode != 0:
            raise RuntimeError(f"{argv[0]} exited with {proc.returncode}")
        with open(method_path, "r", encoding="utf-8") as fh:
            return fh.read()


def _json(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)
