"""Agents backed directly by a model API (Anthropic or OpenAI).

These are single-shot variants of the CLI agents: the model is given the paper's
prompt plus the readable context and must answer with one fenced code block,
which the runner writes to disk.  They are useful when no coding-agent CLI is
available, and they keep the same interface, so the orchestration layer is
unchanged.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from .. import prompts
from .base import AttemptContext, DiscoveryAgent, PolicyDeveloper, RevisionRequest, write_proposal

FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = FENCE.findall(text or "")
    if not blocks:
        return (text or "").strip()
    return max(blocks, key=len).strip()


class _Client:
    """Thin provider shim so both agents share one call path."""

    def __init__(self, provider: str = "anthropic", model: Optional[str] = None,
                 max_tokens: int = 16000, temperature: float = 1.0) -> None:
        self.provider = provider
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        if provider == "anthropic":
            import anthropic

            self.model = model or "claude-opus-5"
            self._client = anthropic.Anthropic()
        elif provider == "openai":
            import openai

            self.model = model or "gpt-5"
            self._client = openai.OpenAI()
        else:
            raise KeyError(f"unknown provider {provider!r} (anthropic | openai)")

    def complete(self, prompt: str) -> str:
        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        resp = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class LLMDiscoveryAgent(DiscoveryAgent):
    """One model call per ``CONTINUE``: read the history, write a new candidate."""

    name = "llm-discovery-agent"

    def __init__(self, task, *, provider: str = "anthropic", model: Optional[str] = None,
                 max_history_chars: int = 60000, **kwargs: Any) -> None:
        self.task = task
        self.client = _Client(provider, model, **{k: v for k, v in kwargs.items()
                                                  if k in ("max_tokens", "temperature")})
        self.max_history_chars = int(max_history_chars)

    def propose(self, ctx: AttemptContext) -> None:
        prompt = prompts.exploration_prompt(**ctx.substitutions())
        prompt += "\n\n---\n\n## Problem\n\n" + _read(ctx.problem_file)
        prompt += "\n\n## Baseline\n\n```\n" + _read(
            os.path.join(ctx.baseline_dir, ctx.eval_program)) + "\n```\n"
        if ctx.parent_dir:
            prompt += ("\n\n## Workspace you are resuming (" + os.path.basename(ctx.parent_dir)
                       + ")\n\n```\n" + _read(os.path.join(ctx.parent_dir, ctx.eval_program))
                       + "\n```\n")
        prompt += "\n\n## Complete history\n\n" + _history_digest(ctx, self.max_history_chars)
        prompt += (
            "\n\n---\n\nRespond with exactly two fenced blocks: first a markdown block holding "
            f"`proposal.md`, then a code block holding the complete `{ctx.eval_program}`.\n"
        )

        text = self.client.complete(prompt)
        blocks = FENCE.findall(text)
        if len(blocks) >= 2:
            proposal, program = blocks[0].strip(), blocks[-1].strip()
        else:
            proposal, program = "(model returned no separate proposal)", extract_code(text)
        if not program:
            raise RuntimeError("model returned no program")
        write_proposal(ctx.node_dir, proposal)
        os.makedirs(ctx.node_dir, exist_ok=True)
        with open(os.path.join(ctx.node_dir, ctx.eval_program), "w", encoding="utf-8") as fh:
            fh.write(program + "\n")


class LLMPolicyDeveloper(PolicyDeveloper):
    """One model call per revision: rewrite the exploration policy."""

    name = "llm-policy-developer"

    def __init__(self, *, provider: str = "anthropic", model: Optional[str] = None,
                 **kwargs: Any) -> None:
        self.client = _Client(provider, model, **{k: v for k, v in kwargs.items()
                                                  if k in ("max_tokens", "temperature")})

    def revise(self, request: RevisionRequest) -> str:
        prompt = prompts.policy_improvement_prompt(
            method_file=request.method_file,
            history_dir=request.history_dir,
            trace_pool=request.trace_pool_dir,
        )
        prompt += ("\n\n---\n\n## Current policy (`" + request.method_file + "`)\n\n```python\n"
                   + request.current_source + "\n```\n")
        prompt += ("\n## Replay feedback from the last evaluation\n\n```json\n"
                   + json.dumps(request.feedback, indent=2, default=str) + "\n```\n")
        prompt += ("\n---\n\nRespond with exactly one fenced python block containing the complete "
                   "new contents of the file. No prose outside the block.\n")
        source = extract_code(self.client.complete(prompt))
        if "class OptimalPolicy" not in source:
            raise RuntimeError("model did not return a module defining OptimalPolicy")
        return source


def _read(path: str, limit: int = 20000) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()[:limit]
    except OSError:
        return "(unavailable)"


def _history_digest(ctx: AttemptContext, limit: int) -> str:
    """Every sibling proposal with its measured score, newest last."""
    parts: List[str] = []
    for d in list(ctx.sibling_dirs):
        proposal = os.path.join(d, "proposal.md")
        score = os.path.join(d, "eval", "score.json")
        if not os.path.exists(proposal):
            continue
        verdict = _read(score, 2000) if os.path.exists(score) else "(not evaluated)"
        parts.append(f"### {os.path.basename(d)}\n\n{_read(proposal, 4000)}\n\n"
                     f"measured result:\n```json\n{verdict}\n```\n")
    text = "\n".join(parts)
    return text[-limit:] if len(text) > limit else (text or "(no previous attempts)")
