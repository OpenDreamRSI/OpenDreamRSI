"""Discovery agents and policy developers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import (  # noqa: F401
    AttemptContext,
    DiscoveryAgent,
    PolicyDeveloper,
    RevisionRequest,
    write_proposal,
)
from .mock import MockDiscoveryAgent, MockPolicyDeveloper  # noqa: F401


def make_discovery_agent(kind: str, task, config: Optional[Dict[str, Any]] = None) -> DiscoveryAgent:
    config = dict(config or {})
    if kind == "mock":
        return MockDiscoveryAgent(task, seed=config.get("seed", 0),
                                  failure_rate=config.get("failure_rate", 0.12))
    if kind == "cli":
        from .cli_agent import CLIDiscoveryAgent

        return CLIDiscoveryAgent(task, **config)
    if kind in ("llm", "api"):
        from .llm import LLMDiscoveryAgent

        return LLMDiscoveryAgent(task, **config)
    raise KeyError(f"unknown discovery agent {kind!r} (mock | cli | llm)")


def make_policy_developer(kind: str, config: Optional[Dict[str, Any]] = None) -> PolicyDeveloper:
    config = dict(config or {})
    if kind == "mock":
        return MockPolicyDeveloper(**config)
    if kind == "cli":
        from .cli_agent import CLIPolicyDeveloper

        return CLIPolicyDeveloper(**config)
    if kind in ("llm", "api"):
        from .llm import LLMPolicyDeveloper

        return LLMPolicyDeveloper(**config)
    raise KeyError(f"unknown policy developer {kind!r} (mock | cli | llm)")
