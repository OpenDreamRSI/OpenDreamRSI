"""Load a policy from source and check the hard constraints before running it.

A policy version produced by the policy-development agent is untrusted text: it
may forget ``plan_grid``, rename the class, read ``question.best_so_far`` to
make decisions, or hardcode a winning cell id.  The loader performs a static
check for the constraints the prompt states, then imports the module in an
isolated namespace.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from .api import LLMDesignedMethod

#: Bookkeeping attributes the policy may read for logging but must not use to
#: decide what to explore.
BOOKKEEPING_ATTRS = ("best_so_far", "budget_spent", "best_cell")
#: Attributes that would leak unrevealed outcomes.
FORBIDDEN_ATTRS = ("tree", "_observed", "ceiling", "nodes")


class PolicyRejected(ValueError):
    """The candidate policy violates a hard constraint and is not runnable."""


@dataclass
class PolicySpec:
    """A loaded policy version plus the static findings about it."""

    name: str
    source: str
    cls: Type[LLMDesignedMethod]
    module: types.ModuleType
    path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    digest: str = ""

    def instantiate(self, config: Optional[Dict[str, Any]] = None) -> LLMDesignedMethod:
        return self.cls(dict(config or {}))

    @property
    def overrides_plan_grid(self) -> bool:
        return "plan_grid" in vars(self.cls)


def _static_check(source: str, *, strict: bool) -> List[str]:
    """Static constraints from "Hard constraints" in the improvement prompt."""
    findings: List[str] = []
    tree = ast.parse(source)

    names = {n.targets[0].id for n in tree.body
             if isinstance(n, ast.Assign) and n.targets and isinstance(n.targets[0], ast.Name)}
    if "NAME" not in names:
        findings.append('module must define NAME = "OptimalPolicy"')

    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OptimalPolicy"), None)
    if cls is None:
        findings.append("module must define class OptimalPolicy(LLMDesignedMethod)")
        if strict:
            raise PolicyRejected("; ".join(findings))
        return findings

    methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "solve" not in methods:
        findings.append("OptimalPolicy must implement solve(self, question, budget=None)")
    if "plan_grid" not in methods:
        findings.append("OptimalPolicy must override plan_grid(self, context) -> GridPlan "
                        "on every path; inheriting the template stub is not allowed")

    solve = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "solve"), None)
    if solve is not None:
        for node in ast.walk(solve):
            if isinstance(node, ast.Attribute) and node.attr in BOOKKEEPING_ATTRS:
                base = getattr(node.value, "id", None)
                if base == "question":
                    findings.append(
                        f"solve() reads question.{node.attr}, which is bookkeeping only; "
                        "derive decision statistics from question.observed()"
                    )
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                if getattr(node.value, "id", None) == "question":
                    findings.append(f"solve() reaches into question.{node.attr} (unrevealed outcomes)")

    plan = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "plan_grid"), None)
    if plan is not None:
        returns = [n for n in ast.walk(plan) if isinstance(n, ast.Return)]
        if any(r.value is None or (isinstance(r.value, ast.Constant) and r.value.value is None)
               for r in returns):
            findings.append("plan_grid must always return a non-None GridPlan")
        elif returns and all(_is_bootstrap_call(r.value) for r in returns):
            # A bootstrap plan is the right answer when history is empty or
            # conflicting, but a policy whose *only* answer is the fallback has
            # delegated the grid choice, which the prompt forbids.
            findings.append("plan_grid only ever returns context.bootstrap_plan(...): "
                            "grid choice must not be delegated to the runner's fallback")

    if strict and findings:
        raise PolicyRejected("; ".join(findings))
    return findings


def _is_bootstrap_call(node: Optional[ast.AST]) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "bootstrap_plan")


def load_policy_source(source: str, *, name: str = "candidate_policy",
                       strict: bool = False, path: Optional[str] = None) -> PolicySpec:
    """Import a policy from source text, running the static checks first."""
    warnings = _static_check(source, strict=strict)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    mod_name = f"dreamrsi_policy_{name}_{digest}"

    if path is None:
        fd, path = tempfile.mkstemp(prefix=f"{mod_name}_", suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(source)
    else:
        # Materialise the version on disk so the round is reproducible and the
        # policy-development agent can read back exactly what was evaluated.
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise PolicyRejected(f"cannot import policy from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - the agent wrote this code
        sys.modules.pop(mod_name, None)
        raise PolicyRejected(f"policy failed to import: {exc!r}") from exc

    cls = getattr(module, "OptimalPolicy", None)
    if cls is None or not issubclass(cls, LLMDesignedMethod):
        raise PolicyRejected("policy module must define OptimalPolicy(LLMDesignedMethod)")

    return PolicySpec(
        name=getattr(module, "NAME", name),
        source=source,
        cls=cls,
        module=module,
        path=path,
        warnings=warnings,
        digest=digest,
    )


def load_policy_module(path: str, *, strict: bool = False) -> PolicySpec:
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    return load_policy_source(source, name=os.path.splitext(os.path.basename(path))[0],
                              strict=strict, path=path)
