"""Discovery tasks and the registry the CLI resolves ``task:`` names through."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .base import EvalResult, Task  # noqa: F401

_REGISTRY: Dict[str, Callable[..., Task]] = {}


def register(name: str, factory: Callable[..., Task]) -> None:
    _REGISTRY[name] = factory


def available() -> list:
    _load_all()
    return sorted(_REGISTRY)


def make_task(name: str, config: Optional[Dict[str, Any]] = None) -> Task:
    _load_all()
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](config or {})


def _load_all() -> None:
    from . import autocorrelation, circle_packing, kernelbench, lasso, sum_difference, synthetic  # noqa: F401


def _bootstrap() -> None:
    from .synthetic import SyntheticTask
    register("synthetic", SyntheticTask)


_bootstrap()
