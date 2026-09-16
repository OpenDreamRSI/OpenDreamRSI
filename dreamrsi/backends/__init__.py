"""Engine backends.

The loop this package describes -- explore, freeze, dream, redeploy -- is
implemented natively in :mod:`dreamrsi.online`, :mod:`dreamrsi.world` and
:mod:`dreamrsi.rsi`.  It can also be run on top of `agentdescent
<https://github.com/Birfy/agentdescent>`_, whose ``agentdescent.dream`` module
implements the same three stages against its own primitives.

The point of this subpackage is that swapping the engine must not be visible
from outside.  Cell ids, the policy API, Equation 1, the legality rules, the run
layout and the CLI are the *semantics*; which engine walks the tree is an
implementation detail.  :mod:`~dreamrsi.backends.agentdescent_tree` makes the two
tree models interchangeable and
:mod:`~dreamrsi.backends.agentdescent_engine` makes the two control flows
interchangeable, so the existing test suite is the specification both must meet.

``agentdescent`` is an optional dependency -- ``pip install -e ".[agentdescent]"``
-- and importing this subpackage without it raises a message that says so rather
than an ``ImportError`` from three frames down.
"""

from __future__ import annotations

HINT = ("the agentdescent engine backend needs the `agentdescent` package: "
        "pip install -e \".[agentdescent]\"  (or pip install agentdescent)")


def require_agentdescent():
    """Import ``agentdescent.dream`` or explain what is missing."""
    try:
        from agentdescent import dream
    except ImportError as exc:  # pragma: no cover - exercised by the message
        raise ImportError(HINT) from exc
    return dream


def available() -> bool:
    """Whether the agentdescent backend can be used in this interpreter."""
    try:
        require_agentdescent()
    except ImportError:
        return False
    return True


__all__ = ["require_agentdescent", "available", "HINT"]
