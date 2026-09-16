"""Importable alias for the initial / baseline policy.

``dreamrsi.policy.baseline.ParallelRefinePolicy`` is the same class as
``dreamrsi.policy.initial_policy.OptimalPolicy``; the module exists so library
code can refer to the baseline by an unambiguous name while the file that is
loaded as ``pi_1`` stays in agent-authored shape.
"""

from __future__ import annotations

import os

from .initial_policy import NAME, OptimalPolicy

ParallelRefinePolicy = OptimalPolicy

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_policy.py")


def source() -> str:
    """The exact text deployed as ``pi_1``."""
    with open(SOURCE_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


__all__ = ["ParallelRefinePolicy", "OptimalPolicy", "NAME", "source", "SOURCE_PATH"]
