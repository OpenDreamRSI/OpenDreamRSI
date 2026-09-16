"""Policy sources the offline (mock) developer can propose.

These stand in for the code an LLM policy-development agent would write.  They
are stored as importable modules so the repository can run the full recursive
loop with no API key, and read back as text so they travel through exactly the
same load-and-check path as an agent-authored revision.
"""

from __future__ import annotations

import os
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))


def source_of(name: str) -> str:
    with open(os.path.join(HERE, f"{name}.py"), "r", encoding="utf-8") as fh:
        return fh.read()


def available() -> List[str]:
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(HERE)
        if f.endswith(".py") and not f.startswith("_")
    )


def all_sources() -> Dict[str, str]:
    return {n: source_of(n) for n in available()}
