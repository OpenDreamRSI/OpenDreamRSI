"""The two prompts of Appendix B, stored verbatim.

``exploration_prompt.md``        Listing 1 -- online exploration.
``policy_improvement_prompt.md`` Listing 2 -- replay-based policy improvement.

Variables enclosed by dollar signs or braces are instantiated by the runtime
system before execution, exactly as the paper describes.
"""

from __future__ import annotations

import os
import string
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))

EXPLORATION = "exploration_prompt.md"
POLICY_IMPROVEMENT = "policy_improvement_prompt.md"


def load(name: str) -> str:
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as fh:
        return fh.read()


def exploration_prompt(**subs: Any) -> str:
    """Listing 1 with its ``$node_dir``-style variables filled in."""
    return string.Template(load(EXPLORATION)).safe_substitute(
        {k: ("" if v is None else str(v)) for k, v in subs.items()}
    )


def policy_improvement_prompt(**subs: Any) -> str:
    """Listing 2 with its ``{method_file}``-style variables filled in."""
    text = load(POLICY_IMPROVEMENT)
    for key, value in subs.items():
        text = text.replace("{" + key + "}", "" if value is None else str(value))
    return text
