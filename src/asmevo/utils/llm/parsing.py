"""Reply parsing shared by all methods."""

from __future__ import annotations

import re

_ASM_BLOCK_RE = re.compile(r"```assembly[ \t]*\n(.*?)```", re.DOTALL)


def extract_asm(reply: str) -> str | None:
    """Return the last ```assembly block of ``reply`` (the format every prompt asks for),
    or ``None`` if there is none."""
    blocks = _ASM_BLOCK_RE.findall(reply)
    if not blocks or not blocks[-1].strip():
        return None
    return blocks[-1].strip("\n")
