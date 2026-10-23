"""Reply parsing shared by all methods."""

from __future__ import annotations

import re

_ASM_BLOCK_RE = re.compile(r"```assembly[ \t]*\n(.*?)```", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```.*?(```|$)", re.DOTALL)


def extract_asm(reply: str) -> str | None:
    """Return the last ```assembly block of ``reply`` (the format every prompt asks for),
    or ``None`` if there is none."""
    blocks = _ASM_BLOCK_RE.findall(reply)
    if not blocks or not blocks[-1].strip():
        return None
    return blocks[-1].strip("\n")


def prose(reply: str) -> str:
    """``reply`` with every code block, closed or not, replaced by a space."""
    return _CODE_BLOCK_RE.sub(" ", reply)


def one_line(text: str, limit: int) -> str:
    """``text`` with its whitespace collapsed, cut to ``limit`` characters."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."
