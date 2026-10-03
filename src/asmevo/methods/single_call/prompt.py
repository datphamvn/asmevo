"""Single-call prompt: the benchmark's own SuperCoder prompt, sent unchanged."""

from __future__ import annotations

from asmevo.data.dataset import Problem
from asmevo.utils.llm.client import Message

PROMPT_ID = "supercoder_standard"


def build_prompt(problem: Problem) -> list[Message]:
    """C code + -O3 assembly; the model must answer in a single ```assembly block."""
    return [{"role": "user", "content": problem.standard_prompt}]
