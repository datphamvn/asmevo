"""AsmGraft prompts: drafts, improvements, hinted branches, repairs, and grafts.

Drafts, improvements, branches, and repairs reuse the ``methods/feedback.py`` prompts, so they
report a candidate's build/test/time result exactly as every other feedback-driven method does.
A graft shows two correct programs from different strategy lines and asks for one program that
keeps the optimizations of both.
"""

from __future__ import annotations

from asmevo.data.dataset import Problem
from asmevo.methods.aide.prompt import approach
from asmevo.methods.feedback import ANSWER, build_child_prompt
from asmevo.methods.feedback import HINTS as HINT_TEXTS
from asmevo.schemas import EvalResult
from asmevo.utils.llm.client import Message

HINT_NAMES = ("algorithm", "io", "loops", "branches")
HINTS = dict(zip(HINT_NAMES, HINT_TEXTS, strict=True))

DRAFT_PROMPT_ID = "asmgraft_draft"
IMPROVE_PROMPT_ID = "asmgraft_improve"
BRANCH_PROMPT_ID = "asmgraft_branch"
REPAIR_PROMPT_ID = "asmgraft_repair"
GRAFT_PROMPT_ID = "asmgraft_graft"

Program = tuple[str, str, EvalResult]  # reply, assembly, and result of a correct candidate


def build_draft(problem: Problem, arm: str) -> tuple[list[Message], str]:
    """A program from scratch: the plain SuperCoder prompt, or it with a strategy hint."""
    hint = HINTS.get(arm)
    messages = build_child_prompt(problem, None, hint)
    return messages, DRAFT_PROMPT_ID if hint is None else f"{DRAFT_PROMPT_ID}_{arm}"


def build_expansion(problem: Problem, arm: str,
                    parent: tuple[str, EvalResult]) -> tuple[list[Message], str]:
    """Improve or repair ``parent`` from its feedback, or branch from it with a strategy hint."""
    hint = HINTS.get(arm)
    messages = build_child_prompt(problem, parent, hint)
    if hint is not None:
        return messages, f"{BRANCH_PROMPT_ID}_{arm}"
    return messages, IMPROVE_PROMPT_ID if parent[1].ok else REPAIR_PROMPT_ID


def _program(label: str, program: Program) -> str:
    reply, asm, result = program
    return (f"Program {label} runs {result.speedup:.3f}x as fast as gcc -O3. Its author's notes: "
            f"{approach(reply)}\n```assembly\n{asm}\n```\n")


def build_graft(problem: Problem, a: Program, b: Program,
                partner: int) -> tuple[list[Message], str]:
    """One program that combines the optimizations of ``a`` and ``b``; the id names ``b``."""
    content = (
        f"{problem.standard_prompt}\n\n"
        "Two correct programs for this task make it faster than gcc -O3 in different ways.\n\n"
        f"{_program('A', a)}\n{_program('B', b)}\n"
        "Write one program that combines the optimizations of both, so that it is faster than "
        "either, while printing exactly what the C program prints.\n\n"
        f"{ANSWER}"
    )
    return [{"role": "user", "content": content}], f"{GRAFT_PROMPT_ID}_{partner}"
