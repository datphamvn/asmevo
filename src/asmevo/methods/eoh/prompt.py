"""EoH prompts (arXiv:2401.02051): every individual is a one-sentence thought plus its program.

Each prompt is the SuperCoder prompt followed by the operator's instruction, worded as in the
EoH code with "algorithm" read as "program". As in EoH, parents are shown without their scores.
"""

from __future__ import annotations

import re

from asmevo.data.dataset import Problem
from asmevo.utils.llm.client import Message
from asmevo.utils.llm.parsing import one_line, prose

MAX_THOUGHT_CHARS = 500

_FORMAT = (
    "Answer in this format instead of the one above: first, describe your new program and its "
    "main steps in one sentence. The description must be inside a brace. Next, implement it as "
    "the complete program in a single ```assembly block. Do not give additional explanations."
)

OPERATORS = {
    "i1": _FORMAT,
    "e1": "Please help me create a new program that has a totally different form from the given "
          "ones.\n" + _FORMAT,
    "e2": "Please help me create a new program that has a totally different form from the given "
          "ones but can be motivated from them.\n"
          "Answer in this format instead of the one above: firstly, identify the common backbone "
          "idea in the provided programs. Secondly, based on the backbone idea describe your new "
          "program in one sentence. The description must be inside a brace. Thirdly, implement it "
          "as the complete program in a single ```assembly block. Do not give additional "
          "explanations.",
    "m1": "Please assist me in creating a new program that has a different form but can be a "
          "modified version of the program provided.\n" + _FORMAT,
    "m2": "Please identify the main parameters of the program provided (such as unroll factors, "
          "vector widths, buffer and table sizes, or loop order) and assist me in creating a new "
          "program that has different settings of these parameters.\n" + _FORMAT,
}

_BRACE_RE = re.compile(r"\{(.*)\}", re.DOTALL)


def prompt_id(operator: str) -> str:
    return f"eoh_{operator}"


def thought(reply: str) -> str:
    """The braced description in the reply's prose (all of the prose if there is none),
    on one line and clipped."""
    text = prose(reply)
    m = _BRACE_RE.search(text)
    return one_line(m.group(1) if m else text, MAX_THOUGHT_CHARS) or "(no description)"


def _parents(parents: list[tuple[str, str]]) -> str:
    if len(parents) == 1:
        desc, asm = parents[0]
        return (f"I have one program with its code as follows.\nProgram description: {desc}\n"
                f"Code:\n```assembly\n{asm}\n```\n")
    blocks = "".join(f"No. {i} program and the corresponding code are:\n{desc}\n"
                     f"```assembly\n{asm}\n```\n"
                     for i, (desc, asm) in enumerate(parents, 1))
    return f"I have {len(parents)} existing programs with their codes as follows:\n{blocks}"


def build_offspring_prompt(problem: Problem, operator: str,
                           parents: list[tuple[str, str]]) -> tuple[list[Message], str]:
    """Messages and prompt id for one offspring; ``parents`` are ``(thought, asm)`` pairs."""
    body = OPERATORS[operator] if not parents else f"{_parents(parents)}{OPERATORS[operator]}"
    content = f"{problem.standard_prompt}\n\n{body}"
    return [{"role": "user", "content": content}], prompt_id(operator)
