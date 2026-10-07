"""Mutation prompt and feedback shared by the search methods.

TODO(T9): stand-in until ``methods/common/prompt.py`` lands; swap the imports in
``methods/nodes.py`` then. A child of the -O3 baseline sees the SuperCoder prompt unchanged,
as ``single_call`` does; a child of an LLM-made candidate also sees that candidate's reply and
its feedback.
"""

from __future__ import annotations

from asmevo.data.dataset import Problem, TestCase
from asmevo.methods.single_call.prompt import PROMPT_ID as BASELINE_PROMPT_ID
from asmevo.methods.single_call.prompt import build_prompt
from asmevo.schemas import Candidate, EvalResult, Status
from asmevo.utils.llm.client import Message

MUTATE_PROMPT_ID = "mutate_v0"

MAX_IO_CHARS = 1500

_ANSWER = "Answer with the complete program in a single ```assembly block."


def _clip(text: str) -> str:
    if len(text) <= MAX_IO_CHARS:
        return text
    return f"{text[:MAX_IO_CHARS]}\n... [{len(text) - MAX_IO_CHARS} more characters]"


def _failed_test(tests: tuple[TestCase, ...], i: int) -> str:
    return (
        f"Test {i} stdin:\n```\n{_clip(tests[i].stdin.rstrip())}\n```\n"
        f"Expected stdout:\n```\n{_clip(tests[i].expected_stdout.rstrip())}\n```\n"
    )


def format_feedback(result: EvalResult, tests: tuple[TestCase, ...]) -> str:
    """How a candidate did on ``tests``, the inputs it was evaluated on (never held-out)."""
    match result.status:
        case Status.OK:
            return (
                f"Your program is correct on all tests and runs {result.speedup:.3f}x as fast "
                "as gcc -O3. Make it faster while keeping the output identical."
            )
        case Status.NO_ASM:
            return (
                f"Nothing could be built: {result.detail}. If your reply hit the output token "
                "limit, be more concise."
            )
        case Status.BUILD_FAIL:
            return f"Your assembly failed to assemble or link:\n```\n{result.detail}\n```"
        case Status.WRONG_OUTPUT | Status.CRASH | Status.TIMEOUT:
            what = {Status.WRONG_OUTPUT: "printed the wrong output", Status.CRASH: "crashed",
                    Status.TIMEOUT: "timed out"}[result.status]
            body = f"Your program {what} ({_clip(result.detail)})."
            if result.failed_test is not None:
                body += "\n" + _failed_test(tests, result.failed_test)
            return body
    raise ValueError(f"unknown status {result.status}")


def build_mutation_prompt(problem: Problem, parent: Candidate) -> tuple[list[Message], str]:
    """Messages and prompt id asking for a faster version of ``parent``."""
    if parent.generation == 0:
        return build_prompt(problem), BASELINE_PROMPT_ID
    messages = [
        *build_prompt(problem),
        {"role": "assistant", "content": parent.llm_calls[-1].reply},
        {"role": "user", "content": f"{parent.feedback}\n\n{_ANSWER}"},
    ]
    return messages, MUTATE_PROMPT_ID
