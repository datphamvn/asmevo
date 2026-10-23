"""Mutation prompt and feedback of the search methods.

A child of the -O3 baseline sees the SuperCoder prompt unchanged, as ``single_call`` does; a
child of an LLM-made candidate also sees that candidate's reply and its feedback.
"""

from __future__ import annotations

from asmevo.data.dataset import Problem, TestCase
from asmevo.methods.feedback import ANSWER, clip, failed_test
from asmevo.methods.single_call.prompt import PROMPT_ID as BASELINE_PROMPT_ID
from asmevo.methods.single_call.prompt import build_prompt
from asmevo.schemas import Candidate, EvalResult, Status
from asmevo.utils.llm.client import Message

MUTATE_PROMPT_ID = "mutate_v0"


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
            body = f"Your program {what} ({clip(result.detail)})."
            if result.failed_test is not None:
                body += "\n" + failed_test(tests, result.failed_test)
            return body
    raise ValueError(f"unknown status {result.status}")


def build_mutation_prompt(problem: Problem, parent: Candidate) -> tuple[list[Message], str]:
    """Messages and prompt id asking for a faster version of ``parent``."""
    if parent.generation == 0:
        return build_prompt(problem), BASELINE_PROMPT_ID
    messages = [
        *build_prompt(problem),
        {"role": "assistant", "content": parent.llm_calls[-1].reply},
        {"role": "user", "content": f"{parent.feedback}\n\n{ANSWER}"},
    ]
    return messages, MUTATE_PROMPT_ID
