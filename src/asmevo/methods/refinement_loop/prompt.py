"""Refinement-loop feedback prompt; round 0 sends ``single_call``'s SuperCoder prompt."""

from __future__ import annotations

from asmevo.data.dataset import Problem
from asmevo.schemas import EvalResult, Status
from asmevo.utils.llm.client import Message

PROMPT_ID = "refine_feedback"

MAX_IO_CHARS = 1500

_ANSWER = "Answer with the complete program in a single ```assembly block."


def _clip(text: str) -> str:
    if len(text) <= MAX_IO_CHARS:
        return text
    return f"{text[:MAX_IO_CHARS]}\n... [{len(text) - MAX_IO_CHARS} more characters]"


def _failed_test(problem: Problem, result: EvalResult) -> str:
    test = problem.tests[result.failed_test]
    return (
        f"Test {result.failed_test} stdin:\n```\n{_clip(test.stdin.rstrip())}\n```\n"
        f"Expected stdout:\n```\n{_clip(test.expected_stdout.rstrip())}\n```\n"
    )


def build_feedback(problem: Problem, result: EvalResult) -> Message:
    """Tell the model how its last candidate did on the build/test/time pipeline."""
    match result.status:
        case Status.OK:
            body = (
                f"Your program is correct on all tests and runs {result.speedup:.3f}x as fast as "
                f"gcc -O3 ({result.candidate_mean_s:.4f}s vs {result.baseline_mean_s:.4f}s "
                "summed over the tests). Make it faster while keeping the output identical.\n"
            )
        case Status.NO_ASM:
            body = (
                f"Nothing could be built: {result.detail}. A response status of 'incomplete' "
                "means your reply hit the output token limit, so be more concise.\n"
            )
        case Status.BUILD_FAIL:
            body = f"Your assembly failed to assemble or link:\n```\n{result.detail}\n```\n"
        case Status.WRONG_OUTPUT:
            body = (
                f"Your program printed the wrong output ({_clip(result.detail)}).\n"
                + _failed_test(problem, result)
            )
        case Status.CRASH | Status.TIMEOUT:
            kind = "crashed" if result.status is Status.CRASH else "timed out"
            body = (
                f"Your program {kind} ({_clip(result.detail)}).\n"
                + _failed_test(problem, result)
            )
    return {"role": "user", "content": f"{body}\n{_ANSWER}"}
