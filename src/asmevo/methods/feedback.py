"""Feedback prompts shared by the feedback-driven methods: a child of a program sees the SuperCoder
prompt, the program, and its build/test/time feedback, optionally with a strategy hint."""

from __future__ import annotations

from asmevo.data.dataset import Problem, TestCase
from asmevo.methods.single_call.prompt import build_prompt
from asmevo.schemas import EvalResult, Status
from asmevo.utils.llm.client import Message

HINTS = (
    "Rewrite the algorithm: work out from the C code what the program computes and replace "
    "the computation with an asymptotically faster or closed-form one that prints identical "
    "output for every valid input.",
    "Speed up input and output: replace scanf/printf with one buffered read of stdin, custom "
    "integer parsing and formatting, and direct read/write syscalls.",
    "Optimize the hot loops: strength reduction, unrolling, SSE/AVX2 vectorization, and keeping "
    "values in registers.",
    "Remove branches and improve data layout: cmov/setcc instead of unpredictable branches, "
    "smaller data types, and cache-friendly memory access.",
)

MAX_IO_CHARS = 1500

_ABI = (
    "Preserve the System V AMD64 ABI: integer arguments in rdi, rsi, rdx, rcx, r8, r9; "
    "the return value in rax; callee-saved rbx, rbp, and r12-r15; 16-byte stack alignment "
    "before a call. Do not rename global symbols.\n"
)

ANSWER = "Answer with the complete program in a single ```assembly block."


def clip(text: str) -> str:
    if len(text) <= MAX_IO_CHARS:
        return text
    return f"{text[:MAX_IO_CHARS]}\n... [{len(text) - MAX_IO_CHARS} more characters]"


def failed_test(tests: tuple[TestCase, ...], i: int) -> str:
    """Stdin and expected stdout of ``tests[i]``, clipped."""
    return (
        f"Test {i} stdin:\n```\n{clip(tests[i].stdin.rstrip())}\n```\n"
        f"Expected stdout:\n```\n{clip(tests[i].expected_stdout.rstrip())}\n```\n"
    )


def build_feedback(problem: Problem, result: EvalResult, hint: str | None = None) -> Message:
    """Tell the model how a candidate did on the build/test/time pipeline."""
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
                f"Your program printed the wrong output ({clip(result.detail)}).\n"
                + failed_test(problem.tests, result.failed_test)
            )
        case Status.CRASH | Status.TIMEOUT:
            kind = "crashed" if result.status is Status.CRASH else "timed out"
            body = (
                f"Your program {kind} ({clip(result.detail)}).\n"
                + failed_test(problem.tests, result.failed_test)
            )
    if hint:
        body += f"\nThis time take a different approach. {hint}\n"
    return {"role": "user", "content": f"{body}\n{_ABI}{ANSWER}"}


def build_child_prompt(problem: Problem, parent: tuple[str, EvalResult] | None,
                       hint: str | None) -> list[Message]:
    """Messages for a new child of ``parent`` (``None`` is the -O3 root)."""
    if parent is None:
        if hint is None:
            return build_prompt(problem)
        content = f"Approach to take: {hint}\n\n{problem.standard_prompt}"
        return [{"role": "user", "content": content}]
    reply, result = parent
    messages = [
        *build_prompt(problem),
        {"role": "assistant", "content": reply},
        build_feedback(problem, result, hint),
    ]
    return messages
