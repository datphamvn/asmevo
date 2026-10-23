"""Refined best-of-n prompts: a round's samples see the incumbent with its feedback and a note
on the round before, or, while nothing beats -O3, the SuperCoder prompt with that note.

Only the incumbent and the last round's note are shown, so the context does not grow with the
number of rounds.
"""

from __future__ import annotations

from asmevo.data.dataset import Problem
from asmevo.methods.policy_tree.prompt import _clip, _failed_test, build_child_prompt
from asmevo.schemas import EvalResult, Status
from asmevo.utils.llm.client import Message

REFINE_PROMPT_ID = "rbon_refine"
RETRY_PROMPT_ID = "rbon_retry"

# the policy tree's explanation of a reply without assembly
_INCOMPLETE = ("A response status of 'incomplete' means your reply hit the output token limit, "
               "so be more concise.")


def _outcome(result: EvalResult, kept: bool) -> str:
    if result.ok:
        speed = f"{result.speedup:.3f}x as fast as gcc -O3"
        if kept:
            return f"correct, {speed}: the program above"
        return f"correct but {speed}" + (f" ({result.detail})" if result.detail else "")
    test = "" if result.failed_test is None else f" on test {result.failed_test}"
    detail = _clip(result.detail.strip()).replace("\n", "\n  ")  # indented under its line
    return f"{result.status.value}{test}" + (f": {detail}" if detail else "")


def build_note(problem: Problem, results: list[EvalResult], kept: int | None, baseline: bool,
               min_gain: float) -> str:
    """One line per sample of a round: its speedup if correct, else its status and clipped
    detail, then the stdin and expected stdout of each test a sample failed, as the policy tree
    shows them. ``kept`` is the sample that became the incumbent, if any; ``baseline`` says
    whether the incumbent is still the -O3 program."""
    lines = [f"Your last {len(results)} {'attempt' if len(results) == 1 else 'attempts'}:"]
    lines += [f"- attempt {j + 1}: {_outcome(r, j == kept)}" for j, r in enumerate(results)]
    if any(r.status is Status.NO_ASM for r in results):
        lines.append(f"\n{_INCOMPLETE}")
    failed = {r.failed_test: r for r in results if r.failed_test is not None}  # once per test
    if failed:
        lines.append("\n" + "".join(_failed_test(problem, r) for r in failed.values()).rstrip())
    if kept is None:
        than = "gcc -O3" if baseline else "the program above"
        lines.append(f"\nNo correct attempt was more than {min_gain:.1%} faster than {than}.")
    return "\n".join(lines)


def build_round_prompt(problem: Problem, incumbent: tuple[str, EvalResult] | None,
                       note: str) -> tuple[list[Message], str]:
    """Messages and prompt id every sample of the next round gets; ``incumbent`` is the reply
    and result of the program to improve, ``None`` while that is still the -O3 baseline."""
    if incumbent is None:
        content = f"{note}\n\n{problem.standard_prompt}"
        return [{"role": "user", "content": content}], RETRY_PROMPT_ID
    messages, _ = build_child_prompt(problem, incumbent, None)
    feedback = messages[-1]
    messages[-1] = {**feedback, "content": f"{note}\n\n{feedback['content']}"}
    return messages, REFINE_PROMPT_ID
