"""AIDE prompts: drafts and improvements see a memory of earlier correct attempts; debugs do not.

Feedback reuses the ``methods/feedback.py`` wording, so every feedback-driven method reports a
candidate's build/test/time result the same way.
"""

from __future__ import annotations

from asmevo.data.dataset import Problem
from asmevo.methods.feedback import build_feedback
from asmevo.methods.single_call.prompt import PROMPT_ID as ROOT_PROMPT_ID
from asmevo.methods.single_call.prompt import build_prompt
from asmevo.schemas import EvalResult
from asmevo.utils.llm.client import Message
from asmevo.utils.llm.parsing import one_line, prose

DRAFT_PROMPT_ID = "aide_draft"
IMPROVE_PROMPT_ID = "aide_improve"
DEBUG_PROMPT_ID = "aide_debug"

MAX_APPROACH_CHARS = 300


def approach(reply: str) -> str:
    """The reply's prose without code blocks, on one line and clipped."""
    return one_line(prose(reply), MAX_APPROACH_CHARS)


def build_memory(attempts: list[tuple[str, EvalResult]]) -> str:
    """AIDE's journal summary: the approach and speedup of every correct attempt so far."""
    lines = [
        f"- Attempt {i}: {result.speedup:.3f}x as fast as gcc -O3. Approach: {approach(reply)}"
        for i, (reply, result) in enumerate(attempts, 1)
    ]
    return "Memory of previous correct attempts on this task:\n" + "\n".join(lines)


def build_draft_prompt(problem: Problem, memory: str | None) -> tuple[list[Message], str]:
    """A new solution from scratch; with memory, it must differ from the earlier attempts."""
    if memory is None:
        return build_prompt(problem), ROOT_PROMPT_ID
    content = (f"{problem.standard_prompt}\n\n{memory}\n\nWrite a new program that takes a "
               "different approach from these attempts.")
    return [{"role": "user", "content": content}], DRAFT_PROMPT_ID


def build_child_prompt(problem: Problem, parent: tuple[str, EvalResult],
                       memory: str | None) -> tuple[list[Message], str]:
    """Improve a correct parent (with memory) or debug a broken one (without)."""
    reply, result = parent
    feedback = build_feedback(problem, result)
    prompt_id = DEBUG_PROMPT_ID
    if result.ok:
        prompt_id = IMPROVE_PROMPT_ID
        extra = "Propose and implement a single improvement to this program."
        if memory:
            extra = f"{memory}\n\n{extra}"
        feedback = {**feedback, "content": f"{extra}\n\n{feedback['content']}"}
    messages = [*build_prompt(problem), {"role": "assistant", "content": reply}, feedback]
    return messages, prompt_id
