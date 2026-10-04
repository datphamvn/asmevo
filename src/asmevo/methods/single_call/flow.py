"""Single-call flow: one LLM call per problem, no search."""

from __future__ import annotations

from asmevo.data.dataset import Problem
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient
from asmevo.utils.llm.parsing import extract_asm

NAME = "single_call"


def generate(problem: Problem, llm: LLMClient) -> Generation:
    """Ask once for an optimized version of ``problem.baseline_asm``; no feedback loop."""
    reply, call = llm.complete(build_prompt(problem), prompt_id=PROMPT_ID)
    return Generation(
        problem_id=problem.id,
        method=NAME,
        asm=extract_asm(reply),
        llm_calls=(call,),
    )
