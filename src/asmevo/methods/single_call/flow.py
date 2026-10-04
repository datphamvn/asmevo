"""Single-call flow: one LLM call per problem, no search.

    GenerateAsm >> Finish
"""

from __future__ import annotations

from pocketflow import Flow

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.methods.nodes import GenerateAsm, new_store
from asmevo.methods.single_call.nodes import Finish
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "single_call"


def create_flow() -> Flow:
    generate_asm = GenerateAsm()
    generate_asm >> Finish()
    return Flow(start=generate_asm)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Ask once for an optimized version of ``problem.baseline_asm``; no feedback loop."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    create_flow().run(shared)
    return shared["generation"]
