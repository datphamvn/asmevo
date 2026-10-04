"""Refinement-loop flow: the SuperCoder prompt, then a fixed number of execution-feedback rounds.

    GenerateAsm >> EvaluateAsm - "refine" >> Feedback >> GenerateAsm
                   EvaluateAsm - "done"   >> PickBest

Every candidate is built, tested, and timed on all of ``problem.tests``, so ``generate`` must
run on Linux x86-64.
"""

from __future__ import annotations

from pocketflow import Flow

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.methods.nodes import EvaluateAsm, GenerateAsm, PickBest, new_store
from asmevo.methods.refinement_loop.nodes import Feedback
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "refinement_loop"


def create_flow(rounds: int) -> Flow:
    generate_asm, evaluate_asm = GenerateAsm(), EvaluateAsm(max_rounds=rounds)
    generate_asm >> evaluate_asm
    evaluate_asm - "refine" >> Feedback() >> generate_asm
    evaluate_asm - "done" >> PickBest()
    return Flow(start=generate_asm)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Refine for ``cfg.refinement_loop.rounds`` LLM calls; keep the fastest correct one."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    create_flow(cfg.refinement_loop.rounds).run(shared)
    return shared["generation"]
