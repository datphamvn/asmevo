"""Best-of-n flow: n independent samples of the SuperCoder prompt; keep the fastest correct one.

    GenerateAsm >> EvaluateAsm - "refine" >> Resample >> GenerateAsm
                   EvaluateAsm - "done"   >> PickBest

Every sample is built, tested, and timed on all of ``problem.tests``, so ``generate`` must run
on Linux x86-64.
"""

from __future__ import annotations

from pocketflow import Flow

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.methods.best_of_n.nodes import Resample
from asmevo.methods.nodes import EvaluateAsm, GenerateAsm, PickBest, new_store
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "best_of_n"


def create_flow(n: int) -> Flow:
    generate_asm, evaluate_asm = GenerateAsm(), EvaluateAsm(max_rounds=n)
    generate_asm >> evaluate_asm
    evaluate_asm - "refine" >> Resample() >> generate_asm
    evaluate_asm - "done" >> PickBest()
    return Flow(start=generate_asm)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Sample ``cfg.best_of_n.n`` times; keep the fastest correct sample."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    create_flow(cfg.best_of_n.n).run(shared)
    return shared["generation"]
