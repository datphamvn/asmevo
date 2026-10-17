"""AIDE flow (arXiv:2502.13138): greedy tree search with drafting, debugging, and improving.

    Decide >> ChildPrompt >> GenerateAsm >> EvaluateAsm - "refine" >> Attach >> Decide
                                            EvaluateAsm - "done"   >> Attach >> PickBest

Every candidate is built, tested, and timed on all of ``problem.tests``, so ``generate`` must
run on Linux x86-64.
"""

from __future__ import annotations

import random

from pocketflow import Flow

from asmevo.config import AideConfig, Config
from asmevo.data.dataset import Problem
from asmevo.methods.aide.nodes import Attach, ChildPrompt, Decide
from asmevo.methods.nodes import EvaluateAsm, GenerateAsm, PickBest, new_store
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "aide"


def create_flow(cfg: AideConfig) -> Flow:
    decide = Decide(cfg)
    generate_asm, evaluate_asm = GenerateAsm(), EvaluateAsm(max_rounds=cfg.max_calls)
    decide >> ChildPrompt() >> generate_asm >> evaluate_asm
    evaluate_asm - "refine" >> Attach() >> decide
    evaluate_asm - "done" >> Attach() >> PickBest()
    return Flow(start=decide)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Search for ``cfg.aide.max_calls`` LLM calls; keep the fastest correct candidate."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    shared.update(aide=[], rng=random.Random(problem.id))
    create_flow(cfg.aide).run(shared)
    return shared["generation"]
