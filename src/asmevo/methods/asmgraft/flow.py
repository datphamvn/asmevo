"""AsmGraft flow: Thompson sampling picks a node, then an arm; grafts merge strategy lines.

    Decide >> ChildPrompt >> GenerateAsm >> EvaluateAsm - "refine" >> Attach >> Decide
                                            EvaluateAsm - "done"   >> Attach >> PickBest

Every candidate is built, tested, and timed on all of ``problem.tests``, so ``generate`` must
run on Linux x86-64.
"""

from __future__ import annotations

import random

from pocketflow import Flow

from asmevo.config import AsmGraftConfig, Config
from asmevo.data.dataset import Problem
from asmevo.methods.asmgraft.nodes import Attach, ChildPrompt, Decide, new_arms, new_nodes
from asmevo.methods.nodes import EvaluateAsm, GenerateAsm, PickBest, new_store
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "asmgraft"


def create_flow(cfg: AsmGraftConfig) -> Flow:
    if not 0 < cfg.discount <= 1:
        raise ValueError(f"asmgraft.discount must be in (0, 1]; got {cfg.discount}")
    decide = Decide(cfg)
    generate_asm, evaluate_asm = GenerateAsm(), EvaluateAsm(max_rounds=cfg.max_calls)
    decide >> ChildPrompt() >> generate_asm >> evaluate_asm
    evaluate_asm - "refine" >> Attach(cfg) >> decide
    evaluate_asm - "done" >> Attach(cfg) >> PickBest()
    return Flow(start=decide)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Search for ``cfg.asmgraft.max_calls`` LLM calls; keep the fastest correct candidate."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    shared.update(nodes=new_nodes(), arms=new_arms(), grafted=set(),
                  rng=random.Random(problem.id))
    create_flow(cfg.asmgraft).run(shared)
    return shared["generation"]
