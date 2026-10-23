"""EoH flow (arXiv:2401.02051): a population of thought-and-program pairs evolved by the LLM.

    Decide >> ChildPrompt >> GenerateAsm >> EvaluateAsm - "refine" >> Attach >> Decide
                                            EvaluateAsm - "done"   >> PickBest

Every candidate is built, tested, and timed on all of ``problem.tests``, so ``generate`` must
run on Linux x86-64.
"""

from __future__ import annotations

import random

from pocketflow import Flow

from asmevo.config import Config, EohConfig
from asmevo.data.dataset import Problem
from asmevo.methods.eoh.nodes import Attach, ChildPrompt, Decide
from asmevo.methods.nodes import EvaluateAsm, GenerateAsm, PickBest, new_store
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "eoh"


def create_flow(cfg: EohConfig) -> Flow:
    decide = Decide(cfg)
    generate_asm, evaluate_asm = GenerateAsm(), EvaluateAsm(max_rounds=cfg.max_calls)
    decide >> ChildPrompt() >> generate_asm >> evaluate_asm
    evaluate_asm - "refine" >> Attach(cfg) >> decide
    evaluate_asm - "done" >> PickBest()
    return Flow(start=decide)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Evolve for ``cfg.eoh.max_calls`` LLM calls; keep the fastest correct candidate."""
    shared = new_store(problem, llm, cfg, NAME, [], "")
    shared.update(population=[], offspring=[], rng=random.Random(problem.id))
    create_flow(cfg.eoh).run(shared)
    return shared["generation"]
