"""Best-of-N flow: independent samples of the -O3 baseline, no feedback between them.

    Mutate - "mutate" >> Mutate
    Mutate - "done"   >> Finish

Every sample is a child of the baseline, so each call sends the unchanged SuperCoder prompt,
as ``single_call`` does; the samples differ only through the model's own sampling. Every
sample is evaluated, so ``search`` must run on Linux x86-64.
"""

from __future__ import annotations

import random

from pocketflow import Flow

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.methods.nodes import Finish, Mutate, new_store
from asmevo.schemas import SearchTrace
from asmevo.search.budget import Budget
from asmevo.utils.llm.client import LLMClient

NAME = "best_of_n"


def create_flow() -> Flow:
    mutate = Mutate()
    mutate - "mutate" >> mutate
    mutate - "done" >> Finish()
    return Flow(start=mutate)


def search(problem: Problem, llm: LLMClient, cfg: Config, budget: Budget,
           rng: random.Random, seed: int) -> SearchTrace:
    """Draw ``cfg.search.max_candidates`` samples; select the fittest. ``rng`` is unused."""
    shared = new_store(problem, llm, cfg, budget, rng, NAME, seed)
    shared.update(generation=1, parents=[shared["candidates"][0]] * cfg.search.max_candidates)
    create_flow().run(shared)
    return shared["trace"]
