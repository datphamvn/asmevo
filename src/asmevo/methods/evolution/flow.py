"""Evolution flow: a small population rooted at the -O3 baseline, mutated by the LLM.

    Select >> Mutate - "mutate" >> Mutate
              Mutate - "done"   >> Survive - "select" >> Select
                                   Survive - "done"   >> Finish

Each generation draws ``offspring_per_generation`` parents by tournament, asks the model for a
faster version of each, builds, tests, and times every reply, and keeps the baseline plus the
fittest distinct valid candidates. The first generation can only mutate the baseline.
Every candidate is evaluated, so ``search`` must run on Linux x86-64.
"""

from __future__ import annotations

import random

from pocketflow import Flow

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.methods.evolution.nodes import Select, Survive
from asmevo.methods.nodes import Finish, Mutate, new_store
from asmevo.schemas import SearchTrace
from asmevo.search.budget import Budget
from asmevo.utils.llm.client import LLMClient

NAME = "evolution"


def create_flow() -> Flow:
    select, mutate, survive = Select(), Mutate(), Survive()
    select >> mutate
    mutate - "mutate" >> mutate
    mutate - "done" >> survive
    survive - "select" >> select
    survive - "done" >> Finish()
    return Flow(start=select)


def search(problem: Problem, llm: LLMClient, cfg: Config, budget: Budget,
           rng: random.Random, seed: int) -> SearchTrace:
    """Run ``cfg.search.generations`` generations; select the fittest candidate made."""
    shared = new_store(problem, llm, cfg, budget, rng, NAME, seed)
    shared["population"] = (shared["candidates"][0],)
    create_flow().run(shared)
    return shared["trace"]
