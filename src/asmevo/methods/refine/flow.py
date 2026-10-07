"""Iterative-refinement flow: one trajectory from the -O3 baseline, one candidate per step.

    Step - "mutate" >> Mutate >> Step
    Step - "done"   >> Finish

Step n refines the candidate of step n - 1 from its feedback: its build error, failing test,
or speedup. Each prompt holds only that parent, not the whole history, so calls stay the size
of the other methods'. The trajectory may get worse; ``Finish`` still selects the fittest
candidate of all steps. Every candidate is evaluated, so ``search`` must run on Linux x86-64.
"""

from __future__ import annotations

import random

from pocketflow import Flow

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.methods.nodes import Finish, Mutate, new_store
from asmevo.methods.refine.nodes import Step
from asmevo.schemas import SearchTrace
from asmevo.search.budget import Budget
from asmevo.utils.llm.client import LLMClient

NAME = "refine"


def create_flow() -> Flow:
    step, mutate = Step(), Mutate()
    step - "mutate" >> mutate
    mutate - "done" >> step
    step - "done" >> Finish()
    return Flow(start=step)


def search(problem: Problem, llm: LLMClient, cfg: Config, budget: Budget,
           rng: random.Random, seed: int) -> SearchTrace:
    """Take ``cfg.search.max_candidates`` steps; select the fittest. ``rng`` is unused."""
    shared = new_store(problem, llm, cfg, budget, rng, NAME, seed)
    create_flow().run(shared)
    return shared["trace"]
