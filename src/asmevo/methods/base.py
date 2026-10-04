"""Interface shared by the search methods (best_of_n, refine, evolution).

Each ``methods/<name>/flow.py`` exposes ``NAME: str`` and ``search: SearchMethod``.
"""

from __future__ import annotations

import random
from typing import Protocol

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.schemas import SearchTrace
from asmevo.search.budget import Budget
from asmevo.utils.llm.client import LLMClient


class SearchMethod(Protocol):
    def __call__(self, problem: Problem, llm: LLMClient, cfg: Config,
                 budget: Budget, rng: random.Random) -> SearchTrace:
        """Search for a faster version of ``problem.baseline_asm``.

        Calls ``budget.charge()`` before each LLM-made candidate, draws all randomness from
        ``rng``, and records every candidate (failed ones included) in the trace.
        """
        ...
