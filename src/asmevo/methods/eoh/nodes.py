"""EoH nodes and its evolution loop (``eoh.py`` with ``prob_rank`` and ``pop_greedy``).

``2 * pop_size`` i1 calls make the initial population; then each operator in turn makes
``pop_size`` offspring from the population as it was before them, after which population and
offspring are cut back to the ``pop_size`` fastest correct programs.

``shared["population"]`` holds the round ids of the population, fastest first;
``shared["offspring"]`` the rounds since it was last cut, and ``shared["pending"]`` the
``Pending`` of the next LLM call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pocketflow import Node

from asmevo.config import EohConfig
from asmevo.methods.eoh.prompt import build_offspring_prompt, thought
from asmevo.schemas import EvalResult

ROOT = -1
CROSSOVER = {"e1", "e2"}


@dataclass(frozen=True)
class Pending:
    operator: str  # "i1", "e1", "e2", "m1", or "m2"
    parents: tuple[int, ...]


def operator(calls: int, cfg: EohConfig) -> str:
    """The operator of call ``calls`` (0-based) in EoH's fixed schedule."""
    init = 2 * cfg.pop_size
    if calls < init:
        return "i1"
    return cfg.operators[(calls - init) // cfg.pop_size % len(cfg.operators)]


def ends_batch(calls: int, cfg: EohConfig) -> bool:
    """Whether the population is cut after the first ``calls`` calls."""
    init = 2 * cfg.pop_size
    return calls >= init and (calls - init) % cfg.pop_size == 0


def select_parents(population: list[int], k: int, rng: random.Random) -> tuple[int, ...]:
    """``prob_rank``: rank r (0 is the fastest) is drawn with weight ``1 / (r + 1 + len)``.

    Unlike EoH's ``random.choices``, a parent is not drawn twice.
    """
    pool = list(enumerate(population))
    parents = []
    for _ in range(min(k, len(pool))):
        weights = [1 / (rank + 1 + len(population)) for rank, _ in pool]
        parents.append(pool.pop(rng.choices(range(len(pool)), weights=weights)[0])[1])
    return tuple(parents)


def manage(ids: list[int], rounds: list[EvalResult], asm: list[str | None],
           size: int) -> list[int]:
    """``pop_greedy``: the ``size`` fastest correct programs, one copy of each."""
    ranked = sorted((i for i in ids if rounds[i].ok), key=lambda i: -rounds[i].speedup)
    seen, kept = set(), []
    for i in ranked:
        if asm[i] not in seen:
            seen.add(asm[i])
            kept.append(i)
    return kept[:size]


def choose_action(calls: int, population: list[int], rng: random.Random,
                  cfg: EohConfig) -> Pending:
    """The scheduled operator, or i1 while no program in the population is correct.

    Depends only on the rounds so far and a per-problem RNG, never on the remaining budget.
    """
    op = operator(calls, cfg)
    if op == "i1" or not population:
        return Pending("i1", ())
    k = cfg.num_parents if op in CROSSOVER else 1
    return Pending(op, select_parents(population, k, rng))


class Decide(Node):
    """Pick the next call's operator and parents."""

    def __init__(self, cfg: EohConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def prep(self, shared):
        return len(shared["rounds"]), shared["population"], shared["rng"]

    def exec(self, prep_res):
        return choose_action(*prep_res, self.cfg)

    def post(self, shared, prep_res, exec_res):
        shared["pending"] = exec_res


class ChildPrompt(Node):
    """Set the conversation for the pending offspring."""

    def prep(self, shared):
        pending: Pending = shared["pending"]
        parents = []
        for i in pending.parents:
            reply, asm = shared["candidates"][i]
            parents.append((thought(reply), asm))
        return shared["problem"], pending.operator, parents

    def exec(self, prep_res):
        return build_offspring_prompt(*prep_res)

    def post(self, shared, prep_res, exec_res):
        shared["messages"], shared["prompt_id"] = exec_res
        shared["parent"] = shared["pending"].parents[0] if shared["pending"].parents else ROOT


class Attach(Node):
    """Add the latest evaluated candidate to the offspring; cut the population after a batch."""

    def __init__(self, cfg: EohConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def post(self, shared, prep_res, exec_res):
        rounds = shared["rounds"]
        shared["offspring"].append(len(rounds) - 1)
        if ends_batch(len(rounds), self.cfg):
            asm = [a for _, a in shared["candidates"]]
            ids = shared["population"] + shared["offspring"]
            shared["population"] = manage(ids, rounds, asm, self.cfg.pop_size)
            shared["offspring"] = []
