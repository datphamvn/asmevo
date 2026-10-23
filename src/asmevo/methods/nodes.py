"""PocketFlow store and nodes shared by the search methods (evolution, best_of_n, refine).

Every search is a ``pocketflow.Flow`` over one shared dict made by ``new_store``:

    problem, llm, cfg, budget, rng, method, seed   inputs, read-only
    candidates      every ``Candidate`` made, the -O3 baseline first; the archive
    scored          ``EvalResult`` per ``asm_hash``, so each distinct program is timed once
    generation      generation of the candidates being made; 0 is the baseline
    parents         one parent per candidate to make in this generation, in order
    offspring       candidates made from ``parents`` so far
    trace           the method's ``SearchTrace``, written by ``Finish``

Nodes carry per-run state, so build a new flow for every (problem, seed).
"""

from __future__ import annotations

import math
import random
from typing import Any

from pocketflow import Node

from asmevo.config import Config
from asmevo.data.dataset import Problem, TestCase
from asmevo.eval.runner import evaluate_asm
from asmevo.methods.prompt import build_mutation_prompt, format_feedback
from asmevo.schemas import Candidate, EvalResult, SearchTrace, Status, candidate_id
from asmevo.search.budget import Budget
from asmevo.search.selection import asm_hash, best
from asmevo.utils.llm.client import LLMClient
from asmevo.utils.llm.parsing import extract_asm

Store = dict[str, Any]


def search_tests(problem: Problem) -> tuple[TestCase, ...]:
    """Inputs a search may build prompts from and select on."""
    return problem.tests  # TODO(T6): only the search split, never the held-out inputs


def fitness(result: EvalResult) -> float:
    """``T_baseline / T_candidate`` if correct on every search test, ``-inf`` otherwise."""
    return result.speedup if result.ok else -math.inf


def new_store(problem: Problem, llm: LLMClient, cfg: Config, budget: Budget,
              rng: random.Random, method: str, seed: int) -> Store:
    baseline = Candidate(
        id=candidate_id(problem.id, method, seed, 0), problem_id=problem.id, method=method,
        seed=seed, generation=0, parent_ids=(), prompt_id=None, asm=problem.baseline_asm,
        asm_hash=asm_hash(problem.baseline_asm), result=None, fitness=1.0, feedback="",
        llm_calls=(),
    )
    # A reply that repeats the baseline is the baseline: 1.0x, without timing it again.
    same = EvalResult(problem_id=problem.id, status=Status.OK, detail="the -O3 baseline")
    return {
        "problem": problem, "llm": llm, "cfg": cfg, "budget": budget, "rng": rng,
        "method": method, "seed": seed, "candidates": [baseline],
        "scored": {baseline.asm_hash: same}, "generation": 0, "parents": [], "offspring": [],
    }


class Mutate(Node):
    """Make one candidate from the next of ``shared["parents"]``: charge the budget, ask the
    model, then build, test, and time the reply unless the same program was scored before.

    Returns ``"mutate"`` while parents remain, then ``"done"``.
    """

    def prep(self, shared):
        parent = shared["parents"][len(shared["offspring"])]
        return (shared["problem"], shared["llm"], shared["cfg"], shared["budget"], parent,
                dict(shared["scored"]))

    def exec(self, prep_res):
        problem, llm, cfg, budget, parent, scored = prep_res
        budget.charge()
        messages, prompt_id = build_mutation_prompt(problem, parent)
        reply, call = llm.complete(messages, prompt_id=prompt_id)
        asm = extract_asm(reply)
        digest = asm_hash(asm) if asm is not None else None
        result = scored.get(digest) or evaluate_asm(problem, asm, search_tests(problem), cfg.eval)
        return prompt_id, call, asm, digest, result

    def post(self, shared, prep_res, exec_res):
        problem, parent = shared["problem"], prep_res[4]
        prompt_id, call, asm, digest, result = exec_res
        child = Candidate(
            id=candidate_id(problem.id, shared["method"], shared["seed"],
                            len(shared["candidates"])),
            problem_id=problem.id, method=shared["method"], seed=shared["seed"],
            generation=shared["generation"], parent_ids=(parent.id,), prompt_id=prompt_id,
            asm=asm, asm_hash=digest, result=result, fitness=fitness(result),
            feedback=format_feedback(result, search_tests(problem)), llm_calls=(call,),
        )
        shared["candidates"].append(child)
        shared["offspring"].append(child)
        if digest is not None:
            shared["scored"].setdefault(digest, result)
        return "mutate" if len(shared["offspring"]) < len(shared["parents"]) else "done"


class Finish(Node):
    """Select the fittest candidate of the archive and write the ``SearchTrace``."""

    def prep(self, shared):
        return shared["problem"], shared["method"], shared["seed"], tuple(shared["candidates"])

    def exec(self, prep_res):
        problem, method, seed, candidates = prep_res
        selected = best(candidates)
        # TODO(T13): final check on the held-out inputs, falling back to the baseline.
        final = selected.result or EvalResult(problem_id=problem.id, status=Status.OK)
        return SearchTrace(problem_id=problem.id, method=method, seed=seed,
                           candidates=candidates, selected_id=selected.id, final=final,
                           fallback=False)

    def post(self, shared, prep_res, exec_res):
        shared["trace"] = exec_res
