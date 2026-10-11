"""Refined best-of-n nodes and the round update.

``shared["incumbent"]`` is the round the current samples refine (``BASELINE``, -1, is the -O3
program at 1.0x), ``shared["round_prompt"]`` the ``(messages, prompt_id)`` every sample of the
current round gets, with the note on the last round in its last message, and
``shared["eval_cache"]`` the ``EvalResult`` of each distinct program evaluated so far.
"""

from __future__ import annotations

import math

from pocketflow import Node

from asmevo.data.dataset import Problem
from asmevo.methods.nodes import EvaluateAsm
from asmevo.methods.policy_tree.nodes import score
from asmevo.methods.refined_best_of_n.prompt import build_note, build_round_prompt
from asmevo.schemas import EvalResult, Status

BASELINE = -1


def normalize_asm(asm: str) -> str:
    """``asm`` without trailing whitespace or blank lines; equal results are one program."""
    # T11's asm_hash replaces this once develop is merged.
    return "\n".join(line.rstrip() for line in asm.splitlines() if line.strip())


def baseline_copy(problem: Problem) -> EvalResult:
    """Result of the -O3 program itself: correct at exactly 1.0x, as SuperCoder scores it.

    Not built or timed, so a re-timed copy cannot win on noise. The timings stay ``None``: at
    1.0x it never becomes the incumbent, the only result whose timings a prompt shows.
    """
    return EvalResult(problem_id=problem.id, status=Status.OK, speedup=1.0,
                      detail="identical to the gcc -O3 baseline, not timed")


def next_incumbent(rounds: list[EvalResult], start: int, incumbent: int, min_gain: float) -> int:
    """The fastest correct of ``rounds[start:]`` (the earliest of equals) if it beats the
    incumbent by more than ``min_gain`` as the policy tree counts gains, else the incumbent.

    Depends only on the rounds so far, never on the remaining budget, so the first k calls of
    a longer run are exactly a k-call run.
    """
    ok = [i for i in range(start, len(rounds)) if rounds[i].ok]
    if not ok:
        return incumbent
    best = max(ok, key=lambda i: rounds[i].speedup)
    reference = 1.0 if incumbent == BASELINE else score(rounds[incumbent])
    return best if score(rounds[best]) - reference > math.log1p(min_gain) else incumbent


class RoundPrompt(Node):
    """Start the next sample from a fresh copy of the round's prompt, so the samples of a round
    are independent draws."""

    def post(self, shared, prep_res, exec_res):
        messages, shared["prompt_id"] = shared["round_prompt"]
        shared["messages"] = list(messages)
        shared["parent"] = shared["incumbent"]


class DedupEvaluateAsm(EvaluateAsm):
    """``EvaluateAsm`` that builds, tests, and times each distinct program once.

    A repeat reuses the cached result and a copy of the -O3 baseline scores 1.0x untimed; either
    still counts as one call and one round. A reply without assembly is never cached.
    """

    def prep(self, shared):
        problem, gen, cfg = super().prep(shared)
        key = None if gen.asm is None else normalize_asm(gen.asm)
        cached = None if key is None else shared["eval_cache"].get(key)
        return problem, gen, cfg, key, cached

    def exec(self, prep_res):
        problem, gen, cfg, key, cached = prep_res
        if cached is not None:
            return cached
        if key is not None and key == normalize_asm(problem.baseline_asm):
            return baseline_copy(problem)
        return super().exec((problem, gen, cfg))

    def post(self, shared, prep_res, exec_res):
        key = prep_res[3]
        if key is not None:
            shared["eval_cache"][key] = exec_res
        return super().post(shared, prep_res, exec_res)


class CloseRound(Node):
    """After a round's last sample, update the incumbent and write the next round's prompt;
    mid-round, do nothing, so the next sample gets the same prompt."""

    def __init__(self, samples_per_round: int, min_gain: float) -> None:
        super().__init__()
        self.samples_per_round = samples_per_round
        self.min_gain = min_gain

    def prep(self, shared):
        if len(shared["rounds"]) % self.samples_per_round:
            return None
        return shared["problem"], shared["candidates"], shared["rounds"], shared["incumbent"]

    def exec(self, prep_res):
        if prep_res is None:
            return None
        problem, candidates, rounds, incumbent = prep_res
        start = len(rounds) - self.samples_per_round
        best = next_incumbent(rounds, start, incumbent, self.min_gain)
        kept = best - start if best != incumbent else None
        note = build_note(problem, rounds[start:], kept, best == BASELINE, self.min_gain)
        program = None if best == BASELINE else (candidates[best][0], rounds[best])
        return best, build_round_prompt(problem, program, note)

    def post(self, shared, prep_res, exec_res):
        if exec_res is not None:
            shared["incumbent"], shared["round_prompt"] = exec_res
