"""Refined best-of-n flow: rounds of independent samples from one prompt; the best correct is kept.

    RoundPrompt >> GenerateAsm >> DedupEvaluateAsm - "refine" >> CloseRound >> RoundPrompt
                                  DedupEvaluateAsm - "done"   >> PickBest

"Refined best-of-n" is read as ACCLAIM's iterated best-of-n (arXiv:2604.04238, section 3.4):
rounds of ``samples_per_round`` samples that all get the same prompt. When a round closes, its
fastest correct sample replaces the incumbent (at first the -O3 baseline, 1.0x) if it is more
than ``min_gain`` faster, and the next round's prompt shows the incumbent with its feedback and a
note on the closed round's samples and the tests they failed, or, while the incumbent is still
-O3, the SuperCoder prompt with that note. The first round is the SuperCoder prompt, so a
one-call run is ``single_call``, and with ``samples_per_round >= max_calls`` the one round left
is plain best-of-N.

Every candidate is built, tested, and timed on all of ``problem.tests``, so ``generate`` must
run on Linux x86-64.
"""

from __future__ import annotations

from pocketflow import Flow

from asmevo.config import Config, RefinedBestOfNConfig
from asmevo.data.dataset import Problem
from asmevo.methods.nodes import GenerateAsm, PickBest, new_store
from asmevo.methods.refined_best_of_n.nodes import (
    BASELINE,
    CloseRound,
    DedupEvaluateAsm,
    RoundPrompt,
)
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "refined_best_of_n"


def create_flow(cfg: RefinedBestOfNConfig) -> Flow:
    round_prompt, evaluate_asm = RoundPrompt(), DedupEvaluateAsm(max_rounds=cfg.max_calls)
    round_prompt >> GenerateAsm() >> evaluate_asm
    evaluate_asm - "refine" >> CloseRound(cfg.samples_per_round, cfg.min_gain) >> round_prompt
    evaluate_asm - "done" >> PickBest()
    return Flow(start=round_prompt)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Sample in rounds of ``cfg.refined_best_of_n.samples_per_round``, each refining the best
    so far, for ``cfg.refined_best_of_n.max_calls`` LLM calls; keep the fastest correct."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    shared.update(incumbent=BASELINE, eval_cache={},
                  round_prompt=(build_prompt(problem), PROMPT_ID))
    create_flow(cfg.refined_best_of_n).run(shared)
    return shared["generation"]
