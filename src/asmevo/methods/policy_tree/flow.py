"""Policy-tree flow: a hand-written policy walks a tree of complete programs, rooted at -O3.

    Decide - "expand"    >> ChildPrompt >> GenerateAsm >> EvaluateAsm - "refine" >> Attach >> Decide
    Decide - "branch"    >> ChildPrompt
    Decide - "backtrack" >> Backtrack >> ChildPrompt
    Decide - "terminate" >> PickBest
                                          EvaluateAsm - "done"   >> Attach >> PickBest

Every candidate is built, tested, and timed on all of ``problem.tests``, so ``generate`` must
run on Linux x86-64.
"""

from __future__ import annotations

from pocketflow import Flow

from asmevo.config import Config, PolicyTreeConfig
from asmevo.data.dataset import Problem
from asmevo.methods.nodes import EvaluateAsm, GenerateAsm, PickBest, new_store
from asmevo.methods.policy_tree.nodes import (
    ROOT,
    Attach,
    Backtrack,
    ChildPrompt,
    Decide,
    new_tree,
)
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient

NAME = "policy_tree"


def create_flow(cfg: PolicyTreeConfig) -> Flow:
    decide, child_prompt, pick_best = Decide(cfg), ChildPrompt(), PickBest()
    generate_asm, evaluate_asm = GenerateAsm(), EvaluateAsm(max_rounds=cfg.max_calls)
    decide - "expand" >> child_prompt
    decide - "branch" >> child_prompt
    decide - "backtrack" >> Backtrack() >> child_prompt
    decide - "terminate" >> pick_best
    child_prompt >> generate_asm >> evaluate_asm
    evaluate_asm - "refine" >> Attach(cfg) >> decide
    evaluate_asm - "done" >> Attach(cfg) >> pick_best
    return Flow(start=decide)


def generate(problem: Problem, llm: LLMClient, cfg: Config) -> Generation:
    """Search for at most ``cfg.policy_tree.max_calls`` LLM calls; keep the fastest correct."""
    shared = new_store(problem, llm, cfg, NAME, build_prompt(problem), PROMPT_ID)
    shared.update(tree=new_tree(), cursor=ROOT, stale=0)
    create_flow(cfg.policy_tree).run(shared)
    return shared["generation"]
