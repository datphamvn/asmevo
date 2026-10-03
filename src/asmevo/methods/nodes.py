"""PocketFlow store and nodes shared by every method.

Every method is a ``pocketflow.Flow`` over one shared dict made by ``new_store``:

    problem, llm, cfg, method   inputs, read-only
    messages                    conversation so far; ``GenerateAsm`` appends each reply
    prompt_id                   id recorded on the next LLM call
    calls                       every ``LLMCall`` made
    candidates                  ``(reply, asm)`` per LLM call
    rounds                      ``EvalResult`` per evaluated candidate
    generation                  the method's ``Generation``, written by its last node

Nodes carry per-run state, so build a new flow for every problem.
"""

from __future__ import annotations

from typing import Any

from pocketflow import Node

from asmevo.config import Config
from asmevo.data.dataset import Problem
from asmevo.eval.runner import evaluate
from asmevo.schemas import Generation
from asmevo.utils.llm.client import LLMClient, Message
from asmevo.utils.llm.parsing import extract_asm

Store = dict[str, Any]


def new_store(problem: Problem, llm: LLMClient, cfg: Config, method: str,
              messages: list[Message], prompt_id: str) -> Store:
    return {
        "problem": problem, "llm": llm, "cfg": cfg, "method": method,
        "messages": list(messages), "prompt_id": prompt_id,
        "calls": [], "candidates": [], "rounds": [],
    }


def candidate(shared: Store, i: int) -> Generation:
    """``Generation`` holding candidate ``i`` and only the LLM call that produced it."""
    reply, asm = shared["candidates"][i]
    return Generation(
        problem_id=shared["problem"].id,
        method=shared["method"],
        asm=asm,
        reply=reply,
        llm_calls=(shared["calls"][i],),
    )


class GenerateAsm(Node):
    """One LLM call on the conversation so far; records the reply and its assembly."""

    def prep(self, shared):
        return shared["llm"], list(shared["messages"]), shared["prompt_id"]

    def exec(self, prep_res):
        llm, messages, prompt_id = prep_res
        reply, call = llm.complete(messages, prompt_id=prompt_id)
        return reply, extract_asm(reply), call

    def post(self, shared, prep_res, exec_res):
        reply, asm, call = exec_res
        shared["messages"].append({"role": "assistant", "content": reply})
        shared["calls"].append(call)
        shared["candidates"].append((reply, asm))


class EvaluateAsm(Node):
    """Build, test, and time the latest candidate against the -O3 baseline on all tests.

    Returns ``"refine"`` until ``max_rounds`` candidates have been evaluated, then ``"done"``.
    """

    def __init__(self, max_rounds: int) -> None:
        super().__init__()
        self.max_rounds = max_rounds

    def prep(self, shared):
        return shared["problem"], candidate(shared, -1), shared["cfg"].eval

    def exec(self, prep_res):
        return evaluate(*prep_res)

    def post(self, shared, prep_res, exec_res):
        shared["rounds"].append(exec_res)
        return "refine" if len(shared["rounds"]) < self.max_rounds else "done"
