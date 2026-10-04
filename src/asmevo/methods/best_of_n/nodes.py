"""Best-of-n nodes."""

from __future__ import annotations

from dataclasses import replace

from pocketflow import Node

from asmevo.methods.nodes import candidate
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt


class Resample(Node):
    """Reset the conversation to the original prompt so the next sample is independent."""

    def prep(self, shared):
        return shared["problem"]

    def exec(self, prep_res):
        return build_prompt(prep_res)

    def post(self, shared, prep_res, exec_res):
        shared["messages"] = exec_res
        shared["prompt_id"] = PROMPT_ID


class PickBest(Node):
    """Return the fastest correct sample, or the last one if none passed every test."""

    def prep(self, shared):
        return shared["rounds"]

    def exec(self, prep_res):
        ok = [i for i, r in enumerate(prep_res) if r.ok]
        return max(ok, key=lambda i: prep_res[i].speedup) if ok else len(prep_res) - 1

    def post(self, shared, prep_res, exec_res):
        shared["generation"] = replace(
            candidate(shared, exec_res),
            llm_calls=tuple(shared["calls"]),
            rounds=tuple(prep_res),
        )
