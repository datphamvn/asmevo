"""Refinement-loop nodes."""

from __future__ import annotations

from dataclasses import replace

from pocketflow import Node

from asmevo.methods.nodes import candidate
from asmevo.methods.refinement_loop.prompt import PROMPT_ID, build_feedback


class Feedback(Node):
    """Append the last round's build/test/time result to the conversation."""

    def prep(self, shared):
        return shared["problem"], shared["rounds"][-1]

    def exec(self, prep_res):
        return build_feedback(*prep_res)

    def post(self, shared, prep_res, exec_res):
        shared["messages"].append(exec_res)
        shared["prompt_id"] = PROMPT_ID


class PickBest(Node):
    """Return the fastest correct candidate, or the last one if none passed every test."""

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
