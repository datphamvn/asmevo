"""Refinement-loop nodes."""

from __future__ import annotations

from pocketflow import Node

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
        shared["parent"] = len(shared["rounds"]) - 1
