"""Best-of-n nodes."""

from __future__ import annotations

from pocketflow import Node

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
        shared["parent"] = -1
