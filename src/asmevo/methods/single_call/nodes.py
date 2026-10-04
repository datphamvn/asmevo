"""Single-call nodes."""

from __future__ import annotations

from pocketflow import Node

from asmevo.methods.nodes import candidate


class Finish(Node):
    """Return the one candidate as the method's ``Generation``."""

    def prep(self, shared):
        return candidate(shared, 0)

    def post(self, shared, prep_res, exec_res):
        shared["generation"] = prep_res
