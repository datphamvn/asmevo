"""Refinement node: one step of a single trajectory, from the candidate made last."""

from __future__ import annotations

from pocketflow import Node


class Step(Node):
    """Make the latest candidate the next one's parent, so a broken candidate is repaired
    from its feedback and a correct one made faster.

    Returns ``"mutate"`` until ``max_candidates`` steps were taken, then ``"done"``.
    """

    def prep(self, shared):
        return shared["generation"], shared["cfg"].search.max_candidates

    def post(self, shared, prep_res, exec_res):
        step, steps = prep_res
        if step >= steps:
            return "done"
        shared["generation"] = step + 1
        shared["parents"] = [shared["candidates"][-1]]
        shared["offspring"] = []
        return "mutate"
