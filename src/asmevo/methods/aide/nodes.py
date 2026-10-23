"""AIDE nodes and its search policy (``aideml``'s ``Agent.search_policy``).

``shared["aide"]`` holds one ``AideNode`` per round; drafts have parent ``ROOT`` (-1).
``shared["pending"]`` is the ``(action, parent)`` of the next LLM call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pocketflow import Node

from asmevo.config import AideConfig
from asmevo.methods.aide.prompt import build_child_prompt, build_draft_prompt, build_memory

ROOT = -1


@dataclass
class AideNode:
    parent: int
    ok: bool
    speedup: float
    debug_depth: int  # consecutive debug steps that led here; 0 for drafts and improvements
    children: int = 0


def choose_action(nodes: list[AideNode], rng: random.Random,
                  cfg: AideConfig) -> tuple[str, int]:
    """Draft until ``num_drafts`` drafts exist; then, with ``debug_prob``, debug a random broken
    leaf that is not too deep; otherwise improve the fastest correct node, or draft if none is.

    Depends only on the nodes so far and a per-problem RNG, never on the remaining budget.
    """
    if sum(n.parent == ROOT for n in nodes) < cfg.num_drafts:
        return "draft", ROOT
    if rng.random() < cfg.debug_prob:
        debuggable = [i for i, n in enumerate(nodes)
                      if not n.ok and n.children == 0 and n.debug_depth <= cfg.max_debug_depth]
        if debuggable:
            return "debug", rng.choice(debuggable)
    good = [i for i, n in enumerate(nodes) if n.ok]
    if not good:
        return "draft", ROOT
    return "improve", max(good, key=lambda i: nodes[i].speedup)


class Decide(Node):
    """Apply the search policy; the next call drafts, debugs, or improves."""

    def __init__(self, cfg: AideConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def prep(self, shared):
        return shared["aide"], shared["rng"]

    def exec(self, prep_res):
        return choose_action(*prep_res, self.cfg)

    def post(self, shared, prep_res, exec_res):
        shared["pending"] = exec_res


class ChildPrompt(Node):
    """Set the conversation for the pending call."""

    def prep(self, shared):
        action, parent = shared["pending"]
        correct = [(shared["candidates"][i][0], r) for i, r in enumerate(shared["rounds"]) if r.ok]
        memory = build_memory(correct) if correct else None
        if action == "draft":
            return build_draft_prompt, (shared["problem"], memory)
        reply, _ = shared["candidates"][parent]
        return build_child_prompt, (shared["problem"], (reply, shared["rounds"][parent]), memory)

    def exec(self, prep_res):
        build, args = prep_res
        return build(*args)

    def post(self, shared, prep_res, exec_res):
        shared["messages"], shared["prompt_id"] = exec_res
        shared["parent"] = shared["pending"][1]


class Attach(Node):
    """Record the latest evaluated candidate as a node."""

    def post(self, shared, prep_res, exec_res):
        nodes = shared["aide"]
        action, parent = shared["pending"]
        result = shared["rounds"][-1]
        depth = nodes[parent].debug_depth + 1 if action == "debug" else 0
        nodes.append(AideNode(parent=parent, ok=result.ok, speedup=result.speedup,
                              debug_depth=depth))
        if parent != ROOT:
            nodes[parent].children += 1
