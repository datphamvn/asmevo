"""Policy-tree nodes and the hand-written search policy.

``shared["tree"]`` maps a node id to its record; ids are round indices and ``ROOT`` (-1) is the
-O3 baseline, which scores 1.0x without an LLM call. ``shared["cursor"]`` is the node the policy
is at, and ``shared["pending"]`` the child the next LLM call will produce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pocketflow import Node

from asmevo.config import PolicyTreeConfig
from asmevo.methods.policy_tree.prompt import HINTS, build_child_prompt
from asmevo.schemas import EvalResult

ROOT = -1


@dataclass
class TreeNode:
    parent: int | None
    depth: int
    score: float
    ok: bool
    repairs: int = 0  # consecutive repair expansions that led here
    hint: str | None = None
    children: int = 0
    hints_used: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Pending:
    action: str  # "expand", "branch", or "backtrack"
    parent: int
    hint: str | None


def score(result: EvalResult) -> float:
    """1 for the -O3 baseline, ``1 + log(speedup)`` if correct, 0 otherwise."""
    return 1 + math.log(result.speedup) if result.ok else 0.0


def new_tree() -> dict[int, TreeNode]:
    return {ROOT: TreeNode(parent=None, depth=0, score=1.0, ok=True)}


def _next_hint(node: TreeNode) -> str | None:
    return next((h for h in HINTS if h not in node.hints_used), None)


def _path(tree: dict[int, TreeNode], node_id: int) -> list[int]:
    path = [node_id]
    while (parent := tree[path[-1]].parent) is not None:
        path.append(parent)
    return path


def choose_action(tree: dict[int, TreeNode], cursor: int, stale: int,
                  cfg: PolicyTreeConfig) -> Pending | None:
    """The next child to generate, or ``None`` to terminate.

    Depends only on the tree so far, never on the remaining budget, so the first k calls of a
    longer run are exactly a k-call run.
    """
    if cursor == ROOT and tree[ROOT].children == 0:
        return Pending("expand", ROOT, None)
    if stale >= cfg.patience:
        return None
    node = tree[cursor]
    parent = tree[node.parent]
    if node.ok and node.score - parent.score > math.log1p(cfg.min_gain):
        return Pending("expand", cursor, None)
    if not node.ok and node.repairs < cfg.max_repairs:
        return Pending("expand", cursor, None)
    if node.ok and parent.children < cfg.max_children:
        return Pending("branch", node.parent, _next_hint(parent))

    def room(i: int) -> bool:
        return tree[i].children < cfg.max_children

    # deepest among equally good nodes: it has had the most refinement
    candidates = [i for i in _path(tree, cursor) if room(i)] or [i for i in tree if room(i)]
    if not candidates:
        return None
    target = max(candidates, key=lambda i: (tree[i].score, tree[i].depth))
    return Pending("backtrack", target, _next_hint(tree[target]))


class Decide(Node):
    """Apply the policy at the cursor; returns the action, or ``"terminate"``."""

    def __init__(self, cfg: PolicyTreeConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def prep(self, shared):
        return shared["tree"], shared["cursor"], shared["stale"]

    def exec(self, prep_res):
        return choose_action(*prep_res, self.cfg)

    def post(self, shared, prep_res, exec_res):
        if exec_res is None:
            return "terminate"
        shared["pending"] = exec_res
        return exec_res.action


class Backtrack(Node):
    """Move the cursor to the pending parent; the branch from there follows."""

    def post(self, shared, prep_res, exec_res):
        shared["cursor"] = shared["pending"].parent


class ChildPrompt(Node):
    """Set the conversation for the pending child of a node."""

    def prep(self, shared):
        pending: Pending = shared["pending"]
        parent = None
        if pending.parent != ROOT:
            reply, _ = shared["candidates"][pending.parent]
            parent = (reply, shared["rounds"][pending.parent])
        return shared["problem"], parent, pending.hint

    def exec(self, prep_res):
        return build_child_prompt(*prep_res)

    def post(self, shared, prep_res, exec_res):
        shared["messages"], shared["prompt_id"] = exec_res
        shared["parent"] = shared["pending"].parent


class Attach(Node):
    """Add the latest evaluated candidate to the tree and move the cursor to it."""

    def __init__(self, cfg: PolicyTreeConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def post(self, shared, prep_res, exec_res):
        tree, pending = shared["tree"], shared["pending"]
        result = shared["rounds"][-1]
        parent = tree[pending.parent]
        node_id = len(shared["rounds"]) - 1
        tree[node_id] = TreeNode(
            parent=pending.parent,
            depth=parent.depth + 1,
            score=score(result),
            ok=result.ok,
            repairs=parent.repairs + 1 if not parent.ok else 0,
            hint=pending.hint,
        )
        parent.children += 1
        if pending.hint:
            parent.hints_used.append(pending.hint)
        shared["cursor"] = node_id
        best = max(n.score for i, n in tree.items() if i != node_id)
        if tree[node_id].score - best > math.log1p(self.cfg.min_gain):
            shared["stale"] = 0
        else:
            shared["stale"] += 1
