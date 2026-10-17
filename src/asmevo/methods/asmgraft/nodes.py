"""AsmGraft nodes and its search policy: Thompson sampling over nodes, then over arms.

``shared["nodes"]`` maps a node id to its record; ids are round indices and ``ROOT`` (-1) is the
-O3 baseline, which scores 1.0x without an LLM call. ``shared["arms"]`` holds the Beta posterior
of every arm in each context, ``shared["grafted"]`` the node pairs already grafted, and
``shared["pending"]`` the child the next LLM call will produce. Each step:

1. A broken latest candidate is repaired, up to ``max_repairs`` times in a row.
2. Otherwise a correct node or the root is drawn by REx's Thompson sampling (Tang et al. 2024):
   ``Beta(1 + C h, 1 + C (1 - h) + N)``, with ``h`` its log speedup min-max normalized over the
   correct nodes and ``N`` the children it already has.
3. An arm is drawn by discounted Thompson sampling within the node's context. At the root the
   arms are a plain draft and one hinted draft per strategy; at a program they are improve, one
   hinted branch per strategy, and a graft with the fastest correct program of another strategy
   line, offered only while such a partner exists.

A pull earns its arm a reward of 1 if its child, after any repairs, beats the pull's reference
speedup by ``min_gain``: the parent's, or for a graft the faster of both parents'.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pocketflow import Node

from asmevo.config import AsmGraftConfig
from asmevo.methods.asmgraft.prompt import (
    HINT_NAMES,
    build_draft,
    build_expansion,
    build_graft,
)

ROOT = -1
DRAFT, IMPROVE, GRAFT, REPAIR = "draft", "improve", "graft", "repair"
CONTEXTS = {
    "root": (DRAFT, *HINT_NAMES),
    "program": (IMPROVE, *HINT_NAMES, GRAFT),
}


@dataclass(frozen=True)
class Pull:
    context: str
    arm: str
    reference: float  # speedup the child must beat by min_gain to earn a reward


@dataclass
class GraftNode:
    parent: int | None
    ok: bool
    speedup: float
    tags: frozenset[str]  # strategy hints on its line; a graft holds both parents' hints
    repairs: int = 0  # consecutive repairs that led here
    children: int = 0  # non-repair children; REx's N
    owed: Pull | None = None  # pull still waiting on this broken line's repairs


@dataclass(frozen=True)
class Pending:
    arm: str  # an arm of the parent's context, or "repair"
    parent: int
    pull: Pull | None  # None for repairs, which settle the pull of the line they repair
    partner: int | None = None  # the second parent of a graft


def new_nodes() -> dict[int, GraftNode]:
    return {ROOT: GraftNode(parent=None, ok=True, speedup=1.0, tags=frozenset())}


def new_arms() -> dict[str, dict[str, list[float]]]:
    """A ``Beta(1, 1)`` prior ``[alpha, beta]`` for every arm of every context."""
    return {context: {arm: [1.0, 1.0] for arm in arms} for context, arms in CONTEXTS.items()}


def gains(speedup: float, reference: float, min_gain: float) -> bool:
    return speedup > reference * (1 + min_gain)


def _ancestors(nodes: dict[int, GraftNode], i: int) -> set[int]:
    found = set()
    while (i := nodes[i].parent) is not None:
        found.add(i)
    return found


def graft_partner(nodes: dict[int, GraftNode], a: int, grafted: set[frozenset[int]],
                  min_gain: float) -> int | None:
    """The fastest program faster than -O3 on another strategy line than ``a``, neither its
    ancestor nor its descendant, and not grafted with it before."""
    node = nodes[a]
    if a == ROOT or not gains(node.speedup, 1.0, min_gain):
        return None
    above = _ancestors(nodes, a)
    partners = [
        b for b, n in nodes.items()
        if b not in above and b not in (ROOT, a) and n.ok and n.tags != node.tags
        and gains(n.speedup, 1.0, min_gain) and frozenset((a, b)) not in grafted
        and a not in _ancestors(nodes, b)
    ]
    return max(partners, key=lambda b: nodes[b].speedup, default=None)


def select_node(nodes: dict[int, GraftNode], rng: random.Random, rex_c: float) -> int:
    ids = [i for i in sorted(nodes) if nodes[i].ok]
    logs = [math.log(nodes[i].speedup) for i in ids]
    lo, hi = min(logs), max(logs)
    draws = []
    for i, x in zip(ids, logs, strict=True):
        h = (x - lo) / (hi - lo) if hi > lo else 1.0
        draws.append(rng.betavariate(1 + rex_c * h, 1 + rex_c * (1 - h) + nodes[i].children))
    return ids[max(range(len(ids)), key=draws.__getitem__)]


def select_arm(posteriors: dict[str, list[float]], arms: tuple[str, ...],
               rng: random.Random) -> str:
    draws = [rng.betavariate(*posteriors[arm]) for arm in arms]
    return arms[max(range(len(arms)), key=draws.__getitem__)]


def choose_action(nodes: dict[int, GraftNode], arms: dict[str, dict[str, list[float]]],
                  grafted: set[frozenset[int]], rng: random.Random,
                  cfg: AsmGraftConfig) -> Pending:
    """Repair the latest candidate if it is broken and has repairs left; otherwise draw a node,
    then an arm.

    Depends only on the nodes so far and a per-problem RNG, never on the remaining budget.
    """
    latest = max(nodes)
    if not nodes[latest].ok and nodes[latest].repairs < cfg.max_repairs:
        return Pending(REPAIR, latest, None)
    parent = select_node(nodes, rng, cfg.rex_c)
    context = "root" if parent == ROOT else "program"
    partner = graft_partner(nodes, parent, grafted, cfg.min_gain)
    offered = tuple(a for a in CONTEXTS[context] if a != GRAFT or partner is not None)
    arm = select_arm(arms[context], offered, rng)
    reference = nodes[parent].speedup
    if arm != GRAFT:
        partner = None
    else:
        reference = max(reference, nodes[partner].speedup)
    return Pending(arm, parent, Pull(context, arm, reference), partner)


def settle(arms: dict[str, dict[str, list[float]]], pull: Pull, won: bool,
           discount: float) -> None:
    """Decay the context's posteriors toward ``Beta(1, 1)``, then credit the pulled arm."""
    for posterior in arms[pull.context].values():
        posterior[0] = 1 + discount * (posterior[0] - 1)
        posterior[1] = 1 + discount * (posterior[1] - 1)
    arms[pull.context][pull.arm][0 if won else 1] += 1


class Decide(Node):
    """Apply the search policy; the next call drafts, improves, branches, grafts, or repairs."""

    def __init__(self, cfg: AsmGraftConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def prep(self, shared):
        return shared["nodes"], shared["arms"], shared["grafted"], shared["rng"]

    def exec(self, prep_res):
        return choose_action(*prep_res, self.cfg)

    def post(self, shared, prep_res, exec_res):
        shared["pending"] = exec_res


class ChildPrompt(Node):
    """Set the conversation for the pending child."""

    def prep(self, shared):
        pending: Pending = shared["pending"]
        problem = shared["problem"]
        if pending.parent == ROOT:
            return build_draft, (problem, pending.arm)

        def program(i: int):
            reply, asm = shared["candidates"][i]
            return reply, asm, shared["rounds"][i]

        if pending.partner is not None:
            return build_graft, (problem, program(pending.parent), program(pending.partner),
                                 pending.partner)
        reply, _, result = program(pending.parent)
        return build_expansion, (problem, pending.arm, (reply, result))

    def exec(self, prep_res):
        build, args = prep_res
        return build(*args)

    def post(self, shared, prep_res, exec_res):
        shared["messages"], shared["prompt_id"] = exec_res
        shared["parent"] = shared["pending"].parent


class Attach(Node):
    """Add the latest evaluated candidate to the tree and settle the pull it completes."""

    def __init__(self, cfg: AsmGraftConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def post(self, shared, prep_res, exec_res):
        nodes, pending, cfg = shared["nodes"], shared["pending"], self.cfg
        result = shared["rounds"][-1]
        parent = nodes[pending.parent]
        if pending.arm == REPAIR:
            tags, repairs, owed = parent.tags, parent.repairs + 1, parent.owed
        else:
            parent.children += 1
            tags, repairs, owed = parent.tags, 0, pending.pull
            if pending.arm in HINT_NAMES:
                tags |= {pending.arm}
            if pending.partner is not None:
                tags |= nodes[pending.partner].tags
                shared["grafted"].add(frozenset((pending.parent, pending.partner)))
        node = GraftNode(parent=pending.parent, ok=result.ok, speedup=result.speedup, tags=tags,
                         repairs=repairs)
        if owed is not None:
            if result.ok:
                settle(shared["arms"], owed, gains(result.speedup, owed.reference, cfg.min_gain),
                       cfg.discount)
            elif repairs >= cfg.max_repairs:
                settle(shared["arms"], owed, False, cfg.discount)
            else:
                node.owed = owed
        nodes[len(shared["rounds"]) - 1] = node
