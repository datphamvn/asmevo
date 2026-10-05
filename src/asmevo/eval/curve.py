"""Budget curves: how good each method's result would be had it stopped after k LLM calls.

Every method that evaluates in the loop is budget-agnostic, so the first k rounds of a run are
exactly a k-call run. The pick at each k is re-scored on all tests rather than trusting the
in-loop timing, which favours whichever of many noisy measurements happened to be fastest.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from asmevo.config import EvalConfig
from asmevo.data.dataset import Problem
from asmevo.eval.report import summarize
from asmevo.eval.runner import evaluate
from asmevo.methods.nodes import best_round
from asmevo.schemas import EvalResult, Generation


def prefix_pick(gen: Generation, k: int) -> int:
    """Round the method would return had it stopped after ``k`` calls."""
    return best_round(list(gen.rounds[:k]))


def _key(problem_id: int, asm: str | None) -> tuple[int, str | None]:
    return problem_id, None if asm is None else hashlib.sha256(asm.encode()).hexdigest()


def budget_curve(
    runs: dict[str, list[Generation]],
    problems: dict[int, Problem],
    ks: list[int],
    cfg: EvalConfig,
    log: Callable[[str], None] = print,
) -> dict[str, list[dict[str, Any]]]:
    """Per run, one point per k: ``summarize`` of the re-scored picks plus calls used."""
    for name, gens in runs.items():
        if any(not g.rounds or len(g.round_asm) != len(g.rounds) for g in gens):
            raise SystemExit(f"{name}: generations lack per-round candidates (round_asm)")

    picks = {
        (name, k, g.problem_id): (g, prefix_pick(g, k))
        for name, gens in runs.items() for k in ks for g in gens
    }
    todo: dict[tuple[int, str | None], tuple[Problem, Generation]] = {}
    for g, i in picks.values():
        key = _key(g.problem_id, g.round_asm[i])
        todo.setdefault(key, (problems[g.problem_id], replace(
            g, asm=g.round_asm[i], llm_calls=(g.llm_calls[i],), rounds=(), round_asm=(),
            round_parent=())))
    log(f"[curve] re-scoring {len(todo)} distinct picks, {cfg.workers} workers")
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {key: pool.submit(evaluate, p, g, cfg) for key, (p, g) in todo.items()}
        scored: dict[tuple[int, str | None], EvalResult] = {
            key: f.result() for key, f in futures.items()
        }

    curves: dict[str, list[dict[str, Any]]] = {}
    for name, gens in runs.items():
        points = []
        for k in ks:
            results, prefixes = [], []
            for g in gens:
                _, i = picks[name, k, g.problem_id]
                results.append(scored[_key(g.problem_id, g.round_asm[i])])
                prefixes.append(replace(g, llm_calls=g.llm_calls[:k]))
            summary = summarize(results, prefixes)
            summary.pop("dropped_tests")
            points.append({
                "k": k,
                "mean_calls_used": sum(min(k, len(g.rounds)) for g in gens) / len(gens),
                **summary,
                "speedups": {r.problem_id: r.speedup for r in results},
            })
        curves[name] = points
    return curves


def plot_curves(curves: dict[str, list[dict[str, Any]]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "geomean_speedup", "geometric mean speedup"),
                              (axes[1], "speedup_p50", "median speedup (slower counts 1.0x)")]:
        for name, points in curves.items():
            ax.plot([p["mean_calls_used"] for p in points], [p[metric] for p in points],
                    marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("LLM calls per problem (mean used)")
        ax.set_ylabel("faster than gcc -O3")
        lo, hi = ax.get_ylim()
        digits = 1 if hi - lo < 0.1 else 0
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _, d=digits: f"{100 * (y - 1):+.{d}f}%"))
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
