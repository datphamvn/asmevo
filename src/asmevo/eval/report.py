"""Aggregate per-problem results into the summary reported for each method."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from asmevo.schemas import EvalResult, Generation, Status


def summarize(results: list[EvalResult], generations: list[Generation]) -> dict[str, Any]:
    """Failed candidates count as 1.0x, i.e. the -O3 baseline is kept."""
    n = len(results)
    speedups = [r.speedup for r in results]
    correct = [r for r in results if r.ok]
    calls = [c for g in generations for c in g.llm_calls]
    return {
        "problems": n,
        "status_counts": dict(Counter(r.status.value for r in results)),
        "build_rate": sum(r.status not in {Status.NO_ASM, Status.BUILD_FAIL} for r in results) / n,
        "correct_rate": len(correct) / n,
        "improved_rate": sum(r.speedup > 1.0 for r in correct) / n,
        "mean_speedup": statistics.fmean(speedups),
        "geomean_speedup": math.exp(statistics.fmean(map(math.log, speedups))),
        "mean_speedup_correct_only": (
            statistics.fmean(r.speedup for r in correct) if correct else None
        ),
        "llm_calls": len(calls),
        "incomplete_responses": sum(c.response_status == "incomplete" for c in calls),
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
        "cost_usd": sum(c.cost_usd for c in calls),
        "median_latency_s": statistics.median(c.latency_s for c in calls),
    }
