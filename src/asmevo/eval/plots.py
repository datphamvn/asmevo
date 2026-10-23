"""Figures and a summary table for a finished run. Does not rebuild or retime."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

from asmevo.eval.report import GENERATIONS, load_generations, load_results, summarize
from asmevo.schemas import EvalResult, Generation, Status


def format_run_summary(results: list[EvalResult], generations: list[Generation]) -> str:
    """One text table: speedup, pass rates, and token use from ``summarize``."""
    summary = summarize(results, generations)
    correct_times = [r.candidate_mean_s for r in results if r.ok and r.candidate_mean_s is not None]
    if correct_times:
        median_s = statistics.median(correct_times)
        runtime = f"{median_s:.4f}s (pstdev {statistics.pstdev(correct_times):.4f}s)"
    else:
        runtime = "n/a"
    return "\n".join([
        f"problems:          {summary['problems']}",
        f"geomean speedup:   {summary['geomean_speedup']:.4f}",
        f"median speedup:    {summary['speedup_p50']:.4f}",
        f"speedup p25–p75:   {summary['speedup_p25']:.4f}–{summary['speedup_p75']:.4f}",
        f"build rate:        {summary['build_rate']:.1%}",
        f"correct rate:      {summary['correct_rate']:.1%}",
        f"improved rate:     {summary['improved_rate']:.1%}",
        f"median runtime:    {runtime}",
        f"llm calls:         {summary['llm_calls']}",
        f"input tokens:      {summary['input_tokens']}",
        f"output tokens:     {summary['output_tokens']}",
        f"reasoning tokens:  {summary['reasoning_tokens']}",
        f"median latency:    {summary['median_latency_s']:.2f}s",
        f"status:            {summary['status_counts']}",
    ])


def write_run_plots(run_dir: Path, out_dir: Path) -> tuple[str, list[Path]]:
    """Write the summary and figures. Returns the summary text and the paths created."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(run_dir)
    generations = list(load_generations(run_dir).values())
    if not any(g.llm_calls for g in generations):
        raise SystemExit(f"no LLM calls in {run_dir / GENERATIONS}; cannot summarize tokens")
    out_dir.mkdir(parents=True, exist_ok=True)
    text = format_run_summary(results, generations)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(text + "\n")
    written = [
        summary_path,
        _plot_status(results, out_dir / "status.png", plt),
        _plot_speedups(results, out_dir / "speedups.png", plt),
        _plot_tokens(generations, out_dir / "tokens.png", plt),
    ]
    return text, written


def _plot_status(results: list[EvalResult], path: Path, plt) -> Path:
    counts = Counter(r.status.value for r in results)
    labels = [status.value for status in Status]
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(labels, values, color="#4c78a8")
    ax.set_xlabel("problems")
    ax.set_title("Evaluation outcomes")
    return _save(fig, path, plt)


def _plot_speedups(results: list[EvalResult], path: Path, plt) -> Path:
    values = [r.speedup for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(20, max(5, len(values))), color="#4c78a8", edgecolor="white")
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1, label="gcc -O3")
    ax.set_xlabel("speedup (failures score 1x)")
    ax.set_ylabel("problems")
    ax.set_title("Speedup distribution against gcc -O3")
    ax.legend()
    return _save(fig, path, plt)


def _plot_tokens(generations: list[Generation], path: Path, plt) -> Path:
    values = [sum(c.output_tokens for c in g.llm_calls) for g in generations]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=min(20, max(5, len(values))), color="#f28e2b", edgecolor="white")
    ax.set_xlabel("output tokens")
    ax.set_ylabel("problems")
    ax.set_title("LLM output tokens per problem")
    return _save(fig, path, plt)


def _save(fig, path: Path, plt) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
