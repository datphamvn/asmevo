"""Figures for a finished run. Reads results.jsonl and does not rebuild or retime."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from asmevo.schemas import EvalResult, Generation, Status

RESULTS = "results.jsonl"
GENERATIONS = "generations.jsonl"


def load_results(run_dir: Path) -> list[EvalResult]:
    path = run_dir / RESULTS
    if not path.is_file():
        raise SystemExit(f"no {RESULTS} in {run_dir}; run evaluate first")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty")
    return [EvalResult.from_dict(json.loads(line)) for line in lines]


def load_generations(run_dir: Path) -> list[Generation]:
    path = run_dir / GENERATIONS
    if not path.is_file():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [Generation.from_dict(json.loads(line)) for line in lines]


def write_run_plots(run_dir: Path, out_dir: Path) -> list[Path]:
    """Write status and speedup figures. Returns the paths created."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results(run_dir)
    generations = load_generations(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _plot_status(results, out_dir / "status.png", plt),
        _plot_speedups(results, out_dir / "speedups.png", plt),
    ]
    if generations:
        written.append(_plot_tokens(generations, out_dir / "tokens.png", plt))
    return written


def _plot_status(results: list[EvalResult], path: Path, plt) -> Path:
    counts = Counter(r.status.value for r in results)
    labels = [status.value for status in Status]
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(labels, values, color="#4c78a8")
    ax.set_xlabel("problems")
    ax.set_title("Evaluation outcomes")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_speedups(results: list[EvalResult], path: Path, plt) -> Path:
    ordered = sorted(results, key=lambda r: r.problem_id)
    labels = [str(r.problem_id) for r in ordered]
    values = [r.speedup for r in ordered]
    colors = ["#59a14f" if r.ok and r.speedup > 1 else "#4c78a8" if r.ok else "#e15759"
              for r in ordered]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(ordered)), 4))
    ax.bar(labels, values, color=colors)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="gcc -O3")
    ax.set_xlabel("problem id")
    ax.set_ylabel("speedup (failures score 1x)")
    ax.set_title("Speedup against gcc -O3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_tokens(generations: list[Generation], path: Path, plt) -> Path:
    ordered = sorted(generations, key=lambda g: g.problem_id)
    labels = [str(g.problem_id) for g in ordered]
    values = [sum(c.output_tokens for c in g.llm_calls) for g in ordered]
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(ordered)), 4))
    ax.bar(labels, values, color="#f28e2b")
    ax.set_xlabel("problem id")
    ax.set_ylabel("output tokens")
    ax.set_title("LLM output tokens per problem")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
