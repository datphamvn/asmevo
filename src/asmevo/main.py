"""Command-line entry point.

  uv run asmevo generate --method single_call          # LLM calls; runs on any host
  uv run asmevo generate --method policy_tree          # tree search; Linux x86-64 only
  uv run asmevo generate --method refined_best_of_n    # rounds of N samples; Linux x86-64 only
  uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time; Linux x86-64 only
  uv run asmevo curve runs/policy_tree-<timestamp>     # best of the first k calls; x86-64 only
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from asmevo.config import ROOT, Config, load_config, load_dotenv
from asmevo.data.dataset import Problem, load_test_set
from asmevo.eval.curve import budget_curve, plot_curves
from asmevo.eval.report import summarize
from asmevo.eval.runner import evaluate, pin_work_cpus
from asmevo.methods.policy_tree import flow as policy_tree
from asmevo.methods.refined_best_of_n import flow as refined_best_of_n
from asmevo.methods.single_call import flow as single_call
from asmevo.schemas import Generation
from asmevo.utils.llm.client import OpenAICompatibleClient

# Each method is ``generate(problem, llm, cfg) -> Generation`` running a PocketFlow flow.
METHODS = {
    single_call.NAME: single_call.generate,
    policy_tree.NAME: policy_tree.generate,
    refined_best_of_n.NAME: refined_best_of_n.generate,
}
# build/test/time candidates while generating
EXEC_METHODS = {policy_tree.NAME, refined_best_of_n.NAME}

CONFIG = "config.json"
GENERATIONS = "generations.jsonl"
RESULTS = "results.jsonl"
SUMMARY = "summary.json"
CURVE = "curve.json"


def prepare_timing(cfg: Config, what: str) -> None:
    """Check the host can build and time candidates; call before starting any threads."""
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit(f"{what} needs Linux x86-64; this host is {platform.machine()}")
    if cfg.eval.pin_cpus:
        pin_work_cpus()


def load_problems(cfg: Config, limit: int | None) -> list[Problem]:
    ids = cfg.dataset.problem_ids
    problems = load_test_set(cache_dir=str(ROOT / cfg.dataset.cache_dir),
                             limit=None if ids is not None else limit)
    if ids is not None:
        by_id = {p.id: p for p in problems}
        problems = [by_id[i] for i in ids][:limit]
    return problems


def load_generations(run_dir: Path) -> dict[int, Generation]:
    path = run_dir / GENERATIONS
    if not path.exists():
        return {}
    gens = (Generation.from_dict(json.loads(line)) for line in path.read_text().splitlines())
    return {g.problem_id: g for g in gens}


def generate(cfg: Config, method: str, run_dir: Path) -> None:
    """Run ``method`` on every problem with ``cfg.run.workers`` concurrent LLM calls.

    Stops at the first error; re-running on the same ``run_dir`` resumes where it stopped.
    """
    if method in EXEC_METHODS:
        prepare_timing(cfg, f"generate --method {method}")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.dumps({"method": method, "config": asdict(cfg)}, indent=2) + "\n"
    config_path = run_dir / CONFIG
    if config_path.exists() and config_path.read_text() != snapshot:
        raise SystemExit(f"{run_dir} was started with a different method or config")
    config_path.write_text(snapshot)

    done = load_generations(run_dir)
    todo = [p for p in load_problems(cfg, cfg.dataset.limit) if p.id not in done]
    workers = cfg.run.workers
    print(f"[generate] {method}: {len(todo)} to run, {len(done)} done, {workers} workers")

    llm = OpenAICompatibleClient(cfg.llm)
    method_fn = METHODS[method]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool, (run_dir / GENERATIONS).open("a") as f:
        futures = [pool.submit(method_fn, p, llm, cfg) for p in todo]
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                gen = fut.result()
                f.write(json.dumps(asdict(gen)) + "\n")
                f.flush()  # keep finished calls if a later one fails
                tokens = sum(c.output_tokens for c in gen.llm_calls)
                rounds = (f", rounds {[r.status.value for r in gen.rounds]}, "
                          f"best speedup {max(r.speedup for r in gen.rounds):.3f}"
                          if gen.rounds else "")
                print(f"[generate] {i}/{len(todo)} problem {gen.problem_id}: "
                      f"{'ok' if gen.asm else 'no asm'}, {tokens} output tokens{rounds}, "
                      f"{time.perf_counter() - start:.0f}s elapsed")
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def evaluate_run(cfg: Config, run_dir: Path) -> None:
    prepare_timing(cfg, "evaluate")
    gens = load_generations(run_dir)
    if not gens:
        raise SystemExit(f"no {GENERATIONS} in {run_dir}")
    problems = [p for p in load_problems(cfg, None) if p.id in gens]
    workers = cfg.eval.workers
    print(f"[evaluate] {len(problems)} problems, {workers} workers")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, p, gens[p.id], cfg.eval) for p in problems]
        results = []
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(f"[evaluate] {i}/{len(problems)} problem {r.problem_id}: "
                  f"{r.status.value} speedup={r.speedup:.3f}")
    results.sort(key=lambda r: r.problem_id)

    (run_dir / RESULTS).write_text("".join(json.dumps(asdict(r)) + "\n" for r in results))
    summary = json.dumps(summarize(results, list(gens.values())), indent=2) + "\n"
    (run_dir / SUMMARY).write_text(summary)
    print(summary, end="")


def curve_runs(cfg: Config, run_dirs: list[Path], ks: list[int], out: Path) -> None:
    """Write ``curve.json`` into each run and one plot of all runs to ``out``."""
    prepare_timing(cfg, "curve")
    runs = {}
    for run_dir in run_dirs:
        method = json.loads((run_dir / CONFIG).read_text())["method"]
        name = method if method not in runs else f"{method} ({run_dir.name})"
        runs[name] = list(load_generations(run_dir).values())
    ids = {g.problem_id for gens in runs.values() for g in gens}
    problems = {p.id: p for p in load_problems(cfg, None) if p.id in ids}
    curves = budget_curve(runs, problems, ks, cfg.eval)
    for run_dir, (name, points) in zip(run_dirs, curves.items(), strict=True):
        (run_dir / CURVE).write_text(json.dumps(points, indent=2) + "\n")
        print(f"[curve] {name}")
        for p in points:
            print(f"  k={p['k']:<3d} calls={p['mean_calls_used']:5.1f} "
                  f"correct={p['correct_rate']:.3f} geomean={p['geomean_speedup']:.4f} "
                  f"median={p['speedup_p50']:.4f} avg={p['avg_speedup']:.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_curves(curves, str(out))
    print(f"[curve] plot: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asmevo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="overrides applied on top of "
                   "configs/default.yaml and configs/local.yaml (if it exists)")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--method", choices=sorted(METHODS), default="single_call")
    gen.add_argument("--run-dir", type=Path, help="default: runs/<method>-<timestamp>; "
                     "pass an existing one to resume")

    ev = sub.add_parser("evaluate")
    ev.add_argument("run_dir", type=Path)

    cv = sub.add_parser("curve", help="re-score each run's best result within its first k calls")
    cv.add_argument("run_dirs", type=Path, nargs="+")
    cv.add_argument("--ks", default="1,2,4,8,16,32",
                    type=lambda s: sorted({int(k) for k in s.split(",")}))
    cv.add_argument("--out", type=Path, help="plot path (default: runs/curves-<timestamp>.png)")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = load_config(args.config)

    if args.command == "generate":
        run_dir = args.run_dir or (
            ROOT / cfg.run.output_dir / f"{args.method}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        generate(cfg, args.method, run_dir)
    elif args.command == "evaluate":
        evaluate_run(cfg, args.run_dir)
    else:
        out = args.out or ROOT / cfg.run.output_dir / f"curves-{time.strftime('%Y%m%d-%H%M%S')}.png"
        curve_runs(cfg, args.run_dirs, args.ks, out)


if __name__ == "__main__":
    main()
