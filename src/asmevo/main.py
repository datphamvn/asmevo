"""Command-line entry point.

  uv run asmevo generate --method single_call          # LLM calls; runs on any host
  uv run asmevo generate --method refined_best_of_n    # rounds of N samples; Linux x86-64 only
  uv run asmevo generate --method refinement_loop      # LLM calls + feedback; Linux x86-64 only
  uv run asmevo generate --method aide                 # AIDE tree search; Linux x86-64 only
  uv run asmevo generate --method eoh                  # EoH population search; x86-64 only
  uv run asmevo generate --method asmgraft             # bandit tree search; Linux x86-64 only
  uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time; Linux x86-64 only
  uv run asmevo curve runs/asmgraft-<timestamp>        # best of the first k calls; x86-64 only
  uv run asmevo plots runs/<run>                      # figures from results.jsonl; any host
  uv run asmevo gcc-baseline --limit 5                 # dataset -O3 vs our gcc -O3 -S
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from asmevo.config import ROOT, Config, load_config, load_dotenv
from asmevo.data.dataset import Problem, download_test_file, load_test_set, sha256_file
from asmevo.data.manifest import (
    MANIFEST,
    REPRODUCIBLE,
    build_manifest,
    read_manifest,
    record_evaluate,
    select_problems,
    write_manifest,
)
from asmevo.eval import gcc_baseline
from asmevo.eval.curve import budget_curve, plot_curves
from asmevo.eval.plots import write_run_plots
from asmevo.eval.report import GENERATIONS, RESULTS, load_generations, summarize
from asmevo.eval.runner import evaluate, pin_work_cpus, require_linux_x86_64
from asmevo.methods.aide import flow as aide
from asmevo.methods.asmgraft import flow as asmgraft
from asmevo.methods.eoh import flow as eoh
from asmevo.methods.refined_best_of_n import flow as refined_best_of_n
from asmevo.methods.refinement_loop import flow as refinement_loop
from asmevo.methods.single_call import flow as single_call
from asmevo.utils.llm.client import make_client

# Each method is ``generate(problem, llm, cfg) -> Generation`` running a PocketFlow flow.
METHODS = {
    single_call.NAME: single_call.generate,
    refined_best_of_n.NAME: refined_best_of_n.generate,
    refinement_loop.NAME: refinement_loop.generate,
    aide.NAME: aide.generate,
    eoh.NAME: eoh.generate,
    asmgraft.NAME: asmgraft.generate,
}
# build/test/time candidates while generating
EXEC_METHODS = set(METHODS) - {single_call.NAME}

CONFIG = "config.json"
SUMMARY = "summary.json"
CURVE = "curve.json"


def prepare_timing(cfg: Config) -> None:
    """Check the host can build and time candidates; call before starting any threads."""
    require_linux_x86_64()
    if cfg.eval.pin_cpus:
        pin_work_cpus()


def stamped(cfg: Config, name: str) -> Path:
    return ROOT / cfg.run.output_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"


def load_problems(cfg: Config, limit: int | None,
                  manifest: dict | None = None) -> tuple[list[Problem], str]:
    """Problems and the SHA-256 of the parquet they came from.

    Without a manifest, the first ``limit`` problems, or of ``dataset.problem_ids`` in that
    order if set. With a manifest, ``limit`` and ``problem_ids`` are ignored: the manifest's
    problems are returned in its order, after checking that the parquet is byte-identical to
    the one it was made from.
    """
    cache_dir = str(ROOT / cfg.dataset.cache_dir)
    path = download_test_file(cache_dir=cache_dir)
    sha = sha256_file(path)
    if manifest is None:
        ids = cfg.dataset.problem_ids
        problems = load_test_set(cache_dir=cache_dir, limit=None if ids is not None else limit,
                                 path=path)
        if ids is not None:
            by_id = {p.id: p for p in problems}
            problems = [by_id[i] for i in ids][:limit]
        return problems, sha
    try:
        return select_problems(load_test_set(cache_dir=cache_dir, path=path), manifest, sha), sha
    except ValueError as e:
        raise SystemExit(str(e)) from None


def check_in_manifest(run_dir: Path, gen_ids: set[int], problems: list[Problem]) -> None:
    extra = sorted(gen_ids - {p.id for p in problems})
    if extra:
        raise SystemExit(f"{run_dir} has generations for problems {extra} that are not in "
                         f"its {MANIFEST}")


def generate(cfg: Config, method: str, run_dir: Path, source: Path | None = None) -> None:
    """Run ``method`` on every problem with ``cfg.run.workers`` concurrent LLM calls.

    Stops at the first error; re-running on the same ``run_dir`` resumes where it stopped.
    ``source`` (an earlier run's manifest or run dir) fixes the problems and splits to use.
    """
    if method in EXEC_METHODS:
        prepare_timing(cfg)
    llm = make_client(cfg.llm)  # fail on a missing key or price before touching run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.dumps({"method": method, "config": asdict(cfg)}, indent=2) + "\n"
    config_path = run_dir / CONFIG
    if config_path.exists() and config_path.read_text() != snapshot:
        raise SystemExit(f"{run_dir} was started with a different method or config")
    config_path.write_text(snapshot)

    if source is None and (run_dir / MANIFEST).exists():
        source = run_dir
    base = read_manifest(source) if source else None
    problems, sha = load_problems(cfg, cfg.dataset.limit, base)
    manifest = build_manifest(problems, dataset_sha256=sha, cfg=cfg, seed=None)
    if base:
        manifest.update({k: base[k] for k in REPRODUCIBLE})
    try:
        write_manifest(run_dir, manifest)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    done = load_generations(run_dir)
    check_in_manifest(run_dir, set(done), problems)
    todo = [p for p in problems if p.id not in done]
    workers = cfg.run.workers
    print(f"[generate] {method}: {len(todo)} to run, {len(done)} done, {workers} workers")

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
    prepare_timing(cfg)
    gens = load_generations(run_dir)
    if not gens:
        raise SystemExit(f"no {GENERATIONS} in {run_dir}")
    manifest = read_manifest(run_dir) if (run_dir / MANIFEST).exists() else None
    problems, _ = load_problems(cfg, None, manifest)
    if manifest:
        check_in_manifest(run_dir, set(gens), problems)
        record_evaluate(run_dir, cfg)
    problems = [p for p in problems if p.id in gens]
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


def gcc_baseline_run(cfg: Config, out_dir: Path, limit: int | None, ids: list[int] | None,
                     cc: str | None) -> None:
    """Compare the dataset's -O3 baseline with ``cc -O3 -S`` of the same C code (T7)."""
    prepare_timing(cfg)
    ev = replace(cfg.eval, cc=cc) if cc else cfg.eval
    cc_version = gcc_baseline.gcc_version(ev.cc)
    if ids:
        cfg, limit = replace(cfg, dataset=replace(cfg.dataset, problem_ids=tuple(ids))), None
    problems, _ = load_problems(cfg, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gcc-baseline] {len(problems)} problems with {cc_version}")

    rows = []
    for i, p in enumerate(problems, 1):
        r = gcc_baseline.compare(p, ev)
        rows.append(r)
        timing = f"ratio={r.ratio:.3f}" if r.ratio is not None else r.detail[:80]
        print(f"[gcc-baseline] {i}/{len(problems)} problem {p.id}: "
              f"identical={r.identical} diff={r.diff_lines} {r.ours_status} {timing}")

    (out_dir / "comparison.jsonl").write_text(
        "".join(json.dumps(asdict(r)) + "\n" for r in rows))
    summary = json.dumps(gcc_baseline.summarize(rows, cc_version), indent=2) + "\n"
    (out_dir / SUMMARY).write_text(summary)
    print(summary, end="")


def curve_runs(cfg: Config, run_dirs: list[Path], ks: list[int], out: Path) -> None:
    """Write ``curve.json`` into each run and one plot of all runs to ``out``."""
    prepare_timing(cfg)
    runs = {}
    for run_dir in run_dirs:
        method = json.loads((run_dir / CONFIG).read_text())["method"]
        name = method if method not in runs else f"{method} ({run_dir.name})"
        runs[name] = list(load_generations(run_dir).values())
    ids = {g.problem_id for gens in runs.values() for g in gens}
    problems = {p.id: p for p in load_problems(cfg, None)[0] if p.id in ids}
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
    gen.add_argument("--manifest", type=Path, help="reuse the problems and splits of an "
                     "earlier run (its manifest.json or run dir)")

    ev = sub.add_parser("evaluate")
    ev.add_argument("run_dir", type=Path)

    cv = sub.add_parser("curve", help="re-score each run's best result within its first k calls")
    cv.add_argument("run_dirs", type=Path, nargs="+")
    cv.add_argument("--ks", default="1,2,4,8,16,32",
                    type=lambda s: sorted({int(k) for k in s.split(",")}))
    cv.add_argument("--out", type=Path, help="plot path (default: runs/curves-<timestamp>.png)")

    pl = sub.add_parser("plots", help="plot status, speedup, and tokens from a finished run")
    pl.add_argument("run_dir", type=Path)
    pl.add_argument("--out", type=Path, help="figure directory (default: <run_dir>/plots)")

    gb = sub.add_parser("gcc-baseline", help="compare the dataset -O3 baseline with ours")
    gb.add_argument("--limit", type=int, default=5, help="first N problems (default: 5)")
    gb.add_argument("--ids", type=int, nargs="+", help="problem ids; overrides --limit")
    gb.add_argument("--cc", help="compiler to use instead of eval.cc, e.g. gcc-12")
    gb.add_argument("--out", type=Path, help="default: runs/gcc_baseline-<timestamp>")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = load_config(args.config)

    if args.command == "generate":
        generate(cfg, args.method, args.run_dir or stamped(cfg, args.method), args.manifest)
    elif args.command == "evaluate":
        evaluate_run(cfg, args.run_dir)
    elif args.command == "gcc-baseline":
        gcc_baseline_run(cfg, args.out or stamped(cfg, "gcc_baseline"), args.limit, args.ids,
                         args.cc)
    elif args.command == "plots":
        out = args.out or args.run_dir / "plots"
        text, written = write_run_plots(args.run_dir, out)
        print(text)
        print("[plots] " + ", ".join(str(path) for path in written))
    else:
        out = args.out or stamped(cfg, "curves").with_suffix(".png")
        curve_runs(cfg, args.run_dirs, args.ks, out)


if __name__ == "__main__":
    main()
