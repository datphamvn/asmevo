"""Command-line entry point.

  uv run asmevo generate --method single_call          # LLM calls; runs on any host
  uv run asmevo evaluate runs/single_call-<timestamp>  # build/test/time; Linux x86-64 only
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
from asmevo.eval.report import summarize
from asmevo.eval.runner import evaluate
from asmevo.methods.single_call import flow as single_call
from asmevo.schemas import Generation
from asmevo.utils.llm.client import make_client

METHODS = {single_call.NAME: single_call.generate}

CONFIG = "config.json"
GENERATIONS = "generations.jsonl"
RESULTS = "results.jsonl"
SUMMARY = "summary.json"


def load_problems(cfg: Config, limit: int | None,
                  manifest: dict | None = None) -> tuple[list[Problem], str]:
    """Problems and the SHA-256 of the parquet they came from.

    With a manifest, ``limit`` is ignored: the manifest's problems are returned in its order,
    after checking that the parquet is byte-identical to the one it was made from.
    """
    cache_dir = str(ROOT / cfg.dataset.cache_dir)
    path = download_test_file(cache_dir=cache_dir)
    sha = sha256_file(path)
    if manifest is None:
        return load_test_set(cache_dir=cache_dir, limit=limit, path=path), sha
    try:
        return select_problems(load_test_set(cache_dir=cache_dir, path=path), manifest, sha), sha
    except ValueError as e:
        raise SystemExit(str(e)) from None


def load_generations(run_dir: Path) -> dict[int, Generation]:
    path = run_dir / GENERATIONS
    if not path.exists():
        return {}
    gens = (Generation.from_dict(json.loads(line)) for line in path.read_text().splitlines())
    return {g.problem_id: g for g in gens}


def generate(cfg: Config, method: str, run_dir: Path, source: Path | None = None) -> None:
    """Run ``method`` on every problem with ``cfg.run.workers`` concurrent LLM calls.

    Stops at the first error; re-running on the same ``run_dir`` resumes where it stopped.
    ``source`` (an earlier run's manifest or run dir) fixes the problems and splits to use.
    """
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
    todo = [p for p in problems if p.id not in done]
    workers = cfg.run.workers
    print(f"[generate] {method}: {len(todo)} to run, {len(done)} done, {workers} workers")

    method_fn = METHODS[method]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool, (run_dir / GENERATIONS).open("a") as f:
        futures = [pool.submit(method_fn, p, llm) for p in todo]
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                gen = fut.result()
                f.write(json.dumps(asdict(gen)) + "\n")
                f.flush()  # keep finished calls if a later one fails
                tokens = sum(c.output_tokens for c in gen.llm_calls)
                print(f"[generate] {i}/{len(todo)} problem {gen.problem_id}: "
                      f"{'ok' if gen.asm else 'no asm'}, {tokens} output tokens, "
                      f"{time.perf_counter() - start:.0f}s elapsed")
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def evaluate_run(cfg: Config, run_dir: Path) -> None:
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit(f"evaluate needs Linux x86-64; this host is {platform.machine()}")
    gens = load_generations(run_dir)
    if not gens:
        raise SystemExit(f"no {GENERATIONS} in {run_dir}")
    manifest = read_manifest(run_dir) if (run_dir / MANIFEST).exists() else None
    problems = [p for p in load_problems(cfg, None, manifest)[0] if p.id in gens]
    if manifest:
        record_evaluate(run_dir, cfg)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asmevo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="overrides applied on top of "
                   "configs/default.yaml (default: configs/local.yaml if it exists)")
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--method", choices=sorted(METHODS), default="single_call")
    gen.add_argument("--run-dir", type=Path, help="default: runs/<method>-<timestamp>; "
                     "pass an existing one to resume")
    gen.add_argument("--manifest", type=Path, help="reuse the problems and splits of an "
                     "earlier run (its manifest.json or run dir)")

    ev = sub.add_parser("evaluate")
    ev.add_argument("run_dir", type=Path)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = load_config(args.config)

    if args.command == "generate":
        run_dir = args.run_dir or (
            ROOT / cfg.run.output_dir / f"{args.method}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        generate(cfg, args.method, run_dir, args.manifest)
    else:
        evaluate_run(cfg, args.run_dir)


if __name__ == "__main__":
    main()
