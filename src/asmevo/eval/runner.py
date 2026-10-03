"""Isolated build -> test -> time pipeline for one candidate against the -O3 baseline.

Must run on Linux x86-64 (natively or in the docker image): the dataset assembly is x86-64.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

from asmevo.config import EvalConfig
from asmevo.data.dataset import Problem, TestCase
from asmevo.schemas import EvalResult, Generation, Status


def build(asm: str, workdir: Path, name: str, cfg: EvalConfig) -> tuple[Path | None, str]:
    """Assemble and link ``asm``; return ``(exe, "")`` or ``(None, toolchain stderr)``."""
    src, exe = workdir / f"{name}.s", workdir / name
    src.write_text(asm.rstrip("\n") + "\n")
    try:
        proc = subprocess.run(
            [cfg.cc, str(src), "-o", str(exe), *cfg.link_flags],
            capture_output=True,
            text=True,
            timeout=cfg.compile_timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "assemble/link timed out"
    if proc.returncode != 0:
        return None, proc.stderr[-2000:]
    return exe, ""


def outputs_match(actual: str, expected: str) -> bool:
    """Compare line by line, ignoring trailing whitespace and trailing blank lines."""

    def norm(s: str) -> list[str]:
        return [line.rstrip() for line in s.rstrip().splitlines()]

    return norm(actual) == norm(expected)


def check_tests(
    exe: Path, tests: tuple[TestCase, ...], cfg: EvalConfig
) -> tuple[Status, int | None, str]:
    """Run every test under memory/CPU limits; stop at the first failure."""
    cpu_s = math.ceil(cfg.run_timeout_s) + 1
    cmd = ["prlimit", f"--as={cfg.memory_limit_mb << 20}", f"--cpu={cpu_s}", "--", str(exe)]
    for i, test in enumerate(tests):
        try:
            proc = subprocess.run(
                cmd,
                input=test.stdin,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=cfg.run_timeout_s,
                cwd=exe.parent,
            )
        except subprocess.TimeoutExpired:
            return Status.TIMEOUT, i, f"exceeded {cfg.run_timeout_s}s"
        if proc.returncode < 0:
            return Status.CRASH, i, f"killed by signal {-proc.returncode}: {proc.stderr[-500:]}"
        if not outputs_match(proc.stdout, test.expected_stdout):
            return Status.WRONG_OUTPUT, i, f"got {proc.stdout[:200]!r}"
    return Status.OK, None, ""


def measure(exes: list[Path], tests: tuple[TestCase, ...], cfg: EvalConfig) -> dict[Path, float]:
    """Execution time over ``tests`` per executable, as in SuperCoder (arXiv:2505.11480).

    For each test input, hyperfine discards ``cfg.warmup_runs`` runs and averages the next
    ``cfg.timing_repeats``; a program's time is the sum of those means over all tests.
    """
    workdir = exes[0].parent
    totals = dict.fromkeys(exes, 0.0)
    # hyperfine has no per-run timeout, so bound the whole call instead.
    timeout_s = (cfg.warmup_runs + cfg.timing_repeats) * cfg.run_timeout_s * len(exes)
    for i, test in enumerate(tests):
        stdin, report = workdir / f"test{i}.in", workdir / f"test{i}.json"
        stdin.write_text(test.stdin)
        subprocess.run(
            [
                cfg.hyperfine, "--shell=none", "--style=none", "--output=null",
                f"--input={stdin}", f"--warmup={cfg.warmup_runs}",
                f"--runs={cfg.timing_repeats}", f"--export-json={report}",
                *map(str, exes),
            ],
            capture_output=True,
            timeout=timeout_s,
            cwd=workdir,
            check=True,  # it already passed these tests, so any failure here is a flaky binary
        )
        for exe, r in zip(exes, json.loads(report.read_text())["results"], strict=True):
            totals[exe] += r["mean"]
    return totals


def evaluate(problem: Problem, gen: Generation, cfg: EvalConfig) -> EvalResult:
    """Score ``gen`` on all of ``problem.tests``; anything but OK keeps speedup 1.0."""

    if gen.asm is None:
        statuses = [c.response_status for c in gen.llm_calls]
        return EvalResult(problem_id=problem.id, status=Status.NO_ASM,
                          detail=f"no ```assembly block; responses {statuses}")

    with tempfile.TemporaryDirectory(prefix=f"asmevo-{problem.id}-") as tmp:
        workdir = Path(tmp)
        base_exe, err = build(problem.baseline_asm, workdir, "baseline", cfg)
        if base_exe is None:
            raise RuntimeError(f"problem {problem.id}: -O3 baseline does not build:\n{err}")
        # Tests the reference cannot finish within the limit are dropped for both programs;
        # any other baseline failure means speedups would be meaningless on this toolchain.
        kept, dropped = [], []
        for i, test in enumerate(problem.tests):
            status, _, _ = check_tests(base_exe, (test,), cfg)
            if status is Status.TIMEOUT:
                dropped.append(i)
            elif status is not Status.OK:
                raise RuntimeError(f"problem {problem.id}: -O3 baseline {status} on test {i}")
            else:
                kept.append(i)
        if not kept:
            raise RuntimeError(f"problem {problem.id}: -O3 baseline times out on every test")
        tests = tuple(problem.tests[i] for i in kept)

        def result(status: Status, **kw) -> EvalResult:
            return EvalResult(problem_id=problem.id, status=status,
                              dropped_tests=tuple(dropped), **kw)

        cand_exe, err = build(gen.asm, workdir, "candidate", cfg)
        if cand_exe is None:
            return result(Status.BUILD_FAIL, detail=err)
        status, failed, detail = check_tests(cand_exe, tests, cfg)
        if status is not Status.OK:
            return result(status, failed_test=kept[failed], detail=detail)

        times = measure([base_exe, cand_exe], tests, cfg)
        base_s, cand_s = times[base_exe], times[cand_exe]
        return result(
            Status.OK,
            speedup=base_s / cand_s,
            baseline_mean_s=base_s,
            candidate_mean_s=cand_s,
        )
