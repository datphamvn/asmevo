"""Isolated build -> test -> time pipeline for one candidate against the -O3 baseline.

Must run on Linux x86-64 (natively or in the docker image): the dataset assembly is x86-64.
"""

from __future__ import annotations

import math
import statistics
import subprocess
import tempfile
import time
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


def _run_all(exe: Path, tests: tuple[TestCase, ...], timeout_s: float) -> float:
    total = 0.0
    for test in tests:
        start = time.perf_counter()
        subprocess.run(
            [str(exe)],
            input=test.stdin.encode(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            cwd=exe.parent,
            check=True,  # it already passed these tests, so any failure here is a flaky binary
        )
        total += time.perf_counter() - start
    return total


def measure(
    exes: list[Path], tests: tuple[TestCase, ...], cfg: EvalConfig
) -> dict[Path, list[float]]:
    """Total wall time over ``tests`` per executable and repeat.

    Executables are interleaved within each repeat (order alternating) so that machine
    load, including other evaluation workers, affects all of them equally.
    """
    for exe in exes:
        for _ in range(cfg.warmup_runs):
            _run_all(exe, tests, cfg.run_timeout_s)
    times: dict[Path, list[float]] = {exe: [] for exe in exes}
    for r in range(cfg.timing_repeats):
        for exe in exes if r % 2 == 0 else reversed(exes):
            times[exe].append(_run_all(exe, tests, cfg.run_timeout_s))
    return times


def evaluate(problem: Problem, gen: Generation, cfg: EvalConfig) -> EvalResult:
    """Score ``gen`` on all of ``problem.tests``; anything but OK keeps speedup 1.0."""

    def result(status: Status, **kw) -> EvalResult:
        return EvalResult(problem_id=problem.id, status=status, **kw)

    if gen.asm is None:
        statuses = [c.response_status for c in gen.llm_calls]
        return result(Status.NO_ASM, detail=f"no ```assembly block; responses {statuses}")

    with tempfile.TemporaryDirectory(prefix=f"asmevo-{problem.id}-") as tmp:
        workdir = Path(tmp)
        base_exe, err = build(problem.baseline_asm, workdir, "baseline", cfg)
        if base_exe is None:
            raise RuntimeError(f"problem {problem.id}: -O3 baseline does not build:\n{err}")
        status, failed, detail = check_tests(base_exe, problem.tests, cfg)
        if status is not Status.OK:
            # Speedups are meaningless if the reference itself misbehaves on this toolchain.
            raise RuntimeError(f"problem {problem.id}: -O3 baseline {status} on test {failed}")

        cand_exe, err = build(gen.asm, workdir, "candidate", cfg)
        if cand_exe is None:
            return result(Status.BUILD_FAIL, detail=err)
        status, failed, detail = check_tests(cand_exe, problem.tests, cfg)
        if status is not Status.OK:
            return result(status, failed_test=failed, detail=detail)

        times = measure([base_exe, cand_exe], problem.tests, cfg)
        base_s = statistics.median(times[base_exe])
        cand_s = statistics.median(times[cand_exe])
        return result(
            Status.OK,
            speedup=base_s / cand_s,
            baseline_median_s=base_s,
            candidate_median_s=cand_s,
        )
