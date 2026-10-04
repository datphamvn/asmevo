"""Isolated build -> test -> time pipeline for one candidate against the -O3 baseline.

Must run on Linux x86-64 (natively or in the docker image): the dataset assembly is x86-64.
"""

from __future__ import annotations

import functools
import json
import math
import os
import queue
import statistics
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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
        if proc.returncode != 0:  # hyperfine cannot time a program that exits non-zero
            return Status.CRASH, i, f"exit status {proc.returncode}: {proc.stderr[-500:]}"
    return Status.OK, None, ""


class TimingFailed(Exception):
    """A program that passed ``test`` failed on it while being timed, i.e. it is flaky."""

    def __init__(self, test: int, stderr: str) -> None:
        super().__init__(stderr)
        self.test = test


def _cpu_list(text: str) -> list[int]:
    cpus = []
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


@functools.cache
def cpu_layout() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(timing, work)`` CPUs: one hardware thread per performance core times programs, alone
    on its core; everything else runs on the remaining cores.

    On a hybrid CPU the efficiency cores do the work; otherwise half the cores are kept for
    timing. Read once, before ``pin_work_cpus`` narrows this process's affinity.
    """
    allowed = set(os.sched_getaffinity(0))
    sysfs = Path("/sys/devices/system/cpu")
    p_cores = Path("/sys/devices/cpu_core/cpus")
    fast = set(_cpu_list(p_cores.read_text())) & allowed if p_cores.exists() else allowed
    cores = sorted({tuple(_cpu_list((sysfs / f"cpu{c}/topology/thread_siblings_list")
                                    .read_text())) for c in fast})
    if not allowed - fast:
        cores = cores[: max(1, len(cores) // 2)]
    timing = tuple(sorted(core[0] for core in cores))
    work = tuple(sorted(allowed - {c for core in cores for c in core}))
    return timing, work or tuple(sorted(allowed - set(timing)))


def pin_work_cpus() -> None:
    """Keep this thread, the threads it starts, and their subprocesses off the timing CPUs."""
    os.sched_setaffinity(0, cpu_layout()[1])


@functools.cache
def _timing_slots() -> queue.Queue[int]:
    slots: queue.Queue[int] = queue.Queue()
    for cpu in cpu_layout()[0]:
        slots.put(cpu)
    return slots


@contextmanager
def _timing_cpu(cfg: EvalConfig) -> Iterator[list[str]]:
    """Command prefix that runs hyperfine alone on a free timing CPU, waiting for one."""
    if not cfg.pin_cpus:
        yield []
        return
    cpu = _timing_slots().get()
    try:
        yield ["taskset", "-c", str(cpu)]
    finally:
        _timing_slots().put(cpu)


def measure(exes: list[Path], tests: tuple[TestCase, ...], cfg: EvalConfig) -> dict[Path, float]:
    """Execution time over ``tests`` per executable, as in SuperCoder (arXiv:2505.11480).

    For each test input, hyperfine discards ``cfg.warmup_runs`` runs and then times
    ``cfg.timing_repeats`` runs; a program's time is the sum over tests of their
    ``cfg.timing_stat``. With ``cfg.balanced_order`` the programs run in ABBA blocks of half
    the runs each after one discarded block: the first block of every hyperfine call runs a few
    percent slow, so it must not count, and ABBA cancels any drift across the rest.
    """
    workdir = exes[0].parent
    totals = dict.fromkeys(exes, 0.0)
    order = [*exes, *reversed(exes)] if cfg.balanced_order else exes
    runs = math.ceil(cfg.timing_repeats * len(exes) / len(order))
    if cfg.balanced_order:
        order = [exes[0], *order]
    discard = 1 if cfg.balanced_order else 0
    stat = {"mean": statistics.fmean, "median": statistics.median}[cfg.timing_stat]
    # hyperfine has no per-run timeout, so bound the whole call instead.
    timeout_s = (cfg.warmup_runs + runs) * cfg.run_timeout_s * len(order)
    with _timing_cpu(cfg) as prefix:
        for i, test in enumerate(tests):
            stdin, report = workdir / f"test{i}.in", workdir / f"test{i}.json"
            stdin.write_text(test.stdin)
            try:
                subprocess.run(
                    [
                        *prefix, cfg.hyperfine, "--shell=none", "--style=none",
                        "--output=null", f"--input={stdin}", f"--warmup={cfg.warmup_runs}",
                        f"--runs={runs}", f"--export-json={report}", *map(str, order),
                    ],
                    capture_output=True,
                    timeout=timeout_s,
                    cwd=workdir,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise TimingFailed(i, e.stderr.decode(errors="replace")[-500:]) from e
            times: dict[Path, list[float]] = {exe: [] for exe in exes}
            results = json.loads(report.read_text())["results"]
            for exe, r in zip(order[discard:], results[discard:], strict=True):
                times[exe].extend(r["times"])
            for exe in exes:
                totals[exe] += stat(times[exe])
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

        try:
            times = measure([base_exe, cand_exe], tests, cfg)
        except TimingFailed as e:
            return result(Status.CRASH, failed_test=kept[e.test],
                          detail=f"failed while timing: {e}")
        base_s, cand_s = times[base_exe], times[cand_exe]
        return result(
            Status.OK,
            speedup=base_s / cand_s,
            baseline_mean_s=base_s,
            candidate_mean_s=cand_s,
        )
