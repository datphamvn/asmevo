"""Isolated build -> test -> time pipeline for one candidate against the -O3 baseline.

Must run on Linux x86-64 (natively or in the docker image): the dataset assembly is x86-64.

Every candidate gets its own temp directory. Each program run is held to ``prlimit`` limits,
gets a fixed minimal environment, and runs in its own process group, which is killed
afterwards, so whatever a broken, crashing or looping candidate forks dies with it. Its
output goes to size-capped files rather than pipes, so a runaway printer cannot exhaust the
caller's memory. The -O3 baseline of a problem is built once and its result on each test is
cached for the life of the process. With ``eval.pin_cpus``, timing runs alone on one hardware
thread per performance core while all other work stays on the remaining cores.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import math
import os
import platform
import queue
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from asmevo.config import EvalConfig
from asmevo.data.dataset import Problem, TestCase
from asmevo.schemas import EvalResult, Generation, Status

# Same for the baseline and the candidate: the environment's size shifts the initial stack,
# which can change timings, and the API keys in the caller's environment stay out of reach.
_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
_OUTPUT_LIMIT = 64 << 20  # bytes a run may write to a file (its stdout included)


@dataclass(frozen=True)
class _Exit:
    code: int | None  # None: killed after the timeout; negative: killed by that signal
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Timing:
    """Seconds per timed run: ``samples[i][j]`` is run ``j`` on test ``i``."""

    samples: tuple[tuple[float, ...], ...]

    @property
    def median(self) -> float:
        """One pass over the tests: the sum of each test's median run, so a slow outlier
        run does not count."""
        return sum(statistics.median(runs) for runs in self.samples)

    @property
    def mean(self) -> float:
        """The sum of each test's mean run, SuperCoder's (arXiv:2505.11480) measure."""
        return sum(statistics.fmean(runs) for runs in self.samples)

    @property
    def stddev(self) -> float:
        """Spread of one pass over the tests, taking run ``j`` of every test as pass ``j``."""
        passes = [sum(runs) for runs in zip(*self.samples, strict=True)]
        return statistics.stdev(passes) if len(passes) > 1 else 0.0


class MeasureFailed(Exception):
    """The candidate passed its tests but misbehaved while being timed, i.e. it is flaky."""

    def __init__(self, status: Status, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@functools.cache
def _which(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise FileNotFoundError(f"{tool} not found; evaluate in the docker image")
    return path


@functools.cache
def _require_linux_x86_64() -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("evaluating needs Linux x86-64 (natively or in the docker image); "
                           f"this host is {platform.system()} {platform.machine()}")


@functools.cache
def _cache_root() -> Path:
    root = Path(tempfile.mkdtemp(prefix="asmevo-baselines-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def _limits(cfg: EvalConfig) -> list[str]:
    """``prlimit`` prefix that every program run, timed runs included, is held to."""
    cpu_s = math.ceil(cfg.run_timeout_s) + 1
    return [_which("prlimit"), f"--as={cfg.memory_limit_mb << 20}", f"--cpu={cpu_s}",
            f"--fsize={_OUTPUT_LIMIT}", "--core=0", "--"]


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
    if not timing or not work:
        raise SystemExit(f"cannot split CPUs {sorted(allowed)} into timing and work cores; "
                         "set eval.pin_cpus: false")
    return timing, work


def pin_work_cpus() -> None:
    """Keep this thread, the threads it starts, and their subprocesses off the timing CPUs."""
    os.sched_setaffinity(0, cpu_layout()[1])


@functools.cache
def _timing_slots() -> queue.Queue[int]:
    slots: queue.Queue[int] = queue.Queue()
    for cpu in cpu_layout()[0]:
        slots.put(cpu)
    return slots


@contextlib.contextmanager
def _timing_cpu(cfg: EvalConfig) -> Iterator[list[str]]:
    """Command prefix that runs hyperfine alone on a free timing CPU, waiting for one."""
    if not cfg.pin_cpus:
        yield []
        return
    cpu = _timing_slots().get()
    try:
        yield [_which("taskset"), "-c", str(cpu)]
    finally:
        _timing_slots().put(cpu)


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def _read(path: Path) -> str:
    with path.open("rb") as f:
        return f.read(_OUTPUT_LIMIT).decode(errors="replace")


def _run(cmd: list[str], *, timeout: float, cwd: Path, stdin: Path | None = None,
         env: dict[str, str] | None = _ENV) -> _Exit:
    """Run ``cmd`` in a new process group with its output in files under ``cwd``, then kill
    the group, so nothing it started outlives the call."""
    out_path, err_path = cwd / ".stdout", cwd / ".stderr"
    with contextlib.ExitStack() as stack:
        fin = stack.enter_context(stdin.open("rb")) if stdin else subprocess.DEVNULL
        fout = stack.enter_context(out_path.open("wb"))
        ferr = stack.enter_context(err_path.open("wb"))
        proc = subprocess.Popen(cmd, stdin=fin, stdout=fout, stderr=ferr, cwd=cwd, env=env,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = None
        finally:  # also on KeyboardInterrupt, so a stopped run leaves nothing behind
            _kill_group(proc.pid)
            proc.wait()
    return _Exit(code, _read(out_path), _read(err_path))


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def build(asm: str, workdir: Path, name: str, cfg: EvalConfig) -> tuple[Path | None, str]:
    """Assemble and link ``asm``; return ``(exe, "")`` or ``(None, toolchain stderr)``."""
    src, exe = workdir / f"{name}.s", workdir / name
    src.write_text(asm.rstrip("\n") + "\n")
    r = _run([_which(cfg.cc), str(src), "-o", str(exe), *cfg.link_flags],
             timeout=cfg.compile_timeout_s, cwd=workdir, env=None)
    if r.code is None:
        return None, "assemble/link timed out"
    if r.code != 0:
        return None, r.stderr[-2000:]
    return exe, ""


def outputs_match(actual: str, expected: str) -> bool:
    """Compare line by line, ignoring trailing whitespace and trailing blank lines."""

    def norm(s: str) -> list[str]:
        return [line.rstrip() for line in s.rstrip().splitlines()]

    return norm(actual) == norm(expected)


def run_test(exe: Path, test: TestCase, cfg: EvalConfig) -> tuple[Status, int | None, str]:
    """Run ``exe`` on one test under the limits: ``(status, exit code, detail)``.

    ``OK`` means the output matched; comparing the exit code is up to the caller.
    """
    stdin = exe.parent / ".stdin"
    stdin.write_bytes(test.stdin.encode())
    r = _run([*_limits(cfg), str(exe)], timeout=cfg.run_timeout_s, cwd=exe.parent, stdin=stdin)
    if r.code is None or r.code == -signal.SIGXCPU:
        return Status.TIMEOUT, None, f"exceeded {cfg.run_timeout_s}s"
    if r.code == -signal.SIGXFSZ:
        return Status.WRONG_OUTPUT, None, f"wrote more than {_OUTPUT_LIMIT >> 20} MB"
    if r.code < 0:
        return Status.CRASH, None, f"killed by {_signal_name(-r.code)}: {r.stderr[-500:]}"
    if not outputs_match(r.stdout, test.expected_stdout):
        return Status.WRONG_OUTPUT, r.code, f"got {r.stdout[:200]!r}"
    return Status.OK, r.code, ""


def check_tests(
    exe: Path, tests: tuple[TestCase, ...], exit_codes: list[int], cfg: EvalConfig
) -> tuple[Status, int | None, str]:
    """Run every test, stopping at the first failure. Besides the expected output, the
    candidate must exit with the baseline's code (``exit_codes``): that is behavior too."""
    for i, (test, expected) in enumerate(zip(tests, exit_codes, strict=True)):
        status, code, detail = run_test(exe, test, cfg)
        if status is Status.OK and code != expected:
            status, detail = Status.WRONG_OUTPUT, f"exit code {code}, expected {expected}"
        if status is not Status.OK:
            return status, i, detail
    return Status.OK, None, ""


class _Baseline:
    """A problem's -O3 baseline, built on first use. Its exit code per test is cached as
    tests are first asked for; None means it exceeds the time limit, so the test is dropped."""

    def __init__(self, problem: Problem, cfg: EvalConfig) -> None:
        self.problem, self.cfg = problem, cfg
        self.exe: Path | None = None
        self.build_error: str | None = None
        self.exit_codes: dict[TestCase, int | None] = {}
        self.lock = threading.Lock()

    def check(self, tests: tuple[TestCase, ...]) -> list[int | None]:
        """Exit code per test; raise if the baseline does not build or fails a test, since
        speedups against it would be meaningless on this toolchain."""
        pid = self.problem.id
        with self.lock:
            if self.exe is None and self.build_error is None:
                workdir = Path(tempfile.mkdtemp(prefix=f"{pid}-", dir=_cache_root()))
                self.exe, err = build(self.problem.baseline_asm, workdir, "baseline", self.cfg)
                self.build_error = None if self.exe else err
            if self.exe is None:
                raise RuntimeError(f"problem {pid}: -O3 baseline does not build:\n"
                                   f"{self.build_error}")
            for test in tests:
                if test not in self.exit_codes:
                    status, code, detail = run_test(self.exe, test, self.cfg)
                    if status not in (Status.OK, Status.TIMEOUT):
                        raise RuntimeError(f"problem {pid}: -O3 baseline {status} on a test: "
                                           f"{detail}")
                    self.exit_codes[test] = code
            return [self.exit_codes[t] for t in tests]


_baselines: dict[tuple[int, str, EvalConfig], _Baseline] = {}
_baselines_lock = threading.Lock()


def _baseline(problem: Problem, cfg: EvalConfig) -> _Baseline:
    key = (problem.id, problem.baseline_asm, cfg)
    with _baselines_lock:
        if key not in _baselines:
            _baselines[key] = _Baseline(problem, cfg)
        return _baselines[key]


def _hyperfine(order: list[Path], stdin: Path, runs: int, cfg: EvalConfig, workdir: Path,
               prefix: list[str]) -> list[dict]:
    """hyperfine's result per entry of ``order`` (``times``, ``exit_codes``; a None code
    means a signal killed that run): ``cfg.warmup_runs`` discarded runs, then ``runs`` timed
    runs, of each entry on ``stdin`` in that order; an executable may appear more than once.
    hyperfine itself runs under ``prlimit``, so the timed programs inherit the limits without
    a wrapper being timed, and after ``prefix`` (``_timing_cpu``)."""
    report = workdir / "hyperfine.json"
    cmd = [*prefix, *_limits(cfg), _which(cfg.hyperfine), "--shell=none", "--style=none",
           "--output=null", "--ignore-failure", f"--input={stdin}",
           f"--warmup={cfg.warmup_runs}", f"--runs={runs}", f"--export-json={report}",
           *map(str, order)]
    timeout = (cfg.warmup_runs + runs) * len(order) * cfg.run_timeout_s + 10
    r = _run(cmd, timeout=timeout, cwd=workdir)
    if r.code is None:
        raise MeasureFailed(Status.TIMEOUT, f"a timed run exceeded {cfg.run_timeout_s}s")
    if r.code != 0:
        raise RuntimeError(f"hyperfine failed with exit code {r.code}: {r.stderr[-500:]}")
    results = json.loads(report.read_text())["results"]
    if len(results) != len(order):
        raise RuntimeError(f"hyperfine reported {len(results)} results for {len(order)} commands")
    return results


def measure(base_exe: Path, cand_exe: Path, tests: tuple[TestCase, ...],
            exit_codes: list[int], cfg: EvalConfig, workdir: Path) -> tuple[Timing, Timing]:
    """Time both executables on every test, interleaved so that drift (clock frequency,
    heat, other load) hits both alike.

    Per test, one hyperfine call times each program ``cfg.timing_repeats`` times after
    ``cfg.warmup_runs`` discarded runs. With ``cfg.balanced_order`` the programs run in ABBA
    blocks of half the runs each (rounded up) after one discarded baseline block: the first
    block of every hyperfine call runs a few percent slow, so it must not count, and ABBA
    cancels any drift across the rest. With ``cfg.pin_cpus`` the calls run alone on a free
    timing CPU, waiting for one. Raises ``MeasureFailed`` if the candidate exits otherwise
    than in its tests, or hangs.
    """
    if cfg.balanced_order:
        order = [base_exe, base_exe, cand_exe, cand_exe, base_exe]
        runs, discard = math.ceil(cfg.timing_repeats / 2), 1
    else:
        order, runs, discard = [base_exe, cand_exe], cfg.timing_repeats, 0
    samples: dict[Path, list[tuple[float, ...]]] = {base_exe: [], cand_exe: []}
    with _timing_cpu(cfg) as prefix:
        for i, (test, expected) in enumerate(zip(tests, exit_codes, strict=True)):
            stdin = workdir / f"test{i}.in"
            stdin.write_bytes(test.stdin.encode())
            times: dict[Path, list[float]] = {base_exe: [], cand_exe: []}
            results = _hyperfine(order, stdin, runs, cfg, workdir, prefix)
            for j, (exe, res) in enumerate(zip(order, results, strict=True)):
                if any(code != expected for code in res["exit_codes"]):
                    detail = (f"exit codes {res['exit_codes']} while timed on test {i}, "
                              f"{expected} when tested")
                    if exe == base_exe:
                        raise RuntimeError(f"-O3 baseline is unstable: {detail}")
                    raise MeasureFailed(Status.CRASH, detail)
                if j >= discard:
                    times[exe].extend(res["times"])
            for exe, runs_s in times.items():
                samples[exe].append(tuple(runs_s))
    return Timing(tuple(samples[base_exe])), Timing(tuple(samples[cand_exe]))


def evaluate_asm(
    problem: Problem, asm: str | None, tests: tuple[TestCase, ...], cfg: EvalConfig
) -> EvalResult:
    """Build, test and time ``asm`` against ``problem.baseline_asm`` on ``tests`` only.

    Shared by every search method. A broken candidate never raises: ``asm=None`` gives
    ``NO_ASM``, and build errors, wrong output or exit code, crashes and timeouts give their
    ``Status`` with speedup 1.0. Only a broken -O3 baseline raises, as in ``evaluate``.
    ``failed_test`` and ``dropped_tests`` are positions in ``tests``. The speedup is the
    ratio of ``cfg.timing_stat`` times (``Timing.median`` or ``Timing.mean``).
    """
    if asm is None:
        return EvalResult(problem_id=problem.id, status=Status.NO_ASM,
                          detail="no ```assembly block")
    _require_linux_x86_64()
    if not tests:
        raise ValueError(f"problem {problem.id}: no tests to evaluate on")
    baseline = _baseline(problem, cfg)
    codes = baseline.check(tests)
    kept = [i for i, code in enumerate(codes) if code is not None]
    dropped = tuple(i for i, code in enumerate(codes) if code is None)
    if not kept:
        raise RuntimeError(f"problem {problem.id}: -O3 baseline times out on every test given")
    kept_tests = tuple(tests[i] for i in kept)
    kept_codes = [codes[i] for i in kept]

    def result(status: Status, **kw) -> EvalResult:
        return EvalResult(problem_id=problem.id, status=status, dropped_tests=dropped, **kw)

    with tempfile.TemporaryDirectory(prefix=f"asmevo-{problem.id}-",
                                     ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        cand_exe, err = build(asm, workdir, "candidate", cfg)
        if cand_exe is None:
            return result(Status.BUILD_FAIL, detail=err)
        status, failed, detail = check_tests(cand_exe, kept_tests, kept_codes, cfg)
        if status is not Status.OK:
            return result(status, failed_test=kept[failed], detail=detail)
        try:
            base, cand = measure(baseline.exe, cand_exe, kept_tests, kept_codes, cfg, workdir)
        except MeasureFailed as e:
            return result(e.status, detail=e.detail)
    return result(
        Status.OK,
        speedup=getattr(base, cfg.timing_stat) / getattr(cand, cfg.timing_stat),
        baseline_mean_s=base.mean,
        candidate_mean_s=cand.mean,
        baseline_median_s=base.median,
        candidate_median_s=cand.median,
        baseline_stddev_s=base.stddev,
        candidate_stddev_s=cand.stddev,
    )


def evaluate(problem: Problem, gen: Generation, cfg: EvalConfig) -> EvalResult:
    """Score ``gen`` on all of ``problem.tests``; anything but OK keeps speedup 1.0."""
    if gen.asm is None:
        statuses = [c.response_status for c in gen.llm_calls]
        return EvalResult(problem_id=problem.id, status=Status.NO_ASM,
                          detail=f"no ```assembly block; responses {statuses}")
    return evaluate_asm(problem, gen.asm, problem.tests, cfg)
