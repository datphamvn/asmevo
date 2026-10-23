import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from asmevo.config import load_config
from asmevo.data import dataset
from asmevo.data.dataset import Problem
from asmevo.eval import runner
from asmevo.eval.runner import Timing, evaluate_asm, outputs_match
from asmevo.schemas import Status

CFG = replace(load_config().eval, run_timeout_s=2, warmup_runs=1, timing_repeats=4,
              pin_cpus=False)

needs_x86_64 = pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() in ("x86_64", "amd64")
         and all(shutil.which(t) for t in ("gcc", "hyperfine", "prlimit"))),
    reason="needs Linux x86-64 with gcc, hyperfine and prlimit: run in the docker image",
)

C_PROGRAM = r"""
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static long sum(long n) { long s = 0; for (long i = 1; i <= n; i++) s += i % 7; return s; }
int main(void) {
    long n;
    if (scanf("%ld", &n) != 1) return 1;
    /*BEFORE*/
    printf("%ld\n", sum(n) /*OFFSET*/);
    return 0 /*EXIT*/;
}
"""
LOOP = "for (volatile unsigned long x = 0;; x++) {}"
CLOSED_FORM = r'long q = n / 7, r = n % 7; printf("%ld\n", 21 * q + r * (r + 1) / 2); return 0;'
# Passes the 2 tests, then aborts on every later run, i.e. only while being timed.
FAILS_WHEN_TIMED = """
    int runs = 0; FILE *f = fopen("runs", "r");
    if (f) { if (fscanf(f, "%d", &runs) != 1) runs = 0; fclose(f); }
    f = fopen("runs", "w"); fprintf(f, "%d", runs + 1); fclose(f);
    if (runs >= 2) abort();
"""


def c_variant(before: str = "", offset: str = "", exit_code: str = "") -> str:
    return (C_PROGRAM.replace("/*BEFORE*/", before).replace("/*OFFSET*/", offset)
            .replace("/*EXIT*/", exit_code))


def case(n: int) -> dataset.TestCase:
    q, r = divmod(n, 7)  # each full cycle of i % 7 sums to 21
    return dataset.TestCase(f"{n}\n", f"{21 * q + r * (r + 1) // 2}\n")


TESTS = (case(10), case(100_000))


def make_problem(baseline_asm: str, tests: tuple[dataset.TestCase, ...] = TESTS) -> Problem:
    return Problem(id=1, c_code="", baseline_asm=baseline_asm, tests=tests,
                   baseline_binary=b"", standard_prompt="")


def running(name: str) -> bool:
    for comm in Path("/proc").glob("[0-9]*/comm"):
        try:
            if comm.read_text().strip() == name:
                return True
        except OSError:
            pass
    return False


@pytest.fixture(scope="module")
def to_asm(tmp_path_factory):
    """gcc -O3 -S of a C source, the way the dataset's baselines were made."""
    cache = {}

    def compile_c(src: str) -> str:
        if src not in cache:
            path = tmp_path_factory.mktemp("c") / "p.c"
            path.write_text(src)
            cache[src] = subprocess.run(["gcc", "-O3", "-S", "-o", "-", str(path)],
                                        check=True, capture_output=True, text=True).stdout
        return cache[src]

    return compile_c


@pytest.fixture(autouse=True)
def fresh_baselines():
    runner._baselines.clear()


def test_outputs_match():
    assert outputs_match("1 2  \n3\n\n\n", "1 2\n3")
    assert not outputs_match("1\n2\n", "1\n\n2\n")


def test_timing_stats():
    t = Timing(((1.0, 2.0, 9.0), (3.0, 3.0, 3.0)))  # 2 tests, 3 runs each
    assert t.median == 5.0  # 2 + 3: the 9.0 outlier run does not count
    assert t.mean == 7.0  # 4 + 3, as SuperCoder sums means
    assert t.stddev == pytest.approx(19 ** 0.5)  # passes 4, 5, 12
    assert Timing(((1.0,), (2.0,))).stddev == 0.0


def test_no_asm_needs_no_toolchain():
    r = evaluate_asm(make_problem("unused"), None, TESTS, CFG)
    assert (r.status, r.speedup) == (Status.NO_ASM, 1.0)


def test_rejects_other_hosts(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    runner._require_linux_x86_64.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="Linux x86-64"):
            evaluate_asm(make_problem("unused"), "ret", TESTS, CFG)
    finally:
        runner._require_linux_x86_64.cache_clear()


@needs_x86_64
def test_identical_candidate_is_ok_and_timed(to_asm):
    base = to_asm(c_variant())
    r = evaluate_asm(make_problem(base), base, TESTS, CFG)
    assert r.status is Status.OK, r.detail
    assert r.baseline_median_s > 0 and r.candidate_median_s > 0
    assert r.baseline_stddev_s >= 0 and r.candidate_stddev_s >= 0
    assert r.speedup == pytest.approx(r.baseline_median_s / r.candidate_median_s)


@needs_x86_64
@pytest.mark.parametrize("variant, status, detail", [
    (dict(offset="+ 1"), Status.WRONG_OUTPUT, "got"),
    (dict(exit_code="+ 3"), Status.WRONG_OUTPUT, "exit code 3, expected 0"),
    (dict(before="abort();"), Status.CRASH, "SIGABRT"),
    (dict(before=LOOP), Status.TIMEOUT, "exceeded"),
    (dict(before=FAILS_WHEN_TIMED), Status.CRASH, "while timed"),
])
def test_broken_candidate_fails_alone(to_asm, variant, status, detail):
    r = evaluate_asm(make_problem(to_asm(c_variant())), to_asm(c_variant(**variant)), TESTS,
                     CFG)
    assert r.status is status, r.detail
    assert detail in r.detail
    assert r.speedup == 1.0
    assert not running("candidate")


@needs_x86_64
def test_measure_times_each_test_timing_repeats_times(to_asm, tmp_path):
    exes = [runner.build(to_asm(c_variant()), tmp_path, name, CFG)[0]
            for name in ("baseline", "candidate")]
    for t in runner.measure(*exes, TESTS, [0, 0], CFG, tmp_path):
        assert [len(runs) for runs in t.samples] == [CFG.timing_repeats] * len(TESTS)
        assert t.median > 0


@needs_x86_64
def test_build_failure_keeps_toolchain_error(to_asm):
    r = evaluate_asm(make_problem(to_asm(c_variant())), "this is not assembly\n", TESTS, CFG)
    assert r.status is Status.BUILD_FAIL
    assert "Error" in r.detail


@needs_x86_64
def test_runaway_output_is_capped(to_asm, monkeypatch):
    monkeypatch.setattr(runner, "_OUTPUT_LIMIT", 1 << 20)
    spam = to_asm(c_variant(before='for (;;) puts("spam");'))
    r = evaluate_asm(make_problem(to_asm(c_variant())), spam, TESTS, CFG)
    assert r.status is Status.WRONG_OUTPUT
    assert "more than" in r.detail


@needs_x86_64
def test_forked_processes_do_not_outlive_the_run(to_asm):
    forks = to_asm(c_variant(before=f"if (fork() == 0) {LOOP}"))
    r = evaluate_asm(make_problem(to_asm(c_variant())), forks, TESTS, CFG)
    assert r.status is Status.OK, r.detail
    assert not running("candidate")


@needs_x86_64
def test_failed_test_and_dropped_tests_index_the_given_tests(to_asm):
    base = to_asm(c_variant(before=f"if (n == 0) {LOOP}"))  # the baseline hangs on n = 0
    tests = (dataset.TestCase("0\n", "0\n"), *TESTS)
    wrong_on_big = to_asm(c_variant(before=f"if (n == 0) {LOOP}", offset="+ (n > 1000)"))
    r = evaluate_asm(make_problem(base, tests), wrong_on_big, tests, CFG)
    assert (r.status, r.failed_test, r.dropped_tests) == (Status.WRONG_OUTPUT, 2, (0,))
    with pytest.raises(RuntimeError, match="times out on every test"):
        evaluate_asm(make_problem(base, tests), base, tests[:1], CFG)


@needs_x86_64
def test_broken_baseline_raises():
    with pytest.raises(RuntimeError, match="does not build"):
        evaluate_asm(make_problem("not assembly\n"), "ret\n", TESTS, CFG)


@needs_x86_64
def test_baseline_is_built_once_and_candidates_run_concurrently(to_asm, monkeypatch):
    builds = []
    real_build = runner.build

    def counting_build(asm, workdir, name, cfg):
        builds.append(name)
        return real_build(asm, workdir, name, cfg)

    monkeypatch.setattr(runner, "build", counting_build)
    base = to_asm(c_variant())
    problem = make_problem(base)
    cases = [
        (base, Status.OK),
        ("garbage\n", Status.BUILD_FAIL),
        (to_asm(c_variant(offset="+ 1")), Status.WRONG_OUTPUT),
        (to_asm(c_variant(before="abort();")), Status.CRASH),
        (to_asm(c_variant(before=LOOP)), Status.TIMEOUT),
    ]
    with ThreadPoolExecutor(max_workers=len(cases)) as pool:
        results = list(pool.map(lambda c: evaluate_asm(problem, c[0], TESTS, CFG), cases))
    assert [r.status for r in results] == [status for _, status in cases]
    assert builds.count("baseline") == 1


@needs_x86_64
def test_speedup_compares_the_candidate_with_the_baseline(to_asm):
    """Only programs of different speed show a swap of baseline and candidate times."""
    slow, fast = to_asm(c_variant()), to_asm(c_variant(before=CLOSED_FORM))
    heavy = (case(20_000_000),)
    faster = evaluate_asm(make_problem(slow, heavy), fast, heavy, CFG)
    slower = evaluate_asm(make_problem(fast, heavy), slow, heavy, CFG)
    assert faster.status is slower.status is Status.OK
    assert faster.speedup > 2 and slower.speedup < 0.5
    assert faster.baseline_median_s > faster.candidate_median_s


@needs_x86_64
def test_nonzero_exit_codes_of_the_baseline_are_kept_and_timed(to_asm):
    base = to_asm(c_variant(exit_code="+ (int)(n % 2)"))  # exits 1 on odd n
    tests = (case(10), case(7))
    same = evaluate_asm(make_problem(base, tests), base, tests, CFG)
    assert same.status is Status.OK, same.detail
    always_zero = evaluate_asm(make_problem(base, tests), to_asm(c_variant()), tests, CFG)
    assert (always_zero.status, always_zero.failed_test) == (Status.WRONG_OUTPUT, 1)
    assert "exit code 0, expected 1" in always_zero.detail
