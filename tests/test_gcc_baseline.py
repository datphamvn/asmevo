import platform
import shutil
import subprocess

import pytest

from asmevo.config import EvalConfig
from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as IOCase  # not "Test*", so pytest skips it
from asmevo.eval.gcc_baseline import (
    BaselineComparison,
    asm_ident,
    compare,
    compile_c,
    diff_size,
    normalize_asm,
    summarize,
)
from asmevo.schemas import Status

ASM = """\t.file\t"temp.c"
\t.text
# a comment line
\t.globl\tmain   # trailing comment
.LC0:
\t.string\t"#%d"
main:
\tret
\t.ident\t"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
"""


def test_asm_ident():
    assert asm_ident(ASM) == "GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
    assert asm_ident("main:\n\tret\n") is None


def test_normalize_asm_drops_noise_and_keeps_strings():
    assert normalize_asm(ASM) == [".text", ".globl main", ".LC0:", '.string "#%d"', "main:", "ret"]


def test_normalize_asm_ignores_compiler_and_file_name():
    other = ASM.replace("11.4.0", "12.3.0").replace("temp.c", "x.c").replace("\t", "  ")
    assert normalize_asm(other) == normalize_asm(ASM)


def test_diff_size():
    a = ["x", "y", "z"]
    assert diff_size(a, a) == 0
    assert diff_size(a, ["x", "w", "z"]) == 2  # one removed, one added
    assert diff_size(a, ["x"]) == 2


def test_summarize_geomean():
    def row(pid, ratio, status=Status.OK):
        return BaselineComparison(
            problem_id=pid, dataset_ident="d", ours_ident="o", compile_ok=status is not None,
            identical=False, n_lines_dataset=1, n_lines_ours=1, diff_lines=0,
            dataset_status=Status.OK, ours_status=status, ratio=ratio,
        )

    s = summarize([row(0, 0.5), row(1, 2.0), row(2, None, None)], "gcc x")
    assert s["timed"] == 2
    assert s["geomean_ratio"] == pytest.approx(1.0)
    assert s["ours_status_counts"] == {Status.OK: 2, "compile_fail": 1}


CC = shutil.which("gcc") or shutil.which("gcc-11") or shutil.which("gcc-12")
needs_toolchain = pytest.mark.skipif(
    CC is None or shutil.which("hyperfine") is None or platform.machine() != "x86_64",
    reason="needs gcc, hyperfine and Linux x86-64",
)

C_CODE = """#include <stdio.h>
int main(void) {
    long n, s = 0;
    scanf("%ld", &n);
    for (long i = 1; i <= n; i++) s += i;
    printf("%ld\\n", s);
    return 0;
}
"""


def eval_cfg() -> EvalConfig:
    return EvalConfig(cc=CC or "gcc", link_flags=("-lm",), compile_timeout_s=30,
                      run_timeout_s=5, memory_limit_mb=512, hyperfine="hyperfine",
                      warmup_runs=1, timing_repeats=2, workers=1)


def problem(c_code: str = C_CODE, *, baseline_asm: str | None = None) -> Problem:
    if baseline_asm is None:
        baseline_asm = subprocess.run([CC, "-O3", "-S", "-x", "c", "-", "-o", "-"],
                                      input=C_CODE, capture_output=True, text=True,
                                      check=True).stdout
    tests = (IOCase("10\n", "55\n"), IOCase("100\n", "5050\n"))
    return Problem(id=7, c_code=c_code, baseline_asm=baseline_asm, tests=tests,
                   baseline_binary=b"", standard_prompt="")


@needs_toolchain
def test_compile_c(tmp_path):
    asm, err = compile_c(C_CODE, tmp_path, eval_cfg())
    assert err == "" and asm is not None and "main:" in asm
    asm, err = compile_c("int main( {", tmp_path, eval_cfg())
    assert asm is None and "error" in err


@needs_toolchain
def test_compare_same_compiler_is_identical_and_timed():
    r = compare(problem(), eval_cfg())
    assert r.compile_ok and r.identical and r.diff_lines == 0
    assert r.ours_status is Status.OK
    assert r.ratio is not None and r.ratio > 0


@needs_toolchain
def test_compare_reports_wrong_output_without_raising():
    wrong = C_CODE.replace("s += i", "s += 2 * i")
    r = compare(problem(wrong), eval_cfg())
    assert r.compile_ok and not r.identical and r.diff_lines > 0
    assert r.ours_status is Status.WRONG_OUTPUT and r.ratio is None


@needs_toolchain
def test_compare_reports_compile_failure():
    r = compare(problem("int main( {"), eval_cfg())
    assert not r.compile_ok and r.ours_status is None and "error" in r.detail
