"""Check the dataset's -O3 baseline against our own ``gcc -O3 -S`` of the same C code.

The dataset ships ``baseline_asm`` built by its authors' gcc; every speedup is measured
against it. This rebuilds the baseline with ``cfg.cc`` and compares the two programs on the
same tests and timing protocol, so a toolchain mismatch can be reported as a threat to
validity. Linux x86-64 only, like ``runner``.
"""

from __future__ import annotations

import difflib
import math
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asmevo.config import EvalConfig
from asmevo.data.dataset import Problem
from asmevo.eval.runner import build, check_tests, measure
from asmevo.schemas import Status

_IDENT_RE = re.compile(r'^\s*\.ident\s+"(.*)"\s*$', re.MULTILINE)
# Lines that name the source file or compiler but never change the generated code.
_NOISE_RE = re.compile(r"^\.(file|ident)\b")


@dataclass(frozen=True)
class BaselineComparison:
    problem_id: int
    dataset_ident: str | None  # compiler recorded in the dataset's ``.ident``
    ours_ident: str | None
    compile_ok: bool
    identical: bool  # same code after ``normalize_asm``
    n_lines_dataset: int
    n_lines_ours: int
    diff_lines: int  # lines added plus removed between the normalized listings
    dataset_status: Status
    ours_status: Status | None  # None when our compile failed
    dataset_s: float | None = None
    ours_s: float | None = None
    ratio: float | None = None  # ours_s / dataset_s; > 1 means our baseline is slower
    dropped_tests: tuple[int, ...] = ()  # the dataset baseline exceeds run_timeout_s on these
    detail: str = ""


def gcc_version(cc: str) -> str:
    """First line of ``cc --version``."""
    out = subprocess.run([cc, "--version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


def asm_ident(asm: str) -> str | None:
    """The compiler that produced ``asm``, from its ``.ident`` directive."""
    m = _IDENT_RE.search(asm)
    return m.group(1) if m else None


def normalize_asm(asm: str) -> list[str]:
    """Non-empty lines without ``#`` comments, ``.file`` or ``.ident``, whitespace collapsed.

    Lines with a string literal keep their ``#``, which may be part of the string.
    """
    lines = []
    for line in asm.splitlines():
        if '"' not in line:
            line = line.split("#", 1)[0]
        line = " ".join(line.split())
        if line and not _NOISE_RE.match(line):
            lines.append(line)
    return lines


def diff_size(a: list[str], b: list[str]) -> int:
    return sum(1 for d in difflib.ndiff(a, b) if d.startswith(("+ ", "- ")))


def compile_c(c_code: str, workdir: Path, cfg: EvalConfig) -> tuple[str | None, str]:
    """``cfg.cc -O3 -S`` on ``c_code``; return ``(asm, "")`` or ``(None, compiler stderr)``."""
    src, out = workdir / "temp.c", workdir / "temp.s"  # same name as the dataset's .file
    src.write_text(c_code)
    try:
        proc = subprocess.run(
            [cfg.cc, "-O3", "-S", src.name, "-o", out.name],
            capture_output=True,
            text=True,
            timeout=cfg.compile_timeout_s,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return None, "compile timed out"
    if proc.returncode != 0:
        return None, proc.stderr[-2000:]
    return out.read_text(), ""


def compare(problem: Problem, cfg: EvalConfig) -> BaselineComparison:
    """Build both baselines, run ``problem.tests`` on each, and time them interleaved.

    Tests the dataset baseline cannot finish within the limit are dropped, as in
    ``runner.evaluate``. Raises only if the dataset baseline itself is broken.
    """
    dataset_lines = normalize_asm(problem.baseline_asm)
    with tempfile.TemporaryDirectory(prefix=f"asmevo-gcc-{problem.id}-") as tmp:
        workdir = Path(tmp)
        data_exe, err = build(problem.baseline_asm, workdir, "dataset", cfg)
        if data_exe is None:
            raise RuntimeError(f"problem {problem.id}: dataset baseline does not build:\n{err}")
        kept, dropped = [], []
        for i, test in enumerate(problem.tests):
            status, _, _ = check_tests(data_exe, (test,), cfg)
            if status is Status.TIMEOUT:
                dropped.append(i)
            elif status is not Status.OK:
                raise RuntimeError(f"problem {problem.id}: dataset baseline {status} on test {i}")
            else:
                kept.append(i)
        if not kept:
            raise RuntimeError(f"problem {problem.id}: dataset baseline times out on every test")
        tests = tuple(problem.tests[i] for i in kept)

        ours_asm, err = compile_c(problem.c_code, workdir, cfg)
        ours_lines = normalize_asm(ours_asm or "")

        def result(ours_status: Status | None, **kw) -> BaselineComparison:
            return BaselineComparison(
                problem_id=problem.id,
                dataset_ident=asm_ident(problem.baseline_asm),
                ours_ident=asm_ident(ours_asm or ""),
                compile_ok=ours_asm is not None,
                identical=ours_asm is not None and ours_lines == dataset_lines,
                n_lines_dataset=len(dataset_lines),
                n_lines_ours=len(ours_lines),
                diff_lines=diff_size(dataset_lines, ours_lines) if ours_asm else 0,
                dataset_status=Status.OK,
                ours_status=ours_status,
                dropped_tests=tuple(dropped),
                **kw,
            )

        if ours_asm is None:
            return result(None, detail=err)
        ours_exe, err = build(ours_asm, workdir, "ours", cfg)
        if ours_exe is None:
            return result(Status.BUILD_FAIL, detail=err)
        status, failed, detail = check_tests(ours_exe, tests, cfg)
        if status is not Status.OK:
            return result(status, detail=f"test {kept[failed]}: {detail}")

        times = measure([data_exe, ours_exe], tests, cfg)
        data_s, ours_s = times[data_exe], times[ours_exe]
        return result(Status.OK, dataset_s=data_s, ours_s=ours_s, ratio=ours_s / data_s)


def summarize(rows: list[BaselineComparison], cc_version: str) -> dict[str, Any]:
    """Counts plus the geometric mean of ``ratio`` over problems where both pass."""
    ratios = [r.ratio for r in rows if r.ratio is not None]
    return {
        "problems": len(rows),
        "cc": cc_version,
        "dataset_idents": sorted({r.dataset_ident or "?" for r in rows}),
        "compile_ok": sum(r.compile_ok for r in rows),
        "identical": sum(r.identical for r in rows),
        "ours_status_counts": dict(Counter(r.ours_status or "compile_fail" for r in rows)),
        "timed": len(ratios),
        "geomean_ratio": math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None,
        "min_ratio": min(ratios, default=None),
        "max_ratio": max(ratios, default=None),
    }
