"""Run plots read results.jsonl and do not need a compiler."""

import json
from pathlib import Path

from asmevo.eval.plots import write_run_plots
from asmevo.schemas import EvalResult, Generation, LLMCall, Status


def _write_run(tmp_path: Path) -> None:
    results = [
        EvalResult(1, Status.OK, speedup=1.2, baseline_mean_s=1.0, candidate_mean_s=0.8),
        EvalResult(2, Status.BUILD_FAIL, detail="asm error"),
    ]
    generations = [
        Generation(1, "policy_tree", "ret", "```assembly\nret\n```", (
            LLMCall("m", "policy_expand", "completed", 10, 20, 0, 0.4),
        )),
        Generation(2, "policy_tree", None, "no code", (
            LLMCall("m", "policy_expand", "completed", 10, 5, 0, 0.2),
        )),
    ]
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(_result(r)) + "\n" for r in results))
    (tmp_path / "generations.jsonl").write_text(
        "".join(json.dumps(_generation(g)) + "\n" for g in generations))


def _result(r: EvalResult) -> dict:
    return {
        "problem_id": r.problem_id,
        "status": r.status.value,
        "speedup": r.speedup,
        "baseline_mean_s": r.baseline_mean_s,
        "candidate_mean_s": r.candidate_mean_s,
        "failed_test": r.failed_test,
        "dropped_tests": list(r.dropped_tests),
        "detail": r.detail,
    }


def _generation(g: Generation) -> dict:
    return {
        "problem_id": g.problem_id,
        "method": g.method,
        "asm": g.asm,
        "reply": g.reply,
        "llm_calls": [c.__dict__ for c in g.llm_calls],
        "rounds": [],
        "round_asm": [],
        "round_parent": [],
    }


def test_write_run_plots(tmp_path: Path):
    _write_run(tmp_path)
    text, written = write_run_plots(tmp_path, tmp_path / "plots")
    names = {path.name for path in written}
    assert names == {"summary.txt", "status.png", "speedups.png", "tokens.png"}
    assert "geomean speedup:" in text
    assert "improved rate:" in text
    assert "output tokens:     25" in text
    for path in written:
        assert path.stat().st_size > 0
