"""Feedback prompts keep the SuperCoder root and state ABI constraints on repair."""

from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as Case
from asmevo.methods.feedback import build_child_prompt, build_feedback
from asmevo.schemas import EvalResult, Status


def _problem() -> Problem:
    return Problem(
        id=7,
        c_code="int main(void){return 0;}",
        baseline_asm="main:\nret",
        tests=(Case("1 2\n", "3\n"),),
        baseline_binary=b"",
        standard_prompt="SUPERCODER PROMPT",
    )


def test_ok_feedback_reports_speedup_and_abi():
    result = EvalResult(7, Status.OK, speedup=1.25, baseline_mean_s=2.0, candidate_mean_s=1.6)
    message = build_feedback(_problem(), result)
    assert message["role"] == "user"
    assert "1.250x" in message["content"]
    assert "System V AMD64 ABI" in message["content"]
    assert "```assembly" in message["content"]


def test_child_of_parent_appends_feedback_without_replacing_root():
    problem = _problem()
    crashed = EvalResult(7, Status.CRASH, detail="segfault", failed_test=0)
    parent = ("```assembly\nret\n```", crashed)
    messages = build_child_prompt(problem, parent, hint=None)
    assert messages[0]["content"] == "SUPERCODER PROMPT"
    assert messages[1]["role"] == "assistant"
    assert "segfault" in messages[2]["content"]

    hint = "Strength-reduce the inner loop."
    assert hint in build_child_prompt(problem, parent, hint)[-1]["content"]
