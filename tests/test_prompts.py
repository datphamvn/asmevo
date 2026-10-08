"""Feedback prompts keep the SuperCoder root and state ABI constraints on repair."""

from asmevo.data.dataset import Problem, TestCase
from asmevo.methods.policy_tree.prompt import (
    BRANCH_PROMPT_ID,
    EXPAND_PROMPT_ID,
    build_child_prompt,
    build_feedback,
)
from asmevo.schemas import EvalResult, Status


def _problem() -> Problem:
    return Problem(
        id=7,
        c_code="int main(void){return 0;}",
        baseline_asm="main:\nret",
        tests=(TestCase("1 2\n", "3\n"),),
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


def test_build_fail_includes_compiler_log():
    result = EvalResult(7, Status.BUILD_FAIL, detail="undefined reference to foo")
    content = build_feedback(_problem(), result)["content"]
    assert "undefined reference to foo" in content
    assert "callee-saved" in content


def test_wrong_output_clips_the_failed_test():
    result = EvalResult(7, Status.WRONG_OUTPUT, detail="got 0", failed_test=0)
    content = build_feedback(_problem(), result)["content"]
    assert "got 0" in content
    assert "stdin:" in content
    assert "1 2" in content


def test_child_of_parent_appends_feedback_without_replacing_root():
    problem = _problem()
    parent = ("```assembly\nret\n```", EvalResult(7, Status.CRASH, detail="segfault"))
    messages, prompt_id = build_child_prompt(problem, parent, hint=None)
    assert prompt_id == EXPAND_PROMPT_ID
    assert messages[0]["content"] == "SUPERCODER PROMPT"
    assert messages[1]["role"] == "assistant"
    assert "segfault" in messages[2]["content"]

    _, hinted = build_child_prompt(problem, parent, hint="Strength-reduce the inner loop.")
    assert hinted == BRANCH_PROMPT_ID
