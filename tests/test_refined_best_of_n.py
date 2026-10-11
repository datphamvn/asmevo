"""refined_best_of_n with a scripted fake LLM and a fake evaluator (no API key, any
host), plus one end-to-end run through the real build/test/time pipeline (Linux x86-64 only)."""

from __future__ import annotations

import copy
import json
import math
import platform
import shutil
import subprocess
import warnings
from dataclasses import asdict, replace

import pytest
import yaml

import asmevo.main as cli
import asmevo.methods.nodes as shared_nodes
from asmevo.config import DEFAULT_CONFIG, RefinedBestOfNConfig, load_config
from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as Case
from asmevo.eval import curve
from asmevo.methods.nodes import best_round
from asmevo.methods.policy_tree.prompt import MAX_IO_CHARS
from asmevo.methods.refined_best_of_n import flow as refined_best_of_n
from asmevo.methods.refined_best_of_n.nodes import BASELINE, next_incumbent, normalize_asm
from asmevo.methods.refined_best_of_n.prompt import REFINE_PROMPT_ID, RETRY_PROMPT_ID
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import EvalResult, Generation, LLMCall, Status

PROBLEM = Problem(
    id=7,
    c_code="int main(void) { return 0; }",
    baseline_asm="main:\n\txorl\t%eax, %eax\n\tret",
    tests=(Case("1 2\n", "3\n"), Case("5 5\n", "10\n")),
    baseline_binary=b"",
    standard_prompt="SUPERCODER PROMPT\nOptimized Assembly Code:\n",
)
BASELINE_COPY = "main:   \n\n\txorl\t%eax, %eax\n\tret  "  # -O3 up to layout

# Four samples per round; the comments give the incumbent after each round.
SCRIPT = [
    "BUILD_FAIL r0", "WRONG r0", None, "CRASH r0",  # nothing correct: -O3
    "p1 speed=1.003", BASELINE_COPY, "p2 speed=0.97", "TIMEOUT r1",  # within min_gain: -O3
    "p3 speed=1.1", "p4 speed=1.2", "p5 speed=1.2", "WRONG r2",  # tie: the earlier, 9
    "p6 speed=1.203", "BUILD_FAIL r3", "p4 speed=1.2", None,  # within min_gain, a repeat: 9
    "p7 speed=1.5", "CRASH r4", "p8 speed=1.45", BASELINE_COPY,  # 16
    "p9 speed=1.4", "p10 speed=1.6", "WRONG r5", "p7 speed=1.5",  # 21
    "p11 speed=1.55", "p12 speed=1.59", "BUILD_FAIL r6", "CRASH r6",  # 21
    "p13 speed=2.0", "p14 speed=1.9", "p15 speed=2.0", "TIMEOUT r7",  # 28
]
INCUMBENTS = [BASELINE, BASELINE, 9, 9, 16, 21, 21, 28]
N = 4


class ScriptedLLM:
    """Answers the i-th call with ``script[i]`` in an assembly block (``None``: a reply without
    one) and records every request and reply."""

    def __init__(self, script: list[str | None]) -> None:
        self.script = script
        self.requests: list[tuple[list[dict[str, str]], str]] = []
        self.replies: list[str] = []

    def complete(self, messages, *, prompt_id):
        i = len(self.requests)
        self.requests.append((copy.deepcopy(messages), prompt_id))
        asm = self.script[i]
        reply = f"reply {i}: no code" if asm is None else f"reply {i}\n```assembly\n{asm}\n```"
        self.replies.append(reply)
        call = LLMCall(model="fake", prompt_id=prompt_id, response_status="completed",
                       input_tokens=sum(len(m["content"]) for m in messages),
                       output_tokens=len(reply), reasoning_tokens=0, latency_s=0.0)
        return reply, call


class FakeEvaluator:
    """Scores a program by its text: ``speed=1.2`` is correct at 1.2x; ``BUILD_FAIL``, ``WRONG``,
    ``CRASH``, and ``TIMEOUT`` fail that way. Records every program it is asked to evaluate and
    refuses copies of the -O3 baseline, which must never reach it."""

    def __init__(self) -> None:
        self.programs: list[str | None] = []

    def __call__(self, problem: Problem, gen: Generation, cfg) -> EvalResult:
        self.programs.append(gen.asm)
        asm = gen.asm
        if asm is None:
            return EvalResult(problem.id, Status.NO_ASM,
                              detail="no ```assembly block; responses ['completed']")
        assert normalize_asm(asm) != normalize_asm(problem.baseline_asm), "-O3 was re-timed"
        match asm.split()[0]:
            case "BUILD_FAIL":
                return EvalResult(problem.id, Status.BUILD_FAIL,
                                  detail=f"Error: no such instruction <{asm}>\n")
            case "WRONG":
                return EvalResult(problem.id, Status.WRONG_OUTPUT, failed_test=1,
                                  detail=f"got '<{asm}>'")
            case "CRASH":
                return EvalResult(problem.id, Status.CRASH, failed_test=0,
                                  detail=f"killed by signal 11: <{asm}>")
            case "TIMEOUT":
                return EvalResult(problem.id, Status.TIMEOUT, failed_test=1,
                                  detail="exceeded 10s")
        speed = float(asm.split("speed=")[1].split()[0])
        return EvalResult(problem.id, Status.OK, speedup=speed, baseline_mean_s=1.0,
                          candidate_mean_s=1.0 / speed)


@pytest.fixture
def evaluator(monkeypatch) -> FakeEvaluator:
    fake = FakeEvaluator()
    monkeypatch.setattr(shared_nodes, "evaluate", fake)  # what EvaluateAsm.exec calls
    return fake


BASE_CONFIG = load_config()
GENERATE = refined_best_of_n.generate
ONE_ROUND = 1_000  # samples_per_round >= max_calls: a single round, i.e. plain best-of-N


def config(max_calls: int, samples_per_round: int = N, min_gain: float = 0.005):
    return replace(BASE_CONFIG, refined_best_of_n=RefinedBestOfNConfig(max_calls, samples_per_round,
                                                                       min_gain))


def run(script, max_calls=None, samples_per_round=N, min_gain=0.005):
    llm = ScriptedLLM(script)
    cfg = config(len(script) if max_calls is None else max_calls, samples_per_round, min_gain)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # PocketFlow warns when an action has no successor
        gen = GENERATE(PROBLEM, llm, cfg)
    return gen, llm


def ok(speedup: float) -> EvalResult:
    return EvalResult(PROBLEM.id, Status.OK, speedup=speedup, baseline_mean_s=1.0,
                      candidate_mean_s=1.0 / speedup)


def content(llm: ScriptedLLM, i: int) -> str:
    return "\n".join(m["content"] for m in llm.requests[i][0])


# --- one round: plain best-of-N ---


def test_one_round_is_plain_best_of_n(evaluator):
    gen, llm = run(SCRIPT, max_calls=12, samples_per_round=12)
    assert len(llm.requests) == len(gen.llm_calls) == 12
    for messages, prompt_id in llm.requests:
        assert messages == build_prompt(PROBLEM) and prompt_id == PROMPT_ID
        assert all(m["role"] != "assistant" for m in messages)
    assert gen.method == "refined_best_of_n"
    assert gen.round_parent == (BASELINE,) * 12
    assert gen.asm == "p4 speed=1.2"  # the earlier of the two fastest


# --- rounds ---


def test_refined_samples_of_a_round_share_one_prompt(evaluator):
    gen, llm = run(SCRIPT)
    assert len(llm.requests) == len(gen.llm_calls) == len(gen.rounds) == 32
    assert gen.method == "refined_best_of_n"
    for r in range(len(SCRIPT) // N):
        first = llm.requests[r * N]
        assert all(llm.requests[r * N + j] == first for j in range(1, N))
        assert gen.round_parent[r * N:(r + 1) * N] == (INCUMBENTS[r - 1] if r else BASELINE,) * N
    assert llm.requests[0] == (build_prompt(PROBLEM), PROMPT_ID)  # round 0 is single_call
    # literal ids: they are recorded on every LLMCall in generations.jsonl
    assert [p for _, p in llm.requests[::N]] == [PROMPT_ID, "rbon_retry", "rbon_retry",
                                                 *["rbon_refine"] * 5]


def test_all_failed_round_retries_with_a_note_naming_statuses_and_tests(evaluator):
    _, llm = run(SCRIPT, max_calls=9)
    (message,), prompt_id = llm.requests[4]
    assert prompt_id == RETRY_PROMPT_ID and message["role"] == "user"
    text = message["content"]
    assert text.endswith(PROBLEM.standard_prompt)
    for line in ["- attempt 1: build_fail: Error: no such instruction <BUILD_FAIL r0>",
                 "- attempt 2: wrong_output on test 1: got '<WRONG r0>'",
                 "- attempt 3: no_asm: no ```assembly block",
                 "- attempt 4: crash on test 0: killed by signal 11: <CRASH r0>",
                 "A response status of 'incomplete' means your reply hit the output token limit",
                 "Test 1 stdin:\n```\n5 5\n```\nExpected stdout:\n```\n10\n```",
                 "Test 0 stdin:\n```\n1 2\n```\nExpected stdout:\n```\n3\n```",
                 "No correct attempt was more than 0.5% faster than gcc -O3."]:
        assert line in text
    assert "output token limit" not in content(llm, 8)  # round 1 has no reply without assembly

    _, llm = run(["WRONG a", "TIMEOUT b", "CRASH c", "WRONG d", "p speed=1.1"])
    assert content(llm, 4).count("Test 1 stdin:") == content(llm, 4).count("Test 0 stdin:") == 1


def test_min_gain_keeps_the_baseline_for_noise_and_replaces_it_for_real_gains(evaluator):
    gen, llm = run(SCRIPT, max_calls=13)
    (message,), prompt_id = llm.requests[8]  # after 1.003x, a copy of -O3, and 0.97x
    assert prompt_id == RETRY_PROMPT_ID
    for line in ["- attempt 1: correct but 1.003x as fast as gcc -O3",
                 "- attempt 2: correct but 1.000x as fast as gcc -O3 (identical to the gcc -O3",
                 "- attempt 3: correct but 0.970x as fast as gcc -O3",
                 "- attempt 4: timeout on test 1: exceeded 10s",
                 "No correct attempt was more than 0.5% faster than gcc -O3."]:
        assert line in message["content"]
    assert "<BUILD_FAIL r0>" not in message["content"]  # only the last round's note
    assert gen.round_parent[8:13] == (BASELINE,) * 4 + (9,)  # 1.2x replaces -O3

    assert next_incumbent([ok(1.003)], 0, BASELINE, 0.005) == BASELINE
    assert next_incumbent([ok(1.003)], 0, BASELINE, 0.0) == 0
    assert next_incumbent([ok(1.2)], 0, BASELINE, 0.005) == 0
    assert next_incumbent([ok(1.2), ok(1.203)], 1, 0, 0.005) == 0
    assert next_incumbent([ok(1.2), ok(1.21)], 1, 0, 0.005) == 1
    # relative to the incumbent, as the policy tree counts gains: +0.008x is +0.4% here ...
    assert next_incumbent([ok(2.0), ok(2.008)], 1, 0, 0.005) == 0
    assert next_incumbent([ok(2.0), ok(2.012)], 1, 0, 0.005) == 1
    # ... and +0.6% over the incumbent counts though it is +0.3% over the faster 1.203x
    assert next_incumbent([ok(1.2), ok(1.203), ok(1.207)], 2, 0, 0.005) == 2


def test_ties_keep_the_earlier(evaluator):
    assert next_incumbent([ok(1.0)], 0, BASELINE, 0.0) == BASELINE
    assert next_incumbent([ok(1.2), ok(1.2)], 0, BASELINE, 0.0) == 0
    assert next_incumbent([ok(1.2), ok(1.2)], 1, 0, 0.0) == 0
    gen, _ = run(SCRIPT)
    assert gen.round_parent[12] == 9 and gen.round_parent[16] == 9  # not 10, not the repeat 14
    assert gen.asm == "p13 speed=2.0"  # rounds 28 and 30 tie; 28 is returned


def test_refine_prompt_shows_the_incumbent_its_feedback_and_the_last_round(evaluator):
    gen, llm = run(SCRIPT)
    messages, prompt_id = llm.requests[12]  # round 9 just became the incumbent
    assert prompt_id == REFINE_PROMPT_ID
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0] == build_prompt(PROBLEM)[0]
    assert messages[1]["content"] == llm.replies[9]
    feedback = messages[2]["content"]
    assert "Your program is correct on all tests and runs 1.200x as fast as gcc -O3" in feedback
    assert "- attempt 2: correct, 1.200x as fast as gcc -O3: the program above" in feedback
    assert "- attempt 3: correct but 1.200x as fast as gcc -O3" in feedback
    assert "No correct attempt" not in feedback
    assert feedback.endswith("Answer with the complete program in a single ```assembly block.")

    messages, _ = llm.requests[16]  # nothing in round 3 beat round 9 by 0.5%
    assert messages[1]["content"] == llm.replies[9]
    assert "- attempt 1: correct but 1.203x as fast as gcc -O3" in messages[2]["content"]
    assert ("No correct attempt was more than 0.5% faster than the program above."
            in messages[2]["content"])
    assert llm.requests[20][0][1]["content"] == llm.replies[16]


def test_context_stays_bounded(evaluator):
    huge = "BUILD_FAIL " + "x" * 5000
    _, llm = run([huge, *SCRIPT[1:]])
    assert "more characters]" in content(llm, 4)  # the build error is clipped
    assert len(content(llm, 4)) < MAX_IO_CHARS + 1000 + len(PROBLEM.standard_prompt)
    assert all(len(messages) <= 3 for messages, _ in llm.requests)
    for r in range(1, len(SCRIPT) // N):  # each round sees only the round before it
        text = content(llm, r * N)
        assert "- attempt 4:" in text
        assert all(f" r{q}>" not in text for q in range(r - 1))

    # rounds that end alike get the same prompt, however many rounds came before them
    failed = ["BUILD_FAIL same", "WRONG same", None, "CRASH same"]
    for script, first, prompt_id in [(failed * 8, 1, RETRY_PROMPT_ID),
                                     (["p speed=1.5", *failed[1:]] + failed * 7, 2,
                                      REFINE_PROMPT_ID)]:
        _, llm = run(script)
        assert llm.requests[first * N][1] == prompt_id
        assert all(llm.requests[r * N] == llm.requests[first * N] for r in range(first, 8))


# --- shared invariants ---


@pytest.mark.parametrize("n", [N, ONE_ROUND], ids=["rounds", "one-round"])
def test_first_k_rounds_equal_a_k_call_run(evaluator, n):
    long_gen, long_llm = run(SCRIPT, samples_per_round=n)
    for k in range(1, len(SCRIPT) + 1):
        gen, llm = run(SCRIPT, max_calls=k, samples_per_round=n)
        assert llm.requests == long_llm.requests[:k]
        assert gen.rounds == long_gen.rounds[:k]
        assert gen.round_asm == long_gen.round_asm[:k]
        assert gen.round_parent == long_gen.round_parent[:k]
        assert gen.asm == long_gen.round_asm[curve.prefix_pick(long_gen, k)]


@pytest.mark.parametrize("n", [N, ONE_ROUND], ids=["rounds", "one-round"])
def test_generation_records_every_round_with_its_lineage(evaluator, n):
    gen, llm = run(SCRIPT, samples_per_round=n)
    assert len(gen.llm_calls) == len(gen.rounds) == len(gen.round_asm) == len(gen.round_parent)
    assert [c.prompt_id for c in gen.llm_calls] == [p for _, p in llm.requests]
    for i, parent in enumerate(gen.round_parent):
        assert parent == BASELINE or (parent < i - i % n and gen.rounds[parent].ok)
    best = best_round(list(gen.rounds))
    assert gen.asm == gen.round_asm[best] and gen.reply == llm.replies[best]
    assert Generation.from_dict(json.loads(json.dumps(asdict(gen)))) == gen


def test_generations_work_with_budget_curve(evaluator, monkeypatch):
    monkeypatch.setattr(curve, "evaluate", evaluator)
    gens = {name: run(SCRIPT, samples_per_round=n)[0]
            for name, n in [("one round", ONE_ROUND), ("rounds", N)]}
    ks = [1, 4, 12, 32]
    curves = curve.budget_curve({name: [g] for name, g in gens.items()}, {PROBLEM.id: PROBLEM},
                                ks, BASE_CONFIG.eval)
    for points in curves.values():
        assert [p["k"] for p in points] == ks and [p["mean_calls_used"] for p in points] == ks
        assert [p["speedups"][PROBLEM.id] for p in points] == [1.0, 1.0, 1.2, 2.0]


def test_each_distinct_program_is_evaluated_once(evaluator):
    a, a_relaid = "A speed=1.1\n\tnop\n\tret", "A speed=1.1  \n\n\tnop\n\n\tret\t"
    gen, _ = run([a, a, a_relaid, BASELINE_COPY, PROBLEM.baseline_asm, None, None,
                  "B speed=1.3"], samples_per_round=ONE_ROUND)
    assert evaluator.programs == [a, None, None, "B speed=1.3"]
    assert len(gen.llm_calls) == len(gen.rounds) == 8
    assert gen.rounds[1] == gen.rounds[2] == gen.rounds[0]
    for copied in gen.rounds[3:5]:
        assert copied.ok and copied.speedup == 1.0 and copied.baseline_mean_s is None
        assert "identical to the gcc -O3 baseline" in copied.detail
    assert [r.status for r in gen.rounds[5:7]] == [Status.NO_ASM] * 2
    assert gen.round_asm[5:7] == (None, None) and gen.asm == "B speed=1.3"


def test_failed_programs_are_cached_too_but_only_within_a_run(evaluator):
    failed = ["BUILD_FAIL x", "WRONG x", "CRASH x", "TIMEOUT x"]
    script = [*failed, *failed, "B speed=1.3"]
    first, _ = run(script, samples_per_round=ONE_ROUND)
    second, _ = run(script, samples_per_round=ONE_ROUND)  # like another problem: no cache
    assert evaluator.programs == [*failed, "B speed=1.3"] * 2
    assert first.rounds[4:8] == first.rounds[:4] and second.rounds == first.rounds


def test_a_copy_of_the_baseline_never_becomes_the_incumbent(evaluator):
    gen, llm = run([BASELINE_COPY] * 2 + ["p speed=0.9"] * 2, samples_per_round=2,
                   min_gain=0.0)
    assert llm.requests[2][1] == RETRY_PROMPT_ID and gen.round_parent == (BASELINE,) * 4
    assert evaluator.programs == ["p speed=0.9"]
    assert gen.asm == BASELINE_COPY  # still better than a correct but slower program


def test_reply_without_assembly_is_a_no_asm_round(evaluator):
    gen, llm = run([None, None], samples_per_round=1)
    assert [r.status for r in gen.rounds] == [Status.NO_ASM] * 2
    assert gen.asm is None and gen.round_asm == (None, None)
    assert evaluator.programs == [None, None]
    assert "- attempt 1: no_asm: no ```assembly block" in content(llm, 1)


def test_one_call_is_single_call(evaluator):
    gen, llm = run(SCRIPT, max_calls=1)
    assert llm.requests == [(build_prompt(PROBLEM), PROMPT_ID)]
    assert gen.round_asm == ("BUILD_FAIL r0",) and gen.round_parent == (BASELINE,)


# --- configuration and registration ---


def test_default_config():
    assert yaml.safe_load(DEFAULT_CONFIG.read_text())["refined_best_of_n"] == {
        "max_calls": 32, "samples_per_round": 4, "min_gain": 0.005,
    }


@pytest.mark.parametrize("bad", [{"max_calls": 0}, {"max_calls": 32.0}, {"max_calls": True},
                                 {"samples_per_round": 0}, {"samples_per_round": 4.0},
                                 {"min_gain": -0.001}, {"min_gain": math.nan},
                                 {"min_gain": math.inf}])
def test_config_validation(bad, tmp_path):
    good = {"max_calls": 32, "samples_per_round": 4, "min_gain": 0.005}
    with pytest.raises(ValueError, match="refined_best_of_n"):
        RefinedBestOfNConfig(**{**good, **bad})
    override = tmp_path / "bad.yaml"
    override.write_text(yaml.safe_dump({"refined_best_of_n": bad}))
    with pytest.raises(ValueError, match="refined_best_of_n"):
        load_config(override)


def test_registered_in_main(tmp_path, monkeypatch, evaluator):
    assert cli.METHODS["refined_best_of_n"] is GENERATE
    assert "refined_best_of_n" in cli.EXEC_METHODS
    args = cli.build_parser().parse_args(["generate", "--method", "refined_best_of_n"])
    assert args.method == "refined_best_of_n"

    checked = []
    monkeypatch.setattr(cli, "prepare_timing", lambda cfg, what: checked.append(what))
    monkeypatch.setattr(cli, "load_problems", lambda cfg, limit: [PROBLEM])
    llm = ScriptedLLM(SCRIPT)
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda cfg: llm)
    cli.generate(config(8), "refined_best_of_n", tmp_path)
    (line,) = (tmp_path / cli.GENERATIONS).read_text().splitlines()
    gen = Generation.from_dict(json.loads(line))
    assert gen.method == "refined_best_of_n" and len(gen.llm_calls) == len(gen.rounds) == 8
    assert checked == ["generate --method refined_best_of_n"]


# --- end to end ---

SLOW_C = r"""
#include <stdio.h>

int main(void) {
    long n, count = 0;
    if (scanf("%ld", &n) != 1) return 1;
    for (long i = 1; i <= n; i++)
        if (i % 3 == 0 || i % 5 == 0) count++;
    printf("%ld\n", count);
    return 0;
}
"""
FAST_C = r"""
#include <stdio.h>

int main(void) {
    long n;
    if (scanf("%ld", &n) != 1) return 1;
    printf("%ld\n", n / 3 + n / 5 - n / 15);
    return 0;
}
"""
WRONG_C = FAST_C.replace(" - n / 15", "")  # counts multiples of 15 twice
CRASH_C = "#include <stdlib.h>\nint main(void) { abort(); }\n"


def compile_o3(code: str) -> str:
    return subprocess.run(["gcc", "-O3", "-S", "-x", "c", "-", "-o", "-"], input=code,
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.skipif(
    platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}
    or not all(shutil.which(tool) for tool in ("gcc", "hyperfine", "prlimit")),
    reason="builds, tests, and times x86-64 programs with gcc, prlimit, and hyperfine",
)
def test_refined_best_of_n_end_to_end():
    slow, fast, wrong, crash = map(compile_o3, (SLOW_C, FAST_C, WRONG_C, CRASH_C))
    problem = Problem(
        id=1,
        c_code=SLOW_C,
        baseline_asm=slow,
        tests=tuple(Case(f"{n}\n", f"{n // 3 + n // 5 - n // 15}\n")
                    for n in (1, 15, 1000, 50_000_000)),
        baseline_binary=b"",
        standard_prompt="Make this program faster.",
    )
    broken = slow + "\n\tbogus\t%eax\n"
    cfg = replace(
        BASE_CONFIG,
        eval=replace(BASE_CONFIG.eval, pin_cpus=False, warmup_runs=0, timing_repeats=4),
        refined_best_of_n=RefinedBestOfNConfig(max_calls=6, samples_per_round=2, min_gain=0.005),
    )
    llm = ScriptedLLM([broken, wrong, slow, fast, crash, fast])
    gen = GENERATE(problem, llm, cfg)

    assert [r.status for r in gen.rounds] == [Status.BUILD_FAIL, Status.WRONG_OUTPUT, Status.OK,
                                              Status.OK, Status.CRASH, Status.OK]
    assert gen.rounds[1].failed_test == 1 and gen.rounds[4].failed_test == 0
    assert gen.rounds[2].speedup == 1.0 and gen.rounds[2].baseline_mean_s is None  # -O3 copy
    # Only the direction: emulated x86-64 (Docker on Apple Silicon) spends most of every run on
    # process startup, which squeezes a ~15x native speedup to under 2x.
    assert gen.rounds[3].speedup > 1.2 and gen.rounds[5] == gen.rounds[3]  # repeat not re-timed
    assert gen.asm == fast.strip("\n") and gen.round_parent == (-1, -1, -1, -1, 3, 3)
    retry = content(llm, 2)
    assert llm.requests[2][1] == RETRY_PROMPT_ID
    assert "build_fail" in retry and "bogus" in retry and "wrong_output on test 1" in retry
    assert "Test 1 stdin:\n```\n15\n```\nExpected stdout:\n```\n7\n```" in retry
    messages, prompt_id = llm.requests[4]
    assert prompt_id == REFINE_PROMPT_ID and messages[1]["content"] == llm.replies[3]
    assert "as fast as gcc -O3" in messages[2]["content"]
