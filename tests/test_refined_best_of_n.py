"""refined_best_of_n with a scripted fake LLM and a fake evaluator (no API key, any
host), plus one end-to-end run through the real build/test/time pipeline (Linux x86-64 only)."""

from __future__ import annotations

import copy
import platform
import shutil
import subprocess
import warnings
from dataclasses import replace

import pytest

import asmevo.methods.nodes as shared_nodes
from asmevo.config import RefinedBestOfNConfig, load_config
from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as Case
from asmevo.eval import curve
from asmevo.methods.refined_best_of_n import flow as refined_best_of_n
from asmevo.methods.refined_best_of_n.nodes import BASELINE, next_incumbent
from asmevo.methods.refined_best_of_n.prompt import REFINE_PROMPT_ID, RETRY_PROMPT_ID
from asmevo.methods.single_call.prompt import PROMPT_ID, build_prompt
from asmevo.schemas import EvalResult, Generation, LLMCall, Status
from asmevo.search.selection import asm_hash

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
                       output_tokens=len(reply), reasoning_tokens=0, cost_usd=0.0,
                       latency_s=0.0, messages=tuple(dict(m) for m in messages), reply=reply)
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
        assert asm_hash(asm) != asm_hash(problem.baseline_asm), "-O3 was re-timed"
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


def test_one_round_is_plain_best_of_n(evaluator):
    gen, llm = run(SCRIPT, max_calls=12, samples_per_round=12)
    assert len(llm.requests) == len(gen.llm_calls) == 12
    for messages, prompt_id in llm.requests:
        assert messages == build_prompt(PROBLEM) and prompt_id == PROMPT_ID
        assert all(m["role"] != "assistant" for m in messages)
    assert gen.method == "refined_best_of_n"
    assert gen.round_parent == (BASELINE,) * 12
    assert gen.asm == "p4 speed=1.2"  # the earlier of the two fastest


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
    # relative to the incumbent, +0.008x is +0.4% here ...
    assert next_incumbent([ok(2.0), ok(2.008)], 1, 0, 0.005) == 0
    assert next_incumbent([ok(2.0), ok(2.012)], 1, 0, 0.005) == 1
    # ... and +0.6% over the incumbent counts though it is +0.3% over the faster 1.203x
    assert next_incumbent([ok(1.2), ok(1.203), ok(1.207)], 2, 0, 0.005) == 2


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


def test_a_copy_of_the_baseline_never_becomes_the_incumbent(evaluator):
    gen, llm = run([BASELINE_COPY] * 2 + ["p speed=0.9"] * 2, samples_per_round=2,
                   min_gain=0.0)
    assert llm.requests[2][1] == RETRY_PROMPT_ID and gen.round_parent == (BASELINE,) * 4
    assert evaluator.programs == ["p speed=0.9"]
    assert gen.asm == BASELINE_COPY  # still better than a correct but slower program


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
