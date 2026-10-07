"""Fake model and evaluator for the search-method tests: no network, no build, no timing."""

import random
import threading

from asmevo.config import load_config
from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as Case
from asmevo.schemas import EvalResult, LLMCall, SearchTrace, Status
from asmevo.search.budget import Budget

BASELINE_ASM = "main:\n\txorl %eax, %eax\n\tret"
PROBLEM = Problem(id=7, c_code="int main(){}", baseline_asm=BASELINE_ASM,
                  tests=(Case("1\n", "1\n"), Case("2\n", "2\n")),
                  baseline_binary=b"", standard_prompt="Optimize this program.")
CFG = load_config()  # 12 candidates; evolution: 3 generations x 4 offspring, population 4


def asm_reply(asm: str) -> str:
    return f"Here you go.\n```assembly\n{asm}\n```\n"


def program(n: int) -> str:
    return f"main:\n\tmovl ${n}, %eax\n\tret"


class FakeLLM:
    """Replies with ``reply(n)`` on the n-th call (1-based) and keeps every request."""

    def __init__(self, reply=lambda n: asm_reply(program(n))) -> None:
        self.reply = reply
        self.requests: list[tuple[list, str]] = []
        self.lock = threading.Lock()

    def complete(self, messages, *, prompt_id):
        with self.lock:
            self.requests.append((messages, prompt_id))
            n = len(self.requests)
        reply = self.reply(n)
        return reply, LLMCall(model="m", prompt_id=prompt_id, response_status="completed",
                              input_tokens=100, output_tokens=n, reasoning_tokens=0,
                              cost_usd=0.001, latency_s=0.1, messages=tuple(messages),
                              reply=reply)


class FakeEvaluate:
    """Stands in for ``evaluate_asm`` (T12): ``program(n)`` is n% faster than -O3 unless
    ``n`` is in ``wrong``."""

    def __init__(self, wrong: frozenset[int] = frozenset()) -> None:
        self.wrong = wrong
        self.calls: list[str | None] = []

    def __call__(self, problem, asm, tests, cfg):
        assert tests == problem.tests
        self.calls.append(asm)
        if asm is None:
            return EvalResult(problem_id=problem.id, status=Status.NO_ASM, detail="no block")
        n = int(asm.split("$")[1].split(",")[0])
        if n in self.wrong:
            return EvalResult(problem_id=problem.id, status=Status.WRONG_OUTPUT,
                              failed_test=1, detail="got '3'")
        return EvalResult(problem_id=problem.id, status=Status.OK, speedup=1 + n / 100)


def run(search, llm, seed: int = 0) -> tuple[SearchTrace, Budget]:
    """``search`` as the experiment runner calls it, checking the trace against the budget."""
    budget = Budget(CFG.search.max_candidates)
    trace = search(PROBLEM, budget.guard(llm), CFG, budget, random.Random(seed), seed)
    budget.check_trace(trace)  # every LLM call is in the trace exactly once
    return trace, budget
