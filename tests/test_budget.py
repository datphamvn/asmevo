import math
import threading

import pytest

from asmevo.schemas import Candidate, EvalResult, LLMCall, SearchTrace, Status, candidate_id
from asmevo.search.budget import Budget, BudgetExhausted


def make_call(n: int) -> LLMCall:
    return LLMCall(model="m", prompt_id="mutate-v1", response_status="completed",
                   input_tokens=100, output_tokens=10 * n, reasoning_tokens=0,
                   cost_usd=0.01 * n, latency_s=1.0, reply=f"reply {n}")


class FakeLLM:
    def __init__(self, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def complete(self, messages, *, prompt_id):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n == self.fail_on:
            raise ConnectionError("endpoint down")
        return f"reply {n}", make_call(n)


def make_candidate(n: int, calls: tuple[LLMCall, ...]) -> Candidate:
    return Candidate(
        id=candidate_id(7, "best_of_n", 0, n), problem_id=7, method="best_of_n", seed=0,
        generation=0 if n == 0 else 1, parent_ids=() if n == 0 else ("7-best_of_n-s0-0",),
        prompt_id=None if n == 0 else "mutate-v1", asm="ret", asm_hash=f"h{n}",
        result=None, fitness=1.0 if n == 0 else -math.inf, feedback="", llm_calls=calls,
    )


def make_trace(candidate_calls: list[tuple[LLMCall, ...]]) -> SearchTrace:
    candidates = (make_candidate(0, ()),) + tuple(
        make_candidate(n, calls) for n, calls in enumerate(candidate_calls, 1)
    )
    return SearchTrace(problem_id=7, method="best_of_n", seed=0, candidates=candidates,
                       selected_id=candidates[0].id,
                       final=EvalResult(problem_id=7, status=Status.OK), fallback=False)


def test_charge_until_exhausted():
    budget = Budget(3)
    for left in (2, 1, 0):
        budget.charge()
        assert budget.remaining == left
    with pytest.raises(BudgetExhausted):
        budget.charge()
    assert budget.used == 3
    assert budget.remaining == 0


def test_concurrent_charges_never_exceed_cap():
    budget = Budget(12)
    granted = []

    def worker():
        for _ in range(10):
            try:
                budget.charge()
                granted.append(1)
            except BudgetExhausted:
                pass

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 12
    assert budget.remaining == 0


def test_guard_rejects_uncharged_call():
    inner = FakeLLM()
    llm = Budget(12).guard(inner)
    with pytest.raises(RuntimeError, match="charge"):
        llm.complete([{"role": "user", "content": "x"}], prompt_id="p")
    assert inner.calls == 0


def test_check_trace_accepts_exact_accounting():
    budget = Budget(12)
    llm = budget.guard(FakeLLM())
    calls = []
    for _ in range(3):
        budget.charge()
        calls.append(llm.complete([], prompt_id="p")[1])
    trace = make_trace([(c,) for c in calls])
    budget.check_trace(trace)
    assert trace.total_output_tokens == sum(c.output_tokens for c in budget.calls)


def test_check_trace_rejects_dropped_call():
    budget = Budget(12)
    llm = budget.guard(FakeLLM())
    calls = []
    for _ in range(2):
        budget.charge()
        calls.append(llm.complete([], prompt_id="p")[1])
    with pytest.raises(ValueError, match="misses 1"):
        budget.check_trace(make_trace([(calls[0],)]))


