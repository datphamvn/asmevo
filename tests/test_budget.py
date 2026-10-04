import math
import threading
from concurrent.futures import ThreadPoolExecutor

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


def test_default_is_the_protocol_budget():
    assert Budget().max_candidates == 12


def test_zero_and_negative_budgets():
    with pytest.raises(BudgetExhausted):
        Budget(0).charge()
    with pytest.raises(ValueError):
        Budget(-1)


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


def test_guard_caps_a_greedy_method_at_12_calls():
    inner, budget = FakeLLM(), Budget(12)
    llm = budget.guard(inner)
    made = 0
    with pytest.raises(BudgetExhausted):
        while True:
            budget.charge()
            llm.complete([{"role": "user", "content": "x"}], prompt_id="p")
            made += 1
    assert made == inner.calls == len(budget.calls) == 12


def test_parallel_offspring_stay_within_budget():
    inner, budget = FakeLLM(), Budget(12)
    llm = budget.guard(inner)

    def offspring(_):  # each worker tries 5 candidates, 20 in all
        made = 0
        for _ in range(5):
            try:
                budget.charge()
            except BudgetExhausted:
                break
            llm.complete([], prompt_id="p")
            made += 1
        return made

    with ThreadPoolExecutor(max_workers=4) as pool:
        made = sum(pool.map(offspring, range(4)))
    assert made == inner.calls == len(budget.calls) == 12


def test_failed_call_keeps_its_charge():
    budget = Budget(2)
    llm = budget.guard(FakeLLM(fail_on=1))
    budget.charge()
    with pytest.raises(ConnectionError):
        llm.complete([], prompt_id="p")
    assert budget.remaining == 1
    assert budget.calls == ()
    with pytest.raises(RuntimeError):  # the failed attempt used up its charge
        llm.complete([], prompt_id="p")


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


def test_check_trace_rejects_unguarded_call():
    budget = Budget(12)
    with pytest.raises(ValueError, match="did not record"):
        budget.check_trace(make_trace([(make_call(1),)]))


def test_check_trace_rejects_call_listed_twice():
    budget = Budget(12)
    llm = budget.guard(FakeLLM())
    budget.charge()
    call = llm.complete([], prompt_id="p")[1]
    with pytest.raises(ValueError, match="listed twice"):  # would double its cost
        budget.check_trace(make_trace([(call,), (call,)]))
