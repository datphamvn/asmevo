import json
import math
import random
import threading
from dataclasses import asdict

import pytest

from asmevo.config import load_config
from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as Case
from asmevo.methods import nodes
from asmevo.methods.evolution.flow import NAME, search
from asmevo.methods.prompt import MUTATE_PROMPT_ID
from asmevo.methods.single_call.prompt import PROMPT_ID as BASELINE_PROMPT_ID
from asmevo.schemas import EvalResult, LLMCall, SearchTrace, Status
from asmevo.search.budget import Budget
from asmevo.search.selection import survive

BASELINE_ASM = "main:\n\txorl %eax, %eax\n\tret"
PROBLEM = Problem(id=7, c_code="int main(){}", baseline_asm=BASELINE_ASM,
                  tests=(Case("1\n", "1\n"), Case("2\n", "2\n")),
                  baseline_binary=b"", standard_prompt="Optimize this program.")
CFG = load_config()  # generations 3 x 4 offspring, population 4, tournament 2


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


@pytest.fixture
def evaluator(monkeypatch):
    fake = FakeEvaluate(wrong=frozenset({2, 5, 9}))
    monkeypatch.setattr(nodes, "evaluate_asm", fake)
    return fake


def run(llm, seed: int = 0) -> tuple[SearchTrace, Budget]:
    budget = Budget(CFG.search.max_candidates)
    trace = search(PROBLEM, budget.guard(llm), CFG, budget, random.Random(seed), seed)
    budget.check_trace(trace)  # every LLM call is in the trace exactly once
    return trace, budget


def test_spends_the_whole_budget_in_three_generations(evaluator):
    trace, budget = run(FakeLLM())
    assert budget.used == 12
    assert [c.generation for c in trace.candidates] == [0] + [1] * 4 + [2] * 4 + [3] * 4
    assert len({c.id for c in trace.candidates}) == 13
    assert trace.candidates[0].id == "7-evolution-s0-0" and trace.method == NAME
    assert all(len(c.llm_calls) == 1 for c in trace.candidates[1:])


def test_parents_are_survivors_of_the_previous_generation(evaluator):
    trace, _ = run(FakeLLM())
    by_id = {c.id: c for c in trace.candidates}
    population = (trace.candidates[0],)
    for gen in (1, 2, 3):
        children = [c for c in trace.candidates if c.generation == gen]
        for c in children:
            (parent_id,) = c.parent_ids
            assert parent_id in {p.id for p in population}
            assert by_id[parent_id].valid
        population = survive(population, children, CFG.search.population_size)
    assert all(c.parent_ids == (trace.candidates[0].id,)
               for c in trace.candidates if c.generation == 1)


def test_fitness_and_feedback(evaluator):
    trace, _ = run(FakeLLM())
    for c in trace.candidates[1:]:
        if c.result.ok:
            assert c.fitness == c.result.speedup
            assert "as fast as gcc -O3" in c.feedback
        else:
            assert c.fitness == -math.inf
            assert "wrong output" in c.feedback and "Test 1 stdin" in c.feedback
    assert not trace.candidates[2].valid  # program 2 is wrong


def test_selects_the_fittest_candidate(evaluator):
    trace, _ = run(FakeLLM())
    assert trace.selected_id == max(trace.candidates, key=lambda c: c.fitness).id
    assert trace.final == trace.selected.result and not trace.fallback


def test_child_prompts_carry_the_parent_and_its_feedback(evaluator):
    llm = FakeLLM()
    trace, _ = run(llm)
    for c, (messages, prompt_id) in zip(trace.candidates[1:], llm.requests, strict=True):
        parent = next(p for p in trace.candidates if p.id == c.parent_ids[0])
        assert c.prompt_id == prompt_id
        assert messages[0]["content"] == PROBLEM.standard_prompt
        if parent.generation == 0:
            assert prompt_id == BASELINE_PROMPT_ID and len(messages) == 1
        else:
            assert prompt_id == MUTATE_PROMPT_ID
            assert messages[1] == {"role": "assistant", "content": parent.llm_calls[0].reply}
            assert parent.feedback in messages[2]["content"]


def test_each_distinct_program_is_evaluated_once(evaluator):
    # Calls cycle through: the baseline again, program 1, no assembly block, program 3.
    replies = [asm_reply(BASELINE_ASM), asm_reply(program(1)), "no code", asm_reply(program(3))]
    trace, _ = run(FakeLLM(lambda n: replies[(n - 1) % 4]))
    assert sorted(evaluator.calls, key=str) == sorted([program(1), program(3)] + [None] * 3,
                                                      key=str)
    copies = [c for c in trace.candidates[1:] if c.asm_hash == trace.candidates[0].asm_hash]
    assert len(copies) == 3 and all(c.fitness == 1.0 for c in copies)
    no_asm = [c for c in trace.candidates if c.asm is None and c.generation > 0]
    assert len(no_asm) == 3 and all(not c.valid and c.asm_hash is None for c in no_asm)
    assert "Nothing could be built" in no_asm[0].feedback


def test_keeps_the_baseline_when_nothing_works(monkeypatch):
    monkeypatch.setattr(nodes, "evaluate_asm", FakeEvaluate(wrong=frozenset(range(1, 13))))
    trace, _ = run(FakeLLM())
    assert trace.selected_id == trace.candidates[0].id
    assert trace.final.ok and trace.final.speedup == 1.0
    assert all(c.parent_ids == (trace.candidates[0].id,) for c in trace.candidates[1:])


def test_same_seed_same_lineage(evaluator):
    first, _ = run(FakeLLM(), seed=3)
    again, _ = run(FakeLLM(), seed=3)
    assert first == again
    assert first.candidates[0].id == "7-evolution-s3-0" and first.seed == 3


def test_trace_round_trips_through_jsonl(evaluator):
    trace, _ = run(FakeLLM())
    line = json.dumps(asdict(trace))
    assert SearchTrace.from_dict(json.loads(line)) == trace
