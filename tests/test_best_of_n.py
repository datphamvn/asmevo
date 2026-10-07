import json
from dataclasses import asdict

from fakes import PROBLEM, FakeEvaluate, FakeLLM, asm_reply, program
from fakes import run as run_search

from asmevo.methods import nodes
from asmevo.methods.best_of_n.flow import NAME, search
from asmevo.methods.single_call.prompt import PROMPT_ID as BASELINE_PROMPT_ID
from asmevo.schemas import SearchTrace


def run(llm, seed: int = 0):
    return run_search(search, llm, seed)


def test_draws_twelve_independent_samples_of_the_baseline(evaluator):
    llm = FakeLLM()
    trace, budget = run(llm)
    assert budget.used == 12 and len(trace.candidates) == 13
    baseline = trace.candidates[0]
    assert baseline.id == "7-best_of_n-s0-0" and trace.method == NAME
    for c in trace.candidates[1:]:
        assert c.generation == 1 and c.parent_ids == (baseline.id,)
        assert c.prompt_id == BASELINE_PROMPT_ID
    # No feedback: every request is the unchanged SuperCoder prompt.
    assert all(messages == [{"role": "user", "content": PROBLEM.standard_prompt}]
               for messages, _ in llm.requests)


def test_selects_the_fittest_sample(evaluator):
    trace, _ = run(FakeLLM())
    assert trace.selected.asm == program(12)  # program n is n% faster; 2, 5, 9 are wrong
    assert trace.final == trace.selected.result and not trace.fallback
    assert [c.valid for c in trace.candidates[1:]] == [n not in {2, 5, 9} for n in range(1, 13)]


def test_keeps_the_baseline_when_no_sample_works(monkeypatch):
    monkeypatch.setattr(nodes, "evaluate_asm", FakeEvaluate(wrong=frozenset(range(1, 13))))
    trace, _ = run(FakeLLM())
    assert trace.selected_id == trace.candidates[0].id
    assert trace.final.ok and trace.final.speedup == 1.0


def test_each_distinct_sample_is_evaluated_once(evaluator):
    trace, _ = run(FakeLLM(lambda n: asm_reply(program(n % 3 + 1))))
    assert sorted(evaluator.calls) == [program(1), program(2), program(3)]
    assert len({c.result for c in trace.candidates[1:]}) == 3


def test_trace_round_trips_through_jsonl(evaluator):
    trace, _ = run(FakeLLM())
    assert SearchTrace.from_dict(json.loads(json.dumps(asdict(trace)))) == trace
