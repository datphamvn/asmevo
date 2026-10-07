import json
from dataclasses import asdict
from itertools import pairwise

from fakes import PROBLEM, FakeEvaluate, FakeLLM, asm_reply, program
from fakes import run as run_search

from asmevo.methods import nodes
from asmevo.methods.prompt import MUTATE_PROMPT_ID
from asmevo.methods.refine.flow import NAME, search
from asmevo.methods.single_call.prompt import PROMPT_ID as BASELINE_PROMPT_ID
from asmevo.schemas import SearchTrace


def run(llm, seed: int = 0):
    return run_search(search, llm, seed)


def test_twelve_steps_form_one_chain(evaluator):
    trace, budget = run(FakeLLM())
    assert budget.used == 12 and len(trace.candidates) == 13
    assert trace.candidates[0].id == "7-refine-s0-0" and trace.method == NAME
    for step, (parent, child) in enumerate(pairwise(trace.candidates), 1):
        assert child.generation == step and child.parent_ids == (parent.id,)


def test_each_step_sees_its_parent_and_feedback(evaluator):
    llm = FakeLLM()
    trace, _ = run(llm)
    (first, first_id), *rest = llm.requests
    assert first == [{"role": "user", "content": PROBLEM.standard_prompt}]
    assert first_id == BASELINE_PROMPT_ID
    for parent, (messages, prompt_id) in zip(trace.candidates[1:-1], rest, strict=True):
        assert prompt_id == MUTATE_PROMPT_ID
        assert messages[0]["content"] == PROBLEM.standard_prompt
        assert messages[1] == {"role": "assistant", "content": parent.llm_calls[0].reply}
        assert parent.feedback in messages[2]["content"]


def test_a_broken_candidate_is_repaired_from_its_error(evaluator):
    llm = FakeLLM()
    trace, _ = run(llm)
    broken = trace.candidates[2]  # program 2 prints the wrong output
    assert not broken.valid
    messages, _ = llm.requests[2]  # the step after it
    assert messages[1]["content"] == broken.llm_calls[0].reply
    assert "wrong output" in messages[2]["content"] and "Test 1 stdin" in messages[2]["content"]
    assert trace.candidates[3].parent_ids == (broken.id,)


def test_selects_the_fittest_step_even_if_the_trajectory_gets_worse(evaluator):
    trace, _ = run(FakeLLM(lambda n: asm_reply(program(13 - n))))  # each step is slower
    assert trace.selected_id == trace.candidates[1].id
    assert trace.final == trace.candidates[1].result and not trace.fallback


def test_keeps_the_baseline_when_every_step_fails(monkeypatch):
    monkeypatch.setattr(nodes, "evaluate_asm", FakeEvaluate(wrong=frozenset(range(1, 13))))
    trace, _ = run(FakeLLM())
    assert trace.selected_id == trace.candidates[0].id
    assert trace.final.ok and trace.final.speedup == 1.0


def test_trace_round_trips_through_jsonl(evaluator):
    trace, _ = run(FakeLLM())
    assert SearchTrace.from_dict(json.loads(json.dumps(asdict(trace)))) == trace
