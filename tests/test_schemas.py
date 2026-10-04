import json
import math
from dataclasses import asdict

from asmevo.schemas import Candidate, EvalResult, LLMCall, SearchTrace, Status, candidate_id


def roundtrip(obj, cls):
    return cls.from_dict(json.loads(json.dumps(asdict(obj))))


def make_trace() -> SearchTrace:
    call = LLMCall(model="m", prompt_id="mutate-v1", response_status="completed",
                   input_tokens=10, output_tokens=20, reasoning_tokens=0, cost_usd=0.5,
                   latency_s=1.0, messages=({"role": "user", "content": "hi"},), reply="r")
    baseline = Candidate(
        id=candidate_id(7, "evolution", 0, 0), problem_id=7, method="evolution", seed=0,
        generation=0, parent_ids=(), prompt_id=None, asm="ret", asm_hash="h0", result=None,
        fitness=1.0, feedback="", llm_calls=(),
    )
    broken = Candidate(
        id=candidate_id(7, "evolution", 0, 1), problem_id=7, method="evolution", seed=0,
        generation=1, parent_ids=(baseline.id,), prompt_id="mutate-v1", asm="bad",
        asm_hash="h1",
        result=EvalResult(problem_id=7, status=Status.BUILD_FAIL, dropped_tests=(2,),
                          detail="error"),
        fitness=-math.inf, feedback="assembler error", llm_calls=(call,),
    )
    return SearchTrace(problem_id=7, method="evolution", seed=0,
                       candidates=(baseline, broken), selected_id=baseline.id,
                       final=EvalResult(problem_id=7, status=Status.OK), fallback=False)


def test_candidate_id():
    assert candidate_id(7, "evolution", 0, 3) == "7-evolution-s0-3"


def test_eval_result_roundtrip():
    r = EvalResult(problem_id=1, status=Status.TIMEOUT, failed_test=0, dropped_tests=(1, 2))
    back = roundtrip(r, EvalResult)
    assert back == r
    assert back.status is Status.TIMEOUT


def test_search_trace_roundtrip():
    trace = make_trace()
    assert roundtrip(trace, SearchTrace) == trace


def test_derived_properties():
    trace = make_trace()
    baseline, broken = trace.candidates
    assert baseline.valid and not broken.valid
    assert trace.selected is baseline
    assert trace.total_cost_usd == 0.5
    assert trace.total_output_tokens == 20

