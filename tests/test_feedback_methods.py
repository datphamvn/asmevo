"""refinement_loop, aide, eoh, and asmgraft with a scripted fake LLM and a fake evaluator (no API
key, any host): each spends its budget, records every round with its lineage, and keeps the
fastest correct candidate."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

import asmevo.methods.nodes as shared_nodes
from asmevo.config import load_config
from asmevo.data.dataset import Problem
from asmevo.data.dataset import TestCase as Case
from asmevo.methods.aide import flow as aide
from asmevo.methods.asmgraft import flow as asmgraft
from asmevo.methods.eoh import flow as eoh
from asmevo.methods.nodes import best_round
from asmevo.methods.refinement_loop import flow as refinement_loop
from asmevo.schemas import EvalResult, Generation, LLMCall, Status

PROBLEM = Problem(
    id=3,
    c_code="int main(void) { return 0; }",
    baseline_asm="main:\n\txorl\t%eax, %eax\n\tret",
    tests=(Case("1\n", "1\n"), Case("2\n", "2\n")),
    baseline_binary=b"",
    standard_prompt="SUPERCODER PROMPT\nOptimized Assembly Code:\n",
)
CALLS = 24
FAILS = {1: "BUILD_FAIL", 4: "WRONG", 7: "CRASH"}  # call index -> how its program fails


class ScriptedLLM:
    """Call ``i`` answers with a braced thought and program ``p{i}``, ``(1 + i / 100)``x as fast
    as -O3 unless ``FAILS`` says otherwise; records every request."""

    def __init__(self) -> None:
        self.requests: list[tuple[list[dict[str, str]], str]] = []

    def complete(self, messages, *, prompt_id):
        i = len(self.requests)
        self.requests.append(([dict(m) for m in messages], prompt_id))
        asm = f"{FAILS.get(i, 'OK')} p{i} speed={1 + i / 100}"
        reply = f"{{idea {i}}}\n```assembly\n{asm}\n```"
        call = LLMCall(model="fake", prompt_id=prompt_id, response_status="completed",
                       input_tokens=1, output_tokens=1, reasoning_tokens=0, cost_usd=0.0,
                       latency_s=0.0, messages=tuple(dict(m) for m in messages), reply=reply)
        return reply, call


def fake_evaluate(problem: Problem, gen: Generation, cfg) -> EvalResult:
    status, _, speed = gen.asm.split()
    match status:
        case "BUILD_FAIL":
            return EvalResult(problem.id, Status.BUILD_FAIL, detail="Error: bad instruction")
        case "WRONG":
            return EvalResult(problem.id, Status.WRONG_OUTPUT, failed_test=1, detail="got '9'")
        case "CRASH":
            return EvalResult(problem.id, Status.CRASH, failed_test=0, detail="killed by SIGSEGV")
    speedup = float(speed.removeprefix("speed="))
    return EvalResult(problem.id, Status.OK, speedup=speedup, baseline_mean_s=1.0,
                      candidate_mean_s=1.0 / speedup)


BASE = load_config()
CFG = replace(
    BASE,
    refinement_loop=replace(BASE.refinement_loop, rounds=CALLS),
    aide=replace(BASE.aide, max_calls=CALLS),
    eoh=replace(BASE.eoh, max_calls=CALLS),
    asmgraft=replace(BASE.asmgraft, max_calls=CALLS),
)
METHODS = {m.NAME: m.generate for m in (refinement_loop, aide, eoh, asmgraft)}


@pytest.fixture(autouse=True)
def evaluator(monkeypatch):
    monkeypatch.setattr(shared_nodes, "evaluate", fake_evaluate)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_spends_the_budget_and_keeps_the_fastest_correct(name):
    llm = ScriptedLLM()
    gen = METHODS[name](PROBLEM, llm, CFG)

    assert gen.method == name
    assert len(llm.requests) == len(gen.llm_calls) == len(gen.rounds) == CALLS
    assert len(gen.round_asm) == len(gen.round_parent) == CALLS
    assert [c.prompt_id for c in gen.llm_calls] == [p for _, p in llm.requests]
    for i, parent in enumerate(gen.round_parent):
        assert -1 <= parent < i
    best = best_round(list(gen.rounds))
    assert gen.rounds[best].ok and gen.asm == gen.round_asm[best]
    assert gen.rounds[best].speedup == max(r.speedup for r in gen.rounds if r.ok)
    assert Generation.from_dict(json.loads(json.dumps(asdict(gen)))) == gen


def test_refinement_loop_keeps_one_conversation():
    llm = ScriptedLLM()
    gen = refinement_loop.generate(PROBLEM, llm, CFG)
    for i, (messages, _) in enumerate(llm.requests):
        assert len(messages) == 1 + 2 * i
        if i:
            assert messages[-2]["content"] == gen.llm_calls[i - 1].reply
    assert "assemble or link" in llm.requests[2][0][-1]["content"]  # feedback on call 1
    assert gen.round_parent == (-1, *range(CALLS - 1))


def test_asmgraft_uses_every_kind_of_worker():
    llm = ScriptedLLM()
    asmgraft.generate(PROBLEM, llm, CFG)
    ids = {p for _, p in llm.requests}
    assert "asmgraft_repair" in ids  # call 1 is broken, so call 2 repairs it
    for kind in ("draft", "improve", "branch", "graft"):
        assert any(i.startswith(f"asmgraft_{kind}") for i in ids), kind


def test_eoh_seeds_then_evolves():
    llm = ScriptedLLM()
    eoh.generate(PROBLEM, llm, CFG)
    ops = [p for _, p in llm.requests]
    n = CFG.eoh.pop_size  # 2n i1 seeds, then n offspring per operator in turn
    schedule = ["i1"] * 2 * n + [op for op in CFG.eoh.operators for _ in range(n)]
    assert ops == [f"eoh_{op}" for op in schedule[:CALLS]]


