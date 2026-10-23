"""Immutable records persisted as JSONL by the generation and evaluation stages.

``Candidate.fitness`` is ``-inf`` for invalid candidates; ``json.dumps`` writes it as
``-Infinity``, which is not standard JSON but round-trips through ``json.loads``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    OK = "ok"
    NO_ASM = "no_asm"
    BUILD_FAIL = "build_fail"
    WRONG_OUTPUT = "wrong_output"
    CRASH = "crash"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class LLMCall:
    model: str
    prompt_id: str
    response_status: str | None  # "incomplete" means max_output_tokens cut the reply
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_s: float
    # The exact request and reply, so every candidate can be traced back to its prompt.
    # Defaults keep runs recorded before these fields were added loadable.
    messages: tuple[dict[str, str], ...] = ()
    reply: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMCall:
        return cls(**{**d, "messages": tuple(d.get("messages", ()))})


@dataclass(frozen=True)
class EvalResult:
    problem_id: int
    status: Status
    # Baseline over candidate time by eval.timing_stat (median or mean); the protocol scores
    # every non-OK candidate as the -O3 baseline. Times are of one pass over the tests: the sum
    # over tests of each test's median (or mean) run; stddev is the spread of such a pass
    # (eval.runner.Timing).
    speedup: float = 1.0
    baseline_mean_s: float | None = None
    candidate_mean_s: float | None = None
    baseline_median_s: float | None = None
    candidate_median_s: float | None = None
    baseline_stddev_s: float | None = None
    candidate_stddev_s: float | None = None
    failed_test: int | None = None
    dropped_tests: tuple[int, ...] = ()  # the -O3 baseline exceeds run_timeout_s on these
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        return cls(**{**d, "status": Status(d["status"]),
                      "dropped_tests": tuple(d.get("dropped_tests", ()))})


@dataclass(frozen=True)
class Generation:
    """One method's candidate assembly for one problem; replies live in ``llm_calls``."""

    problem_id: int
    method: str
    asm: str | None
    llm_calls: tuple[LLMCall, ...]
    rounds: tuple[EvalResult, ...] = ()  # per-round feedback of methods that evaluate in the loop
    round_asm: tuple[str | None, ...] = ()  # the candidate of each round
    round_parent: tuple[int, ...] = ()  # round each one was derived from; -1 is the -O3 baseline

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Generation:
        d = {k: v for k, v in d.items() if k != "reply"}  # runs before replies moved to LLMCall
        return cls(**{
            **d,
            "llm_calls": tuple(LLMCall.from_dict(c) for c in d["llm_calls"]),
            "rounds": tuple(EvalResult.from_dict(r) for r in d.get("rounds", ())),
            "round_asm": tuple(d.get("round_asm", ())),
            "round_parent": tuple(d.get("round_parent", ())),
        })


def candidate_id(problem_id: int, method: str, seed: int, n: int) -> str:
    """Id of the ``n``-th candidate a search run made; ``n = 0`` is the -O3 baseline."""
    return f"{problem_id}-{method}-s{seed}-{n}"


@dataclass(frozen=True)
class Candidate:
    """One node of a search: the -O3 baseline (generation 0) or one LLM-made program."""

    id: str  # candidate_id(...)
    problem_id: int
    method: str
    seed: int
    generation: int  # 0 = -O3 baseline
    parent_ids: tuple[str, ...]
    prompt_id: str | None  # None for the baseline
    asm: str | None  # None when the reply had no ```assembly block
    asm_hash: str | None
    result: EvalResult | None  # on search_tests only; None for the baseline
    fitness: float  # T_baseline / T_candidate; -inf if invalid
    feedback: str  # goes into the next prompt that mutates this candidate
    llm_calls: tuple[LLMCall, ...]

    @property
    def valid(self) -> bool:
        """Selection may pick it as a parent or survivor."""
        return self.fitness != -math.inf

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        return cls(**{
            **d,
            "parent_ids": tuple(d["parent_ids"]),
            "result": EvalResult.from_dict(d["result"]) if d["result"] is not None else None,
            "llm_calls": tuple(LLMCall.from_dict(c) for c in d["llm_calls"]),
        })


@dataclass(frozen=True)
class SearchTrace:
    """Everything one search method did on one (problem, seed)."""

    problem_id: int
    method: str
    seed: int
    candidates: tuple[Candidate, ...]  # every candidate made, failed ones included
    selected_id: str  # best candidate on search_tests
    final: EvalResult  # after the held-out final check
    fallback: bool  # the final check rejected the selection and scored the baseline

    @property
    def selected(self) -> Candidate:
        return next(c for c in self.candidates if c.id == self.selected_id)

    @property
    def llm_calls(self) -> tuple[LLMCall, ...]:
        return tuple(call for c in self.candidates for call in c.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchTrace:
        return cls(**{
            **d,
            "candidates": tuple(Candidate.from_dict(c) for c in d["candidates"]),
            "final": EvalResult.from_dict(d["final"]),
        })
