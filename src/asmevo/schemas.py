"""Immutable records persisted as JSONL by the generation and evaluation stages."""

from __future__ import annotations

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


@dataclass(frozen=True)
class EvalResult:
    problem_id: int
    status: Status
    speedup: float = 1.0  # the protocol scores every non-OK candidate as the -O3 baseline
    baseline_mean_s: float | None = None
    candidate_mean_s: float | None = None
    failed_test: int | None = None
    dropped_tests: tuple[int, ...] = ()  # the -O3 baseline exceeds run_timeout_s on these
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalResult:
        return cls(**{**d, "status": Status(d["status"]),
                      "dropped_tests": tuple(d["dropped_tests"])})


@dataclass(frozen=True)
class Generation:
    """One method's candidate assembly for one problem."""

    problem_id: int
    method: str
    asm: str | None
    reply: str
    llm_calls: tuple[LLMCall, ...]
    rounds: tuple[EvalResult, ...] = ()  # per-round feedback of methods that evaluate in the loop
    round_asm: tuple[str | None, ...] = ()  # the candidate of each round
    round_parent: tuple[int, ...] = ()  # round each one was derived from; -1 is the -O3 baseline

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Generation:
        return cls(**{
            **d,
            "llm_calls": tuple(LLMCall(**c) for c in d["llm_calls"]),
            "rounds": tuple(EvalResult.from_dict(r) for r in d.get("rounds", ())),
            "round_asm": tuple(d.get("round_asm", ())),
            "round_parent": tuple(d.get("round_parent", ())),
        })
