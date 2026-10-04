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
    # The exact request and reply, so every candidate can be traced back to its prompt.
    # Defaults keep runs recorded before these fields were added loadable.
    messages: tuple[dict[str, str], ...] = ()
    reply: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LLMCall:
        return cls(**{**d, "messages": tuple(d.get("messages", ()))})


@dataclass(frozen=True)
class Generation:
    """One method's candidate assembly for one problem; replies live in ``llm_calls``."""

    problem_id: int
    method: str
    asm: str | None
    llm_calls: tuple[LLMCall, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Generation:
        d = {k: v for k, v in d.items() if k != "reply"}  # runs before replies moved to LLMCall
        return cls(**{**d, "llm_calls": tuple(LLMCall.from_dict(c) for c in d["llm_calls"])})


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
