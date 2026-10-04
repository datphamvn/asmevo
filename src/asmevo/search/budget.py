"""Per-(problem, seed) cap on LLM-made candidates, shared by every search method.

RQ2 compares methods under the same budget, so the cap is enforced here instead of being
trusted to each method: the experiment runner hands a method ``budget.guard(llm)``, which
refuses any model call that was not charged first and records every call it lets through.
``check_trace`` then confirms that the trace accounts for all of them, so its token and cost
totals are what the run really spent.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from asmevo.schemas import LLMCall, SearchTrace

if TYPE_CHECKING:
    from asmevo.utils.llm.client import LLMClient, Message


class BudgetExhausted(Exception):
    """Raised by ``Budget.charge`` once ``max_candidates`` candidates were charged."""


class Budget:
    """Charge once per candidate generation, repairs and failed candidates included.

    Thread-safe, so a method may generate several candidates in parallel.
    """

    def __init__(self, max_candidates: int = 12) -> None:
        if max_candidates < 0:
            raise ValueError(f"max_candidates must be >= 0, got {max_candidates}")
        self.max_candidates = max_candidates
        self._used = 0
        self._calls_started = 0
        self._calls: list[LLMCall] = []
        self._lock = threading.Lock()

    def charge(self) -> None:
        """Count one candidate; raise ``BudgetExhausted`` when none remain."""
        with self._lock:
            if self._used >= self.max_candidates:
                raise BudgetExhausted(f"all {self.max_candidates} candidates are used")
            self._used += 1

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return self.max_candidates - self._used

    @property
    def calls(self) -> tuple[LLMCall, ...]:
        """Every call made through ``guard``, in completion order."""
        with self._lock:
            return tuple(self._calls)

    def guard(self, llm: LLMClient) -> LLMClient:
        """``llm`` limited to one call per ``charge()``; pass this to the search method."""
        return _GuardedClient(llm, self)

    def check_trace(self, trace: SearchTrace) -> None:
        """Raise ``ValueError`` unless ``trace`` holds each call made through ``guard``
        exactly once, so its token and cost totals are what the run spent."""
        unseen = list(self.calls)
        for call in trace.llm_calls:
            if call not in unseen:
                raise ValueError(f"trace has a call to {call.prompt_id!r} that the budget did "
                                 "not record: made without guard, or listed twice")
            unseen.remove(call)
        if unseen:
            raise ValueError(f"trace misses {len(unseen)} of the {len(self.calls)} LLM calls "
                             "made; its token and cost totals would be too low")

    def _start_call(self) -> None:
        with self._lock:
            if self._calls_started >= self._used:
                raise RuntimeError("LLM call without a matching budget.charge(); charge "
                                   "before each candidate, repairs included")
            self._calls_started += 1

    def _record(self, call: LLMCall) -> None:
        with self._lock:
            self._calls.append(call)


class _GuardedClient:
    """``LLMClient`` that spends one charge of ``budget`` per call; a failed call keeps its
    charge, since the attempt was made."""

    def __init__(self, llm: LLMClient, budget: Budget) -> None:
        self._llm = llm
        self._budget = budget

    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        self._budget._start_call()
        reply, call = self._llm.complete(messages, prompt_id=prompt_id)
        self._budget._record(call)
        return reply, call
