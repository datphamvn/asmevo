"""Per-(problem, seed) cap on LLM-made candidates, shared by every search method."""

from __future__ import annotations


class BudgetExhausted(Exception):
    """Raised by ``Budget.charge`` once ``max_candidates`` candidates were charged."""


class Budget:
    """Charge once per candidate generation, repairs and failed candidates included."""

    def __init__(self, max_candidates: int = 12) -> None:
        raise NotImplementedError("T3")

    def charge(self) -> None:
        """Count one candidate; raise ``BudgetExhausted`` when none remain."""
        raise NotImplementedError("T3")

    @property
    def remaining(self) -> int:
        raise NotImplementedError("T3")
