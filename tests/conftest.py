import pytest
from fakes import FakeEvaluate

from asmevo.methods import nodes


@pytest.fixture
def evaluator(monkeypatch):
    """``FakeEvaluate`` in place of ``evaluate_asm``; programs 2, 5 and 9 print wrong output."""
    fake = FakeEvaluate(wrong=frozenset({2, 5, 9}))
    monkeypatch.setattr(nodes, "evaluate_asm", fake)
    return fake
