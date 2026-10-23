"""LLM usage accounting and assembly extraction."""

from types import SimpleNamespace

from asmevo.utils.llm.client import reasoning_tokens
from asmevo.utils.llm.parsing import extract_asm


def test_reasoning_tokens_without_details():
    usage = SimpleNamespace(input_tokens=10, output_tokens=4, output_tokens_details=None)
    assert reasoning_tokens(usage, "output_tokens_details") == 0
    assert reasoning_tokens(SimpleNamespace(input_tokens=10), "output_tokens_details") == 0
    empty = SimpleNamespace(completion_tokens_details=SimpleNamespace(reasoning_tokens=None))
    assert reasoning_tokens(empty, "completion_tokens_details") == 0


def test_reasoning_tokens_kept():
    details = SimpleNamespace(reasoning_tokens=7)
    usage = SimpleNamespace(input_tokens=1, output_tokens=8, output_tokens_details=details)
    assert reasoning_tokens(usage, "output_tokens_details") == 7


def test_extract_asm_returns_the_last_block():
    reply = "draft\n```assembly\nret\n```\nfinal\n```assembly\nmovl $1, %eax\nret\n```\n"
    assert extract_asm(reply) == "movl $1, %eax\nret"


def test_extract_asm_rejects_empty_or_missing_block():
    assert extract_asm("no fence here") is None
    assert extract_asm("```assembly\n   \n```") is None
