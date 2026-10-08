"""LLM usage accounting and assembly extraction."""

from types import SimpleNamespace

from asmevo.utils.llm.client import to_llm_call
from asmevo.utils.llm.parsing import extract_asm


def test_to_llm_call_without_reasoning_details():
    usage = SimpleNamespace(input_tokens=10, output_tokens=4, output_tokens_details=None)
    call = to_llm_call("gpt-test", "policy_expand", "completed", usage, 1.5)
    assert call.input_tokens == 10
    assert call.output_tokens == 4
    assert call.reasoning_tokens == 0
    assert call.prompt_id == "policy_expand"
    assert call.latency_s == 1.5


def test_to_llm_call_keeps_reasoning_tokens():
    details = SimpleNamespace(reasoning_tokens=7)
    usage = SimpleNamespace(input_tokens=1, output_tokens=8, output_tokens_details=details)
    call = to_llm_call("gpt-test", "supercoder_standard", "completed", usage, 0.2)
    assert call.reasoning_tokens == 7


def test_extract_asm_returns_the_last_block():
    reply = "draft\n```assembly\nret\n```\nfinal\n```assembly\nmovl $1, %eax\nret\n```\n"
    assert extract_asm(reply) == "movl $1, %eax\nret"


def test_extract_asm_rejects_empty_or_missing_block():
    assert extract_asm("no fence here") is None
    assert extract_asm("```assembly\n   \n```") is None
