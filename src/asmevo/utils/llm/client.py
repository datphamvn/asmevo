"""LLM client interface. One fixed model is used for all search methods."""

from __future__ import annotations

import os
import time
from typing import Protocol

from openai import OpenAI

from asmevo.config import LLMConfig
from asmevo.schemas import LLMCall

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


def to_llm_call(model: str, prompt_id: str, response_status: str | None, usage: object,
                latency_s: float) -> LLMCall:
    """Build an ``LLMCall`` from a Responses API usage object.

    Some OpenAI-compatible endpoints omit ``output_tokens_details``. Those calls still count;
    reasoning tokens are recorded as zero.
    """
    details = getattr(usage, "output_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) if details is not None else 0
    return LLMCall(
        model=model,
        prompt_id=prompt_id,
        response_status=response_status,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=reasoning or 0,
        latency_s=latency_s,
    )


class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        """Return the model's text reply and a usage record (tokens, latency)."""
        ...


class OpenAICompatibleClient:
    """Client for an OpenAI Responses API endpoint, configured by ``LLMConfig``.

    Transient errors (connection, 408/409/429/5xx) are retried by the SDK with exponential
    backoff; anything else raises. Thread-safe, so all workers share one instance.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        start = time.perf_counter()
        resp = self.client.responses.create(
            model=self.cfg.model,
            input=messages,
            max_output_tokens=self.cfg.max_output_tokens,
            reasoning={"mode": self.cfg.reasoning_mode, "effort": self.cfg.reasoning_effort},
        )
        latency = time.perf_counter() - start
        if resp.usage is None:
            raise RuntimeError(f"response {resp.id} has no usage; cannot account tokens")

        call = to_llm_call(self.cfg.model, prompt_id, resp.status, resp.usage, latency)
        return resp.output_text, call
