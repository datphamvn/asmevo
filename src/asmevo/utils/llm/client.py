"""LLM client interface. One fixed model is used for all search methods."""

from __future__ import annotations

import os
import time
from typing import Protocol

from openai import OpenAI

from asmevo.config import LLMConfig
from asmevo.schemas import LLMCall

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


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

        usage = resp.usage
        call = LLMCall(
            model=self.cfg.model,
            prompt_id=prompt_id,
            response_status=resp.status,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
            latency_s=latency,
        )
        return resp.output_text, call
