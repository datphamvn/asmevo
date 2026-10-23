"""LLM client interface. One fixed model is used for all search methods.

``llm.provider`` picks the wire format; both go through the OpenAI SDK:

- ``openai``: OpenAI Responses API.
- ``fptcloud``: FPT AI Marketplace (https://mkp-api.fptcloud.com/v1), a third-party
  endpoint that speaks the OpenAI Chat Completions format.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from openai import OpenAI

from asmevo.config import LLMConfig
from asmevo.schemas import LLMCall

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


def reasoning_tokens(usage: object, details_field: str) -> int:
    """Reasoning tokens in ``usage.<details_field>``.

    Some OpenAI-compatible endpoints omit the details. Those calls still count; reasoning
    tokens are recorded as zero.
    """
    details = getattr(usage, details_field, None)
    return getattr(details, "reasoning_tokens", 0) or 0


class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        """Return the model's text reply and a record of the call (prompt, reply, tokens,
        cost, latency)."""
        ...


class _OpenAISDKClient:
    """Shared setup for providers reached through the OpenAI SDK.

    Transient errors (connection, 408/409/429/5xx) are retried by the SDK with exponential
    backoff; anything else raises. Thread-safe, so all workers share one instance.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        if cfg.cost_per_1k_input is None or cfg.cost_per_1k_output is None:
            raise ValueError(
                "set llm.cost_per_1k_input and llm.cost_per_1k_output (USD per 1k tokens; "
                "0.0 if the endpoint is free) so reported costs are real"
            )
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise ValueError(
                f"set {cfg.api_key_env} in the environment or .env "
                f"(llm.api_key_env) for provider {cfg.provider!r}"
            )
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    def _record(
        self,
        messages: list[Message],
        reply: str,
        *,
        prompt_id: str,
        status: str | None,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        latency_s: float,
    ) -> LLMCall:
        cost = (
            input_tokens * self.cfg.cost_per_1k_input + output_tokens * self.cfg.cost_per_1k_output
        ) / 1000
        return LLMCall(
            model=self.cfg.model,
            prompt_id=prompt_id,
            response_status=status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=cost,
            latency_s=latency_s,
            messages=tuple(dict(m) for m in messages),
            reply=reply,
        )


class OpenAICompatibleClient(_OpenAISDKClient):
    """Client for an OpenAI Responses API endpoint, configured by ``LLMConfig``."""

    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        start = time.perf_counter()
        reasoning = {
            k: v
            for k, v in (("mode", self.cfg.reasoning_mode), ("effort", self.cfg.reasoning_effort))
            if v is not None
        }
        resp = self.client.responses.create(
            model=self.cfg.model,
            input=messages,
            max_output_tokens=self.cfg.max_output_tokens,
            **({"reasoning": reasoning} if reasoning else {}),
        )
        latency = time.perf_counter() - start
        if resp.usage is None:
            raise RuntimeError(f"response {resp.id} has no usage; cannot account tokens")

        usage = resp.usage
        reply = resp.output_text
        call = self._record(
            messages,
            reply,
            prompt_id=prompt_id,
            status=resp.status,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=reasoning_tokens(usage, "output_tokens_details"),
            latency_s=latency,
        )
        return reply, call


class ChatCompletionsClient(_OpenAISDKClient):
    """Client for a third-party endpoint in the OpenAI Chat Completions format.

    ``reasoning_mode`` has no Chat Completions equivalent and is ignored;
    ``reasoning_effort`` is sent only when set, since many hosted models reject it.
    """

    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        start = time.perf_counter()
        extra = {}
        if self.cfg.reasoning_effort is not None:
            extra["reasoning_effort"] = self.cfg.reasoning_effort
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            max_tokens=self.cfg.max_output_tokens,
            **extra,
        )
        latency = time.perf_counter() - start
        if resp.usage is None:
            raise RuntimeError(f"response {resp.id} has no usage; cannot account tokens")
        if not resp.choices:
            raise RuntimeError(f"response {resp.id} has no choices")

        choice = resp.choices[0]
        usage = resp.usage
        reply = choice.message.content or ""
        call = self._record(
            messages,
            reply,
            prompt_id=prompt_id,
            # Same vocabulary as the Responses API: "incomplete" = cut by max_output_tokens.
            status="incomplete" if choice.finish_reason == "length" else "completed",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            reasoning_tokens=reasoning_tokens(usage, "completion_tokens_details"),
            latency_s=latency,
        )
        return reply, call


PROVIDERS: dict[str, type[_OpenAISDKClient]] = {
    "openai": OpenAICompatibleClient,
    "fptcloud": ChatCompletionsClient,
}


def make_client(cfg: LLMConfig) -> LLMClient:
    """Build the client for ``cfg.provider``."""
    try:
        cls = PROVIDERS[cfg.provider]
    except KeyError:
        raise ValueError(
            f"unknown llm.provider {cfg.provider!r}; choose one of {sorted(PROVIDERS)}"
        ) from None
    return cls(cfg)
