"""LLM client interface. One fixed model is used for all search methods."""

from __future__ import annotations

import os
from typing import Protocol

from asmevo.config import LLMConfig
from asmevo.schemas import LLMCall

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": ...}


class LLMClient(Protocol):
    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        """Return the model's text reply and a usage record (tokens, cost, latency)."""
        ...


class OpenAICompatibleClient:
    """Client for any OpenAI-compatible endpoint (OpenAI, vLLM, OpenRouter, ...).

    Reads ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and ``MODEL`` from the environment.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.model = os.environ.get("MODEL", cfg.model)
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.api_key = os.environ.get("OPENAI_API_KEY")

    def complete(self, messages: list[Message], *, prompt_id: str) -> tuple[str, LLMCall]:
        """Call chat completions with ``cfg.temperature`` and ``cfg.max_output_tokens``.

        Must fill ``LLMCall`` with real token usage and cost computed from
        ``cfg.cost_per_1k_input`` / ``cfg.cost_per_1k_output``, and retry transient errors.
        """
        # TODO(Implement the LLM client)
        raise NotImplementedError
