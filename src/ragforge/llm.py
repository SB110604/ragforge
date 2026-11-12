"""Minimal chat LLM interface plus an OpenAI-compatible client (OpenAI, Ollama, vLLM, LM Studio, ...)."""

from __future__ import annotations

import os
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class LLMError(Exception):
    pass


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model or os.getenv("RAGFORGE_LLM_MODEL", "")
        if not self.model:
            raise LLMError("no model configured (RAGFORGE_LLM_MODEL)")
        self.base_url = (base_url or os.getenv("RAGFORGE_LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("RAGFORGE_LLM_API_KEY", "")
        self.temperature, self._client = temperature, client or httpx.Client(timeout=120)

    def complete(self, messages: list[dict[str, str]]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": messages, "temperature": self.temperature},
            )
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM request failed: {e}") from e
