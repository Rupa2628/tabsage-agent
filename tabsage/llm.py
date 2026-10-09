"""LLM clients.

The agent only needs one method: ``complete(messages) -> str``. That keeps the
agent independent of any vendor SDK and lets the tests swap in a scripted model.
"""
from __future__ import annotations

import json
import os
import time
from typing import Protocol


class LLM(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class OpenAICompatibleLLM:
    """Any provider that speaks the OpenAI chat-completions API.

    Works with Groq, Google Gemini (OpenAI-compatible endpoint), OpenRouter,
    Ollama (local) and OpenAI itself. Configure through environment variables:
    TABSAGE_BASE_URL, TABSAGE_API_KEY, TABSAGE_MODEL.
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0, max_retries: int = 3):
        from openai import OpenAI  # imported lazily so tests don't need network config

        self.model = model or os.getenv("TABSAGE_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_retries = max_retries
        key = api_key or os.getenv("TABSAGE_API_KEY")
        if not key:
            raise RuntimeError("TABSAGE_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = OpenAI(
            base_url=base_url or os.getenv("TABSAGE_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=key,
        )

    def complete(self, messages: list[dict]) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception:  # rate limits, timeouts, 5xx
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, ...
        return ""


class ScriptedLLM:
    """Returns pre-written responses in order. Used for deterministic tests."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        r = self.responses.pop(0)
        return r if isinstance(r, str) else json.dumps(r)
