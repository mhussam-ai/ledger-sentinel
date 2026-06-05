"""Anthropic (Claude) provider.

The SDK is imported lazily inside the client factory so the system imports and
runs (mock mode, tests, other providers) even when `anthropic` isn't installed.
"""
from __future__ import annotations

from .base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    id = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic  # lazy: optional dependency

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def complete(self, *, model, prompt, system=None, max_tokens=512) -> LLMResponse:
        client = self._get_client()
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = await client.messages.create(**kwargs)
        text = resp.content[0].text if getattr(resp, "content", None) else ""
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=model,
        )

    async def list_models(self):
        client = self._get_client()
        out: list[tuple[str, str]] = []
        async for m in client.models.list(limit=100):
            out.append((m.id, getattr(m, "display_name", None) or m.id))
            if len(out) >= 100:
                break
        return out
