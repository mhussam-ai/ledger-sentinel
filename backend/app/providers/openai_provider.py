"""OpenAI (GPT) provider.

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=...)
    await client.chat.completions.create(model=, messages=, max_completion_tokens=)

Newer models expect `max_completion_tokens`; a few older models/proxies only
accept the legacy `max_tokens`. We send the modern param and transparently
retry once with the legacy name if (and only if) the model rejects it. The SDK
is imported lazily so the system runs without `openai` installed.
"""
from __future__ import annotations

from .base import LLMProvider, LLMResponse


def _is_token_param_error(exc: Exception) -> bool:
    param = getattr(exc, "param", None)
    if param in ("max_completion_tokens", "max_tokens"):
        return True
    msg = str(exc).lower()
    return "max_completion_tokens" in msg or "max_tokens" in msg


# Heuristic filter so the dropdown shows chat-capable models, not embeddings/audio/etc.
_CHAT_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")
_NON_CHAT = (
    "embedding", "audio", "realtime", "transcribe", "tts", "whisper",
    "image", "dall-e", "search", "moderation", "codex",
)


class OpenAIProvider(LLMProvider):
    id = "openai"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI  # lazy: optional dependency

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def complete(self, *, model, prompt, system=None, max_tokens=512) -> LLMResponse:
        client = self._get_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            if _is_token_param_error(exc):
                resp = await client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens
                )
            else:
                raise

        text = resp.choices[0].message.content if resp.choices else ""
        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        tout = int(getattr(usage, "completion_tokens", 0) or 0)
        return LLMResponse(text=text or "", input_tokens=tin, output_tokens=tout, model=model)

    async def list_models(self):
        client = self._get_client()
        out: list[tuple[str, str]] = []
        async for m in client.models.list():
            mid = m.id
            low = mid.lower()
            if not low.startswith(_CHAT_PREFIXES) or any(x in low for x in _NON_CHAT):
                continue
            out.append((mid, mid))
            if len(out) >= 100:
                break
        return sorted(out)
