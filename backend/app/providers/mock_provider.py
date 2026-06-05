"""The deterministic mock provider — a first-class peer of the live vendors.

It implements `extract_receipt()` directly from the receipt text (no network,
no key), which is what makes the demo, the tests, and the eval gates fully
deterministic. It is also the universal degradation target: when any live
provider call fails, the worker falls back to this same parse (F6).
"""
from __future__ import annotations

from .base import LLMProvider, LLMResponse
from .parsing import parse_receipt_text


class MockProvider(LLMProvider):
    id = "mock"
    is_mock = True

    def __init__(self, api_key: str = "") -> None:  # signature parity with live providers
        pass

    async def complete(self, *, model, prompt, system=None, max_tokens=512) -> LLMResponse:
        raise RuntimeError("mock provider has no live completion; use extract_receipt()")

    async def extract_receipt(self, text, source_type, *, fast_model, deep_model):
        return parse_receipt_text(text), 0, 0, "mock"

    async def list_models(self):
        return [("mock", "Deterministic mock")]

    def is_transient(self, exc: Exception) -> bool:  # nothing to retry in mock mode
        return False
