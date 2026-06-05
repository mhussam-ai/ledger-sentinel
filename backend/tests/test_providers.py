"""The provider abstraction + runtime control plane.

These prove the plug-and-play contract without any network: the catalog is
internally consistent, the shared self-consistency guardrail works for *any*
provider (exercised via a fake), key-less live providers degrade to mock, and —
critically — secrets never leak out of the config snapshot.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from app.observability import estimate_cost
from app.providers import get_provider, reset_provider_cache
from app.providers.base import LLMProvider, LLMResponse
from app.providers.catalog import PROVIDER_INFO, get_model_pricing
from app.runtime import ConfigError, get_runtime, reset_runtime, update_runtime


# ── A fake live provider: proves the shared extract path is vendor-agnostic ──
class FakeProvider(LLMProvider):
    id = "fake"

    def __init__(self, first: str, second: str):
        self._first, self._second = first, second
        self.calls = 0

    async def complete(self, *, model, prompt, system=None, max_tokens=512) -> LLMResponse:
        self.calls += 1
        text = self._first if self.calls == 1 else self._second
        return LLMResponse(text=text, input_tokens=11, output_tokens=7, model=model)


# ── Catalog integrity ───────────────────────────────────────────────────────────
def test_catalog_defaults_are_real_models():
    for info in PROVIDER_INFO.values():
        ids = info.model_ids()
        assert info.default_fast in ids, f"{info.id} fast default not in its model list"
        assert info.default_deep in ids, f"{info.id} deep default not in its model list"


def test_pricing_known_and_unknown():
    assert get_model_pricing("claude-opus-4-8") == (15.0, 75.0)
    assert get_model_pricing("mock") == (0.0, 0.0)
    assert get_model_pricing("deterministic") == (0.0, 0.0)
    assert get_model_pricing("some-model-we-never-heard-of") == (0.0, 0.0)
    # cost meter spans providers
    assert estimate_cost("gpt-4o", 1_000_000, 0) == 2.5
    assert estimate_cost("gemini-2.5-flash", 0, 1_000_000) == 2.5
    # family fallback: dated/preview/"-latest" Gemini ids the live API returns
    # still price by family instead of silently reading $0.
    assert get_model_pricing("gemini-3.1-pro-preview-customtools") == (2.0, 12.0)
    assert get_model_pricing("gemini-flash-latest") == (0.5, 3.0)
    assert get_model_pricing("gemini-3.5-flash-preview-11-2026") == (1.5, 9.0)


# ── The shared self-consistency guardrail (F1), provider-agnostic ────────────
async def test_extract_receipt_agreement_is_high_confidence():
    p = FakeProvider('{"merchant": "STELLAR", "amount": 1200.00, "date": "2026-05-30"}', "1200.00")
    parsed, tin, tout, model = await p.extract_receipt("ignored", "receipt", fast_model="m", deep_model="d")
    assert parsed["amount"] == Decimal("1200.00")
    assert parsed["confidence"] == 0.97        # two reads agreed
    assert parsed["merchant"] == "STELLAR"
    assert (tin, tout) == (22, 14)             # summed across the two calls
    assert model == "m"


async def test_extract_receipt_disagreement_collapses_confidence():
    # The classic silent digit misread: 450 vs 480 → must NOT look confident.
    p = FakeProvider('{"merchant": "CAFE", "amount": 450.00, "date": "2026-05-30"}', "480.00")
    parsed, *_ = await p.extract_receipt("ignored", "receipt", fast_model="m", deep_model="d")
    assert parsed["confidence"] == 0.55


# ── Runtime control plane ──────────────────────────────────────────────────
def test_default_runtime_is_mock(monkeypatch):
    reset_runtime()
    rt = get_runtime()
    assert rt.effective_provider == "mock"
    assert rt.mock_mode is True
    assert get_provider().is_mock is True


def test_select_provider_without_key_falls_back_to_mock():
    update_runtime(provider="openai")          # selected, but no key configured
    rt = get_runtime()
    assert rt.provider == "openai"             # the *selection* is honored
    assert rt.effective_provider == "mock"     # …but it degrades safely
    assert get_provider().is_mock is True


def test_select_provider_with_key_is_live():
    update_runtime(provider="openai", api_key="sk-test-123", fast_model="gpt-4o-mini")
    rt = get_runtime()
    assert rt.effective_provider == "openai"
    assert rt.mock_mode is False
    assert rt.active_fast_model == "gpt-4o-mini"
    prov = get_provider()
    assert prov.id == "openai" and not prov.is_mock


def test_blank_api_key_does_not_wipe_existing():
    update_runtime(provider="google", api_key="real-key")
    update_runtime(provider="google", api_key="")   # saving form without re-typing
    assert get_runtime().api_key_for("google") == "real-key"


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError):
        update_runtime(provider="not-a-provider")


def test_threshold_validation():
    with pytest.raises(ConfigError):
        update_runtime(confidence_threshold=1.5)
    update_runtime(confidence_threshold=0.6)
    assert get_runtime().confidence_threshold == 0.6


# ── Secret safety: keys must never round-trip out ─────────────────────────────
def test_snapshot_never_leaks_secrets():
    update_runtime(provider="anthropic", api_key="sk-ant-SUPERSECRET")
    snap = get_runtime().public_snapshot()
    flat = repr(snap)
    assert "SUPERSECRET" not in flat
    assert snap["keys_configured"]["anthropic"] is True
    assert snap["keys_configured"]["openai"] is False
    # mock requires no key, so it always reads as "configured"
    assert snap["keys_configured"]["mock"] is True


def test_provider_cache_rebuilds_on_key_change():
    update_runtime(provider="openai", api_key="key-A")
    a = get_provider()
    update_runtime(provider="openai", api_key="key-B")
    b = get_provider()
    assert a is not b   # a rotated key must not be served by a stale client
    reset_provider_cache()


# ── Vendor response-shape mapping (fake clients, no SDK / network) ────────────
# The SDKs aren't installed in CI, so we inject a fake client past the lazy
# import and assert each provider reads text + token usage from the right fields.
class _Boxes:
    def __init__(self, **kw):
        self.__dict__.update(kw)


async def test_anthropic_complete_maps_fields():
    from app.providers.anthropic_provider import AnthropicProvider

    class FakeMessages:
        async def create(self, **kw):
            return _Boxes(content=[_Boxes(text="pong")],
                          usage=_Boxes(input_tokens=12, output_tokens=3))

    p = AnthropicProvider(api_key="x")
    p._client = _Boxes(messages=FakeMessages())
    r = await p.complete(model="claude-haiku-4-5-20251001", prompt="ping")
    assert r.text == "pong" and r.input_tokens == 12 and r.output_tokens == 3


async def test_openai_complete_maps_fields_and_token_param_fallback():
    from app.providers.openai_provider import OpenAIProvider

    seen = {}

    class FakeCompletions:
        async def create(self, **kw):
            # First call uses the modern param; simulate a model that rejects it.
            if "max_completion_tokens" in kw and not seen.get("fellback"):
                seen["fellback"] = True
                raise TypeError("Unsupported parameter: 'max_completion_tokens'")
            seen["final_kwargs"] = kw
            return _Boxes(choices=[_Boxes(message=_Boxes(content="pong"))],
                          usage=_Boxes(prompt_tokens=9, completion_tokens=2))

    p = OpenAIProvider(api_key="x")
    p._client = _Boxes(chat=_Boxes(completions=FakeCompletions()))
    r = await p.complete(model="gpt-4o-mini", prompt="ping", max_tokens=8)
    assert r.text == "pong" and (r.input_tokens, r.output_tokens) == (9, 2)
    assert seen["fellback"] is True              # it retried with legacy param
    assert seen["final_kwargs"].get("max_tokens") == 8


async def test_google_complete_maps_fields_and_floors_token_budget():
    pytest.importorskip("google.genai")
    from app.providers.google_provider import GoogleProvider

    seen = {}

    class FakeModels:
        async def generate_content(self, **kw):
            seen["config"] = kw.get("config")
            return _Boxes(
                text="pong",
                usage_metadata=_Boxes(prompt_token_count=15, candidates_token_count=4,
                                      thoughts_token_count=10),
            )

    p = GoogleProvider(api_key="x")
    p._client = _Boxes(aio=_Boxes(models=FakeModels()))
    r = await p.complete(model="gemini-2.5-flash", prompt="ping", max_tokens=8)
    # output tokens include the billed thinking tokens (4 answer + 10 thoughts).
    assert r.text == "pong" and (r.input_tokens, r.output_tokens) == (15, 14)
    # Thinking-model headroom: a tiny max_tokens is floored so the model's internal
    # reasoning can't swallow the whole budget and return empty text (the real bug
    # that made every Gemini extraction silently degrade to mock).
    assert seen["config"].max_output_tokens >= 2048


# ── Dynamic model listing (fake clients) — this is what fills the dropdown ───
async def _aiter(items):
    for x in items:
        yield x


async def test_mock_lists_its_single_model():
    from app.providers.mock_provider import MockProvider

    assert await MockProvider().list_models() == [("mock", "Deterministic mock")]


async def test_anthropic_list_models():
    from app.providers.anthropic_provider import AnthropicProvider

    class FakeModelsAPI:
        def list(self, **kw):
            return _aiter([
                _Boxes(id="claude-opus-4-8", display_name="Claude Opus 4.8"),
                _Boxes(id="claude-haiku-4-5", display_name=None),
            ])

    p = AnthropicProvider(api_key="x")
    p._client = _Boxes(models=FakeModelsAPI())
    got = await p.list_models()
    assert ("claude-opus-4-8", "Claude Opus 4.8") in got
    assert ("claude-haiku-4-5", "claude-haiku-4-5") in got  # falls back to id as label


async def test_openai_list_models_filters_to_chat():
    from app.providers.openai_provider import OpenAIProvider

    class FakeModelsAPI:
        def list(self):
            return _aiter([
                _Boxes(id="gpt-4o"),
                _Boxes(id="gpt-4o-mini"),
                _Boxes(id="text-embedding-3-large"),  # filtered out
                _Boxes(id="whisper-1"),                # filtered out
                _Boxes(id="o3-mini"),
            ])

    p = OpenAIProvider(api_key="x")
    p._client = _Boxes(models=FakeModelsAPI())
    ids = [mid for mid, _ in await p.list_models()]
    assert "gpt-4o" in ids and "o3-mini" in ids
    assert "text-embedding-3-large" not in ids and "whisper-1" not in ids


async def test_google_list_models_filters_and_strips_prefix():
    pytest.importorskip("google.genai")
    from app.providers.google_provider import GoogleProvider

    class FakeModelsAPI:
        def list(self):
            return _aiter([
                # General text/vision models — these CAN do the job (kept).
                _Boxes(name="models/gemini-flash-latest", display_name="Gemini Flash Latest",
                       supported_actions=["generateContent"]),
                _Boxes(name="models/gemini-2.5-flash", display_name="Gemini 2.5 Flash",
                       supported_actions=["generateContent"]),
                # Specialized families that CANNOT extract a receipt (all filtered).
                _Boxes(name="models/text-embedding-004", display_name="Embeddings",
                       supported_actions=["embedContent"]),
                _Boxes(name="models/gemini-2.5-flash-image", display_name="Nano Banana",
                       supported_actions=["generateContent"]),     # image generation
                _Boxes(name="models/lyria-3-pro", display_name="Lyria 3 Pro",
                       supported_actions=[]),                       # music (empty actions → by name)
                _Boxes(name="models/gemini-robotics-er-1.5", display_name="Robotics-ER",
                       supported_actions=["generateContent"]),      # robotics agent
                _Boxes(name="models/gemini-2.5-flash-tts", display_name="Flash TTS",
                       supported_actions=["generateContent"]),      # speech
            ])

    p = GoogleProvider(api_key="x")
    p._client = _Boxes(aio=_Boxes(models=FakeModelsAPI()))
    ids = [mid for mid, _ in await p.list_models()]
    assert "gemini-flash-latest" in ids and "gemini-2.5-flash" in ids   # prefix stripped, kept
    for excluded in ("text-embedding-004", "gemini-2.5-flash-image",
                     "lyria-3-pro", "gemini-robotics-er-1.5", "gemini-2.5-flash-tts"):
        assert excluded not in ids


# ── Graceful degradation is *observable* (F6 must not hide behind "mock") ─────
async def test_vision_worker_surfaces_provider_degradation(monkeypatch):
    """When a live provider call fails, the worker degrades to the deterministic
    parse — but it must SAY so. A failed Gemini run masquerading as a clean mock
    run is exactly the silent-degradation blind spot AgentOps has to catch."""
    from app.extraction import vision
    from app.providers.base import LLMProvider

    class FailingProvider(LLMProvider):
        id = "google"

        async def complete(self, *, model, prompt, system=None, max_tokens=512):
            raise RuntimeError("unused")

        async def extract_receipt(self, *a, **k):
            raise ValueError("no JSON object found in model response")

    monkeypatch.setattr(vision, "get_provider", lambda: FailingProvider())
    res = await vision.extract_receipt(
        "run_x", "brew.txt", b"BREW & CO\nTOTAL  450.00\n", "receipt"
    )
    assert res.transaction is not None              # F6: the run still completed
    assert res.model == "google→mock"          # fallback is visible, not just "mock"
    assert res.error and "no JSON" in res.error      # the cause is captured
    assert res.tokens_in == 0 and res.tokens_out == 0
