"""The provider & model catalog — one source of truth for the control plane.

This table drives three things so they can never drift apart:
  1. the dashboard's provider list + a *fallback* model set shown before a key is
     entered (the authoritative model list is fetched live from the vendor),
  2. the per-provider default model routing,
  3. the live cost meter (`observability.estimate_cost`).

`key_env` records the conventional environment-variable name for each provider's
key purely for documentation/reference — keys themselves are supplied from the
dashboard at runtime, never read from the environment.

Pricing is **approximate public list price** (USD per 1M tokens, input/output)
and is used only for the on-screen cost estimate — it is intentionally a plain,
auditable table so a routing/cost decision can always be explained. Unknown
models (e.g. a freshly-released one fetched live) fall back to zero cost — we
under-report rather than guess high.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    price_in: float   # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    requires_key: bool
    key_env: tuple[str, ...]
    default_fast: str
    default_deep: str
    models: tuple[ModelInfo, ...]
    docs_url: str = ""

    def model_ids(self) -> list[str]:
        return [m.id for m in self.models]


# ── The catalog ──────────────────────────────────────────────────────────────
PROVIDER_INFO: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo(
        id="anthropic",
        label="Anthropic · Claude",
        requires_key=True,
        key_env=("ANTHROPIC_API_KEY",),
        default_fast="claude-haiku-4-5-20251001",
        default_deep="claude-opus-4-8",
        docs_url="https://console.anthropic.com/settings/keys",
        models=(
            ModelInfo("claude-opus-4-8", "Claude Opus 4.8", 15.0, 75.0),
            ModelInfo("claude-sonnet-4-6", "Claude Sonnet 4.6", 3.0, 15.0),
            ModelInfo("claude-haiku-4-5-20251001", "Claude Haiku 4.5", 1.0, 5.0),
        ),
    ),
    "google": ProviderInfo(
        id="google",
        label="Google · Gemini",
        requires_key=True,
        key_env=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        default_fast="gemini-2.5-flash",
        default_deep="gemini-2.5-pro",
        docs_url="https://aistudio.google.com/apikey",
        models=(
            ModelInfo("gemini-2.5-pro", "Gemini 2.5 Pro", 1.25, 10.0),
            ModelInfo("gemini-2.5-flash", "Gemini 2.5 Flash", 0.30, 2.50),
            ModelInfo("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", 0.10, 0.40),
            ModelInfo("gemini-2.0-flash", "Gemini 2.0 Flash", 0.10, 0.40),
        ),
    ),
    "openai": ProviderInfo(
        id="openai",
        label="OpenAI · GPT",
        requires_key=True,
        key_env=("OPENAI_API_KEY",),
        default_fast="gpt-4o-mini",
        default_deep="gpt-4o",
        docs_url="https://platform.openai.com/api-keys",
        models=(
            ModelInfo("gpt-4o", "GPT-4o", 2.5, 10.0),
            ModelInfo("gpt-4o-mini", "GPT-4o mini", 0.15, 0.60),
            ModelInfo("gpt-4.1", "GPT-4.1", 2.0, 8.0),
            ModelInfo("gpt-4.1-mini", "GPT-4.1 mini", 0.40, 1.60),
        ),
    ),
    "mock": ProviderInfo(
        id="mock",
        label="Mock · deterministic",
        requires_key=False,
        key_env=(),
        default_fast="mock",
        default_deep="mock",
        docs_url="",
        models=(ModelInfo("mock", "Deterministic mock", 0.0, 0.0),),
    ),
}

PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_INFO.keys())

# Flattened {model_id: (price_in, price_out)} for the cost meter, plus the
# internal pseudo-models the deterministic paths report.
_PRICING: dict[str, tuple[float, float]] = {
    "mock": (0.0, 0.0),
    "deterministic": (0.0, 0.0),
}
for _info in PROVIDER_INFO.values():
    for _m in _info.models:
        _PRICING[_m.id] = (_m.price_in, _m.price_out)


def get_model_pricing(model: str) -> tuple[float, float]:
    """(price_in, price_out) per 1M tokens; (0, 0) for unknown / mock models."""
    return _PRICING.get(model, (0.0, 0.0))
