"""Application settings.

The single most important line here is `mock_mode`: if no ANTHROPIC_API_KEY is
present the whole system runs deterministically off canned extractions, so the
live demo can never hard-fail on stage (failure mode F7 in ARCHITECTURE.md).
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"), env_prefix="", extra="ignore"
    )

    anthropic_api_key: str = ""

    # Two-tier model routing (latency × cost × accuracy, see ARCHITECTURE §7).
    ledger_model_fast: str = "claude-haiku-4-5-20251001"
    ledger_model_deep: str = "claude-opus-4-8"

    # Guardrail thresholds.
    ledger_confidence_threshold: float = 0.80
    ledger_match_threshold: float = 0.82

    # Fan-out concurrency ceiling (protects the model rate limit).
    ledger_max_concurrency: int = 8

    # Transient-failure handling for live model calls (rate limits, 5xx, timeouts).
    # Retries with exponential backoff + jitter before degrading to the
    # deterministic parser (ARCHITECTURE.md §8, F4).
    ledger_max_retries: int = 3
    ledger_retry_base_delay: float = 0.5

    # Observability (optional; in-process tracing always works without these).
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def mock_mode(self) -> bool:
        return not bool(self.anthropic_api_key.strip())

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
