"""The configuration HTTP surface — the dashboard's plug-and-play control plane.

In-process (no live server). Verifies the catalog endpoint, that a provider/model
can be selected at runtime, that `/health` reflects it, that secrets never come
back over the wire, and that the admin-token guard works when enabled.
"""
import httpx
from httpx import ASGITransport

from app.main import app


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_providers_catalog_lists_all_four():
    async with _client() as c:
        data = (await c.get("/providers")).json()
    ids = {p["id"] for p in data["providers"]}
    assert ids == {"anthropic", "google", "openai", "mock"}
    google = next(p for p in data["providers"] if p["id"] == "google")
    assert google["requires_key"] is True
    assert any(m["id"] == "gemini-2.5-flash" for m in google["models"])
    assert data["effective"] == "mock"   # no keys in the test env


async def test_select_provider_updates_config_and_health():
    async with _client() as c:
        put = await c.put("/config", json={
            "provider": "google", "api_key": "test-gemini-key", "fast_model": "gemini-2.5-flash",
        })
        assert put.status_code == 200
        body = put.json()
        assert body["effective_provider"] == "google"
        assert body["mock_mode"] is False
        assert body["fast_model"] == "gemini-2.5-flash"

        health = (await c.get("/health")).json()
        assert health["provider"] == "google"
        assert health["mock_mode"] is False
        assert "Gemini" in health["provider_label"]


async def test_config_get_hides_the_key():
    async with _client() as c:
        await c.put("/config", json={"provider": "openai", "api_key": "sk-DO-NOT-LEAK"})
        cfg = (await c.get("/config")).json()
    assert "sk-DO-NOT-LEAK" not in str(cfg)
    assert cfg["keys_configured"]["openai"] is True


async def test_fetch_models_mock_needs_no_key():
    async with _client() as c:
        res = await c.post("/providers/mock/models", json={})
    body = res.json()
    assert res.status_code == 200 and body["ok"] is True
    assert body["models"] == [{"id": "mock", "label": "Deterministic mock"}]


async def test_fetch_models_requires_a_key():
    async with _client() as c:
        res = await c.post("/providers/openai/models", json={})  # no key anywhere
    assert res.status_code == 400


async def test_fetch_models_unknown_provider():
    async with _client() as c:
        res = await c.post("/providers/nope/models", json={"api_key": "x"})
    assert res.status_code == 400


async def test_config_test_endpoint_mock_is_ok():
    async with _client() as c:
        res = await c.post("/config/test", json={"provider": "mock"})
    assert res.status_code == 200 and res.json()["ok"] is True


async def test_config_test_missing_key_is_400():
    async with _client() as c:
        res = await c.post("/config/test", json={"provider": "openai"})  # no key anywhere
    assert res.status_code == 400


async def test_invalid_provider_is_400():
    async with _client() as c:
        res = await c.put("/config", json={"provider": "nope"})
    assert res.status_code == 400


async def test_admin_token_guard(monkeypatch):
    # When an admin token is configured, writes require the header.
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("LEDGER_ADMIN_TOKEN", "s3cret")
    try:
        async with _client() as c:
            denied = await c.put("/config", json={"provider": "mock"})
            assert denied.status_code == 401
            ok = await c.put("/config", json={"provider": "mock"},
                             headers={"X-Admin-Token": "s3cret"})
            assert ok.status_code == 200
    finally:
        get_settings.cache_clear()
