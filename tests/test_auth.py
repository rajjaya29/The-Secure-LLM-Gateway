"""Unit tests for X-API-Key Authentication and API key hashing."""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.resilience.auth import hash_api_key


def test_api_key_hashing_security():
    # Raw API keys must never match their hashed identifiers
    raw_key = "sk-secret-key-999"
    key_hash = hash_api_key(raw_key)
    
    assert key_hash.startswith("key_")
    assert raw_key not in key_hash
    assert len(key_hash) == 16  # "key_" + 12 chars
    # Deterministic
    assert key_hash == hash_api_key(raw_key)


@pytest.mark.asyncio
async def test_auth_missing_api_key_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"model": "mock-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert res.status_code == 401
        assert "Missing API Key" in res.json()["detail"]


@pytest.mark.asyncio
async def test_auth_invalid_api_key_rejected():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.post(
            "/v1/chat/completions",
            headers={"X-API-Key": "invalid-unauthorized-key"},
            json={"model": "mock-model", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert res.status_code == 401
        assert "Invalid API Key" in res.json()["detail"]


@pytest.mark.asyncio
async def test_auth_valid_api_keys():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        for key in ["sk-test-key-123", "demo-key", "sk-admin-master-key"]:
            res = await client.post(
                "/v1/chat/completions",
                headers={"X-API-Key": key},
                json={"model": "mock-model", "messages": [{"role": "user", "content": "ping"}]},
            )
            assert res.status_code == 200
            assert "choices" in res.json()
