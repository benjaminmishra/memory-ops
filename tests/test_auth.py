import pytest
from fastapi import HTTPException

from app import auth, config


@pytest.mark.asyncio
async def test_returns_empty_when_no_keys(monkeypatch):
    monkeypatch.setenv("API_KEYS", "")
    config.get_settings.cache_clear()
    identity = await auth.get_current_identity()
    assert identity == ""


@pytest.mark.asyncio
async def test_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", "key1,key2")
    config.get_settings.cache_clear()
    identity = await auth.get_current_identity("key2")
    assert identity == "key2"


@pytest.mark.asyncio
async def test_missing_key_raises(monkeypatch):
    monkeypatch.setenv("API_KEYS", "abc")
    config.get_settings.cache_clear()
    with pytest.raises(HTTPException) as exc:
        await auth.get_current_identity(None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_key_raises(monkeypatch):
    monkeypatch.setenv("API_KEYS", "valid")
    config.get_settings.cache_clear()
    with pytest.raises(HTTPException):
        await auth.get_current_identity("invalid")
