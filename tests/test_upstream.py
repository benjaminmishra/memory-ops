import pytest

from app import upstream, config
import httpx


@pytest.mark.asyncio
async def test_call_llm_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("UPSTREAM_API_KEY", "secret")
    config.get_settings.cache_clear()
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, url, headers=None, json=None):
            captured["headers"] = headers
            return DummyResponse()

    monkeypatch.setattr(httpx, "AsyncClient", DummyClient)

    result = await upstream.call_llm(messages=[{"role": "user", "content": "hi"}])
    assert result == {"ok": True}
    assert captured["headers"]["Authorization"] == "secret"


@pytest.mark.asyncio
async def test_call_llm_streams(monkeypatch):
    class StreamResponse:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in ['data: {"delta": "hi"}', '', 'data: [DONE]']:
                yield line

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def post(self, url, headers=None, json=None):
            return StreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", DummyClient)

    gen = await upstream.call_llm(messages=[{"role": "u", "content": "h"}], stream=True)
    events = []
    async for event in gen:
        events.append(event)
    assert events == [{"delta": "hi"}]
