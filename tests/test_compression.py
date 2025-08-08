
from app import compression, config


def test_compress_delegates_and_uses_settings(monkeypatch):
    monkeypatch.setenv("TOP_K", "10")
    config.get_settings.cache_clear()

    def fake_reduce(query, context):
        assert config.get_settings().top_k == 10
        return "condensed text", 200, 50

    monkeypatch.setattr(compression.qr_retriever, "reduce", fake_reduce)

    result = compression.compress("question", "context")
    assert result == ("condensed text", 200, 50)
