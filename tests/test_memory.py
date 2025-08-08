import importlib

from app import config


def setup_memory(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    config.get_settings.cache_clear()
    from app import memory
    importlib.reload(memory)
    return memory


def test_add_and_get_messages(monkeypatch, tmp_path):
    memory = setup_memory(monkeypatch, tmp_path)
    memory.add_message("s", "user", "hello", 1)
    memory.add_message("s", "assistant", "hi", 2)
    msgs = memory.get_messages("s")
    assert [m["content"] for m in msgs] == ["hello", "hi"]


def test_get_context(monkeypatch, tmp_path):
    memory = setup_memory(monkeypatch, tmp_path)
    memory.add_message("sess", "user", "first", 1)
    memory.add_message("sess", "assistant", "second", 2)
    context = memory.get_context("sess")
    assert context == "first\nsecond"


def test_clear_session(monkeypatch, tmp_path):
    memory = setup_memory(monkeypatch, tmp_path)
    memory.add_message("id", "user", "msg", 1)
    memory.clear_session("id")
    assert memory.get_messages("id") == []
