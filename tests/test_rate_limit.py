import pytest
from fastapi import HTTPException

import app.rate_limit as rl


def test_allows_within_limits(monkeypatch):
    limiter = rl.RateLimiter(requests_per_minute=2, tokens_per_minute=100)
    t = 0

    def fake_time():
        return t

    monkeypatch.setattr(rl, "time", fake_time)

    limiter.check("id", 10)
    t += 1
    limiter.check("id", 20)


def test_blocks_excess_requests(monkeypatch):
    limiter = rl.RateLimiter(requests_per_minute=2, tokens_per_minute=100)
    t = 0

    def fake_time():
        return t

    monkeypatch.setattr(rl, "time", fake_time)

    limiter.check("id", 1)
    t += 1
    limiter.check("id", 1)
    t += 1
    with pytest.raises(HTTPException) as exc:
        limiter.check("id", 1)
    assert exc.value.status_code == 429


def test_blocks_excess_tokens(monkeypatch):
    limiter = rl.RateLimiter(requests_per_minute=5, tokens_per_minute=100)
    t = 0

    def fake_time():
        return t

    monkeypatch.setattr(rl, "time", fake_time)

    limiter.check("id", 60)
    t += 1
    with pytest.raises(HTTPException):
        limiter.check("id", 50)
