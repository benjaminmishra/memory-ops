"""Simple in-memory rate limiter.

This module implements a sliding window rate limiter for both request count
and token consumption.  Each identity (API key) maintains its own quota.
The limiter uses an in-memory store and is suitable for a single-process
deployment.  For horizontally scaled deployments, a shared cache such as
Redis should be used instead of this module.

Usage::

    from fastapi import Depends
    from .rate_limit import get_rate_limiter

    limiter = get_rate_limiter()

    @app.post("/v1/chat/completions")
    async def completions(..., identity: str = Depends(get_current_identity)):
        limiter.check(identity, tokens_used)
        ...
"""

from collections import deque
from dataclasses import dataclass
from time import time
from typing import Dict, Deque, Tuple
from fastapi import HTTPException, status

from .config import get_settings


@dataclass
class Quota:
    """Track timestamped request and token usage entries.

    Attributes
    ----------
    reqs: Deque[float]
        Timestamps (seconds since epoch) of requests in the current window.
    tokens: Deque[Tuple[float, int]]
        Pairs of (timestamp, token count) recording token usage events.
    """

    reqs: Deque[float]
    tokens: Deque[Tuple[float, int]]


class RateLimiter:
    """Sliding‑window rate limiter for requests and tokens."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        # Use tokens_per_minute if specified; fall back to rate_limit_tpm for backwards compatibility
        settings = get_settings()
        # Prioritise tokens_per_minute from settings if set differently than default
        self.tokens_per_minute = tokens_per_minute or settings.tokens_per_minute or settings.rate_limit_tpm
        # In-memory store of quotas per identity
        self._store: Dict[str, Quota] = {}

    def _get_quota(self, identity: str) -> Quota:
        if identity not in self._store:
            self._store[identity] = Quota(deque(), deque())
        return self._store[identity]

    def check(self, identity: str, tokens: int) -> None:
        """Check and update the rate limits for a given identity.

        Parameters
        ----------
        identity: str
            The caller's API key (or empty string when auth disabled).
        tokens: int
            Number of prompt+completion tokens consumed by the current request.

        Raises
        ------
        HTTPException
            If the request exceeds the configured request or token rate limit.
        """
        now = time()
        quota = self._get_quota(identity)
        # Drop expired entries (older than 60 seconds)
        window_start = now - 60
        while quota.reqs and quota.reqs[0] <= window_start:
            quota.reqs.popleft()
        while quota.tokens and quota.tokens[0][0] <= window_start:
            quota.tokens.popleft()
        # Check request limit
        if self.requests_per_minute > 0 and len(quota.reqs) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: too many requests per minute",
                headers={"Retry-After": "60"},
            )
        # Check token limit
        total_tokens = sum(t for _, t in quota.tokens)
        if self.tokens_per_minute > 0 and (total_tokens + tokens) > self.tokens_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded: too many tokens per minute",
                headers={"Retry-After": "60"},
            )
        # Record current request and tokens
        quota.reqs.append(now)
        quota.tokens.append((now, tokens))


def get_rate_limiter() -> RateLimiter:
    """Instantiate a rate limiter using current settings.

    This factory function ensures that the rate limiter picks up any
    customisation from environment variables (e.g. REQUESTS_PER_MINUTE,
    TOKENS_PER_MINUTE).  The returned instance can be reused across requests.
    """
    settings = get_settings()
    return RateLimiter(settings.requests_per_minute, settings.tokens_per_minute or settings.rate_limit_tpm)