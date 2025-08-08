"""Helper for calling the upstream LLM provider.

This module encapsulates the logic for forwarding chat completions
requests to the configured upstream provider (e.g. OpenAI, Anthropic).
It respects the ``UPSTREAM_BASE`` and ``UPSTREAM_API_KEY`` environment
variables set via :class:`app.config.Settings`.

Calls are made asynchronously using httpx.  In the case of streaming
responses, the caller is expected to handle SSE format on its own; this
module simply yields JSON event payloads as they arrive.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import httpx

from .config import get_settings


async def call_llm(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    stream: bool = False,
    authorization: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """Forward a chat completion request to the upstream provider.

    Parameters
    ----------
    messages: list of dict
        The chat messages as expected by the OpenAI API format.
    model: Optional[str]
        The identifier of the model to use.  If omitted, uses the default
        configured in ``UPSTREAM_MODEL``.
    stream: bool
        Whether to stream the response as Server Sent Events.  If
        ``True``, this function returns an async generator that yields
        decoded JSON events; otherwise it returns the full JSON response.
    authorization: Optional[str]
        The caller's ``Authorization`` header.  If provided, this is
        forwarded directly to the upstream.  Otherwise, the value of
        ``UPSTREAM_API_KEY`` is used.
    extra: Optional[dict]
        Additional payload fields to include in the request body.

    Returns
    -------
    dict or async generator
        Either the parsed JSON response (non‑streaming) or an async
        generator yielding JSON events (streaming).
    """
    settings = get_settings()
    url = settings.upstream_base.rstrip("/") + "/v1/chat/completions"
    headers: Dict[str, str] = {}
    # Choose API key: prefer caller's Authorization header, else env variable
    api_key = authorization or settings.upstream_api_key
    if api_key:
        headers["Authorization"] = api_key
    # Compose request body
    payload: Dict[str, Any] = {
        "model": model or settings.upstream_model,
        "messages": messages,
        "stream": stream,
    }
    if extra:
        payload.update(extra)
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        # Streaming responses return text/event-stream; parse incrementally
        if stream:
            async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
                async for line in resp.aiter_lines():
                    # Skip empty lines and the final '[DONE]'
                    if not line or line.strip() == "data: [DONE]":
                        continue
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        yield json.loads(data)
                return
            return event_generator()
        else:
            return resp.json()