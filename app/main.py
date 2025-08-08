"""FastAPI entrypoint for the LLM proxy service.

This module wires together authentication, rate limiting, context
compression, memory persistence and upstream requests to expose an
OpenAI‑compatible chat completions API under ``/v1/chat/completions``.

Clients should send requests in the same JSON format expected by
OpenAI's API.  The proxy will process the request as follows:

1. Authenticate the caller via the ``X-API-Key`` header (if keys are
   configured).
2. Enforce per‑minute request and token quotas per API key.
3. Fetch previous conversation context from the memory store using the
   ``X-Session-ID`` header as the key.  If not provided, uses the API
   key as the session identifier.
4. Compress the context using QR‑HEAD to discard irrelevant tokens.
5. Forward the condensed prompt to the upstream LLM using the base URL
   and key configured in environment variables, or the caller's
   ``Authorization`` header if present.
6. Store the user's message and assistant's reply in the memory store.

The response format mirrors the OpenAI API and includes custom
headers ``x-tokens-before``, ``x-tokens-after`` and ``x-tokens-saved``
to report compression metrics.
"""

from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import json

from .auth import get_current_identity
from .rate_limit import get_rate_limiter
from .compression import compress
from .memory import add_message, get_context
from .upstream import call_llm
from .config import get_settings


app = FastAPI(title="MemoryOps LLM Proxy")

# Instantiate rate limiter once
limiter = get_rate_limiter()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    response: Response,
    identity: str = Depends(get_current_identity),
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> Response:
    """Handle chat completion requests in OpenAI format."""
    settings = get_settings()
    # Parse JSON body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    # Validate required fields
    if "messages" not in payload or not isinstance(payload["messages"], list):
        raise HTTPException(status_code=422, detail="'messages' must be a list")
    messages: List[Dict[str, Any]] = payload["messages"]
    model = payload.get("model", settings.upstream_model)
    stream = payload.get("stream", False)
    # Determine session ID: provided header > identity > default
    session_id = x_session_id or identity or "anonymous"
    # Fetch previous context
    context = get_context(session_id)
    # Identify current user query
    # The last message in the list is assumed to be the new user input
    if not messages:
        raise HTTPException(status_code=422, detail="No messages provided")
    current_msg = messages[-1]
    if current_msg.get("role") != "user":
        raise HTTPException(status_code=422, detail="Last message must have role 'user'")
    query = current_msg.get("content", "")
    # Compress context relative to query
    condensed_context, tokens_before, tokens_after = compress(query, context)
    # Rate limit using tokens_before (worst case consumption)
    limiter.check(identity, tokens_before)
    # Build new messages for upstream: previous condensed context + current messages
    # Format condensed context as a single assistant message so the model can reference
    upstream_messages: List[Dict[str, str]] = []
    if condensed_context:
        upstream_messages.append({"role": "system", "content": condensed_context})
    # Append all user/assistant messages in current payload
    # (We don't include old memory messages to avoid duplication.)
    for m in messages:
        upstream_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    # Forward to upstream provider
    result = await call_llm(
        messages=upstream_messages,
        model=model,
        stream=stream,
        authorization=authorization,
    )
    # Save messages to memory for future context (non‑streaming case only)
    # Compute tokens used as after compression to approximate consumption
    add_message(session_id, role="user", content=query, tokens=tokens_after)
    # For assistant reply, we approximate tokens by length of choices
    if not stream and isinstance(result, dict):
        choices = result.get("choices", [])
        if choices:
            reply_content = choices[0].get("message", {}).get("content", "")
            reply_tokens = len(reply_content.split())  # naive token count
            add_message(session_id, role="assistant", content=reply_content, tokens=reply_tokens)
    # Set custom headers for compression stats
    response.headers["x-tokens-before"] = str(tokens_before)
    response.headers["x-tokens-after"] = str(tokens_after)
    response.headers["x-tokens-saved"] = str(tokens_before - tokens_after)
    # Non streaming: return JSON directly
    if not stream:
        return JSONResponse(status_code=200, content=result)
    # Streaming: convert event generator to streaming response
    async def event_stream():
        async for event in result:
            yield f"data: {json.dumps(event)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")