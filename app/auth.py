"""API key authentication utilities.

This module provides a FastAPI dependency for extracting and validating
API keys.  Clients are expected to supply their key via the ``X-API-Key``
header.  If the service is configured without any API keys (i.e. the
``API_KEYS`` environment variable is empty), the authentication check is
disabled and all requests are allowed through.

Usage example::

    from fastapi import Depends
    from .auth import get_current_identity

    @app.get("/protected")
    async def protected_route(identity: str = Depends(get_current_identity)):
        return {"hello": identity}

"""

from fastapi import Header, HTTPException, status
from typing import Optional

from .config import get_settings


async def get_current_identity(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """Return the caller's identity after validating the API key.

    If no API keys are configured in ``API_KEYS``, authentication is disabled
    and an empty string is returned as the identity.  Otherwise, the provided
    key must match one of the configured keys.  If validation fails, an
    HTTP 401 error is raised.

    Parameters
    ----------
    x_api_key: Optional[str]
        The value of the ``X-API-Key`` header sent by the client.

    Returns
    -------
    str
        The trimmed API key used to authenticate the request.  This value
        functions as the caller's identity for the purpose of rate limiting.
    """
    settings = get_settings()
    allowed_keys = settings.parsed_api_keys
    # If no keys configured, skip authentication entirely
    if not allowed_keys:
        return ""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "API key"},
        )
    key = x_api_key.strip()
    if key not in allowed_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "API key"},
        )
    return key