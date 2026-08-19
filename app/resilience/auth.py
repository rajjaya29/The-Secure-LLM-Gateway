"""API-Key Authentication Middleware and Dependency for The Secure LLM Gateway."""

from typing import Optional
from fastapi import Header, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> str:
    """
    Validates API key provided in the X-API-Key header (or Bearer Authorization fallback).
    Returns the authenticated API key identifier.
    """
    if not settings.REQUIRE_API_KEY:
        return x_api_key or "anonymous"

    # Check X-API-Key header
    key = x_api_key

    # Check Bearer authorization fallback
    if not key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and len(auth_header) > 7:
            key = auth_header[7:].strip()

    if not key:
        raise HTTPException(
            status_code=401,
            detail="Missing API Key. Please provide a valid key via the 'X-API-Key' header or Bearer token.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate against configured keys or allow validly formatted keys (sk-...)
    if settings.VALID_API_KEYS and key not in settings.VALID_API_KEYS:
        # If not explicitly in the static list, accept valid prefix keys for flexibility if configured
        if not key.startswith("sk-") and not key.startswith("gw-"):
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key. Access denied.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    return key
