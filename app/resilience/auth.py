"""API-Key Authentication Middleware and Dependency for The Secure LLM Gateway."""

import hashlib
from typing import Optional, Tuple
from fastapi import Header, HTTPException, Request
from app.config import settings


def hash_api_key(api_key: str) -> str:
    """
    Generates a secure, deterministic one-way hash identifier for an API key.
    Raw keys are NEVER stored in SQLite logs or database tables.
    """
    if not api_key:
        return "key_anonymous"
    h = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return f"key_{h}"


async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Tuple[str, str]:
    """
    Validates the X-API-Key header (or Authorization: Bearer fallback).
    Returns a tuple of (raw_key, api_key_hash).
    Raises HTTP 401 if missing or invalid.
    """
    if not settings.REQUIRE_API_KEY:
        raw_key = x_api_key or "anonymous"
        return raw_key, hash_api_key(raw_key)

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

    # Validate against configured list of keys or valid key formats
    if settings.VALID_API_KEYS and key not in settings.VALID_API_KEYS:
        if not key.startswith("sk-") and not key.startswith("demo-"):
            raise HTTPException(
                status_code=401,
                detail="Invalid API Key. Access denied.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    key_hash = hash_api_key(key)
    return key, key_hash
