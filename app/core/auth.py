"""
Ruya — API Key Authentication
================================
Simple X-API-Key header-based auth.
If API_KEY is not set in env, auth is disabled (open mode for local dev).
"""

import logging
import secrets

from fastapi import Request, HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


async def verify_api_key(request: Request) -> None:
    """
    FastAPI dependency that checks X-API-Key header.
    Skips auth if API_KEY is not configured (dev mode).
    Always allows health check (/) and docs (/docs, /openapi.json).
    """
    # Skip auth if no key configured (local dev)
    if not settings.API_KEY:
        return

    # Always allow health check and docs
    path = request.url.path
    if path in ("/", "/docs", "/redoc", "/openapi.json"):
        return

    api_key = request.headers.get("X-API-Key", "")
    # Constant-time comparison to avoid leaking the key via timing side-channel.
    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        logger.warning(f"[AUTH] Rejected request to {path} — invalid API key")
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
