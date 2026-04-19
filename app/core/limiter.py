"""
Ruya — Rate Limiter Configuration
====================================
Extracted to its own module to avoid circular imports
between main.py and endpoint files.
Uses X-Forwarded-For when behind a reverse proxy (Nginx).
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def _get_real_ip(request: Request) -> str:
    """Extract real client IP from X-Forwarded-For (proxy) or fall back to remote address."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        # X-Forwarded-For: client, proxy1, proxy2 → take the first (client)
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_get_real_ip, default_limits=["30/minute"])
