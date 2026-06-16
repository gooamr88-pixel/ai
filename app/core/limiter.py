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

from app.core.config import settings


def _get_real_ip(request: Request) -> str:
    """Extract the real client IP for rate-limiting.

    The leftmost X-Forwarded-For entry is client-controlled and trivially
    spoofable (anyone can send `X-Forwarded-For: <anything>` to mint a fresh
    rate-limit bucket). Our reverse proxy (Nginx with $proxy_add_x_forwarded_for)
    APPENDS the real peer to the chain, so the trustworthy client IP is the
    Nth entry from the RIGHT, where N = number of trusted proxies.

    When TRUSTED_PROXY_COUNT is 0 (direct exposure), X-Forwarded-For is
    ignored entirely and we use the real socket peer address.
    """
    proxies = settings.TRUSTED_PROXY_COUNT
    if proxies > 0:
        forwarded = request.headers.get("X-Forwarded-For", "")
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            idx = min(proxies, len(parts))
            return parts[-idx]
    return get_remote_address(request)


limiter = Limiter(key_func=_get_real_ip, default_limits=["30/minute"])
