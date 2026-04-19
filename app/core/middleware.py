"""
Ruya — Request ID Middleware
===============================
Injects a unique X-Request-ID header into every request/response
for end-to-end tracing in logs.
"""

import uuid
import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID and log request duration."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])

        # Store on request state for use in endpoints/services
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        response.headers["X-Request-ID"] = request_id

        logger.info(
            f"[REQ] {request.method} {request.url.path} → {response.status_code} "
            f"({elapsed:.2f}s) [rid={request_id}]"
        )
        return response
