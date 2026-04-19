"""
Ruya — Enterprise Core API
============================
Clean FastAPI entry point with global exception handling, rate limiting,
request tracing, and optional API key auth.
"""

import os
import logging

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.limiter import limiter
from app.core.config import settings
from app.core.auth import verify_api_key
from app.core.middleware import RequestIDMiddleware
from app.api.v1.router import api_v1_router

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ruya Enterprise Core API",
    description="Scalable AI Backend for Ruya Platform — Question Bank, MindMap Image, Video (8-10min), Podcast (7-10min).",
    version="5.1.0",
)

# Attach limiter to app state (required by slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middleware (order matters: first added = outermost) ────────────────────
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key", "X-Request-ID"],
)

# ── Global Exception Handlers ─────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Validation / business logic errors → 422."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Service unavailable (AI timeout, TTS failure) → 503."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch-all — log full error server-side, return clean message to client."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )

# ── Mount API Routers (protected by API key when API_KEY is set) ──────────
app.include_router(
    api_v1_router,
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)],
)

# ── Serve generated media files (videos, podcasts) as static assets ────────
MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
os.makedirs(MEDIA_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# ── Health Check ───────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "operational",
        "service": "Ruya Cognitive AI Engine",
        "version": "5.1.0",
        "modules": ["question-bank", "mindmap-image", "video-8min", "podcast-8min"],
    }