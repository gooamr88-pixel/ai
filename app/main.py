"""
Ruya — Enterprise Core API
============================
Clean FastAPI entry point with global exception handling and rate limiting.
All endpoint logic lives in api/v1/endpoints/.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.limiter import limiter
from app.api.v1.router import api_v1_router
from app.core.config import settings

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ruya Enterprise Core API",
    description="Scalable AI Backend for Ruya Platform — Question Bank, MindMap Image, Video (8-10min), Podcast (7-10min).",
    version="5.0.0",
)

# Attach limiter to app state (required by slowapi)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# ── Mount API Routers ──────────────────────────────────────────────────────
app.include_router(api_v1_router, prefix="/api/v1")

# ── Serve generated media files (videos, podcasts) as static assets ────────
import os
from fastapi.staticfiles import StaticFiles

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
os.makedirs(MEDIA_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")

# ── Health Check ───────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "operational",
        "service": "Ruya Cognitive AI Engine",
        "version": "5.0.0",
        "modules": ["question-bank", "mindmap-image", "video-8min", "podcast-8min"],
    }