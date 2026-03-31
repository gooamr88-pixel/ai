"""
Nabda — Enterprise Core API
============================
Clean FastAPI entry point optimized for VPS/Docker deployment.
All endpoint logic lives in api/v1/endpoints/.
"""

import os
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
import httpx
from elevenlabs.core.api_error import ApiError

from app.core.limiter import limiter
from app.api.v1.router import api_v1_router
from app.core.config import settings

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Ensure temp dir exists ─────────────────────────────────────────────────
os.makedirs(settings.TEMP_DIR, exist_ok=True)

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nabda AI Engine",
    description=(
        "Scalable AI Backend — 4 Specialized Endpoints:\n"
        "- `/question-bank` — Pure JSON question generation\n"
        "- `/mindmap` — Rendered mindmap image URL\n"
        "- `/media/video/generate` — 7-10 min video (async)\n"
        "- `/media/podcast/generate` — 7-10 min podcast (async)\n"
        "- `/media/jobs/{id}` — Poll job status"
    ),
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
    logger.error(f"RuntimeError on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=503, content={"error": f"السيرفر فشل بسبب: {str(exc)}"})


@app.exception_handler(httpx.HTTPStatusError)
async def http_api_error_handler(request: Request, exc: httpx.HTTPStatusError):
    if exc.response.status_code in (401, 403, 429):
        logger.error(f"Quota error: {exc}")
        return JSONResponse(status_code=exc.response.status_code, content={"error": "فشل توليد المحتوى بسبب انتهاء الكوتة (الرصيد) الخاصة بالـ API"})
    logger.error(f"HTTPStatusError on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": f"السيرفر فشل بسبب: {str(exc)}"})


@app.exception_handler(ApiError)
async def elevenlabs_api_error_handler(request: Request, exc: ApiError):
    if exc.status_code in (401, 403, 429):
        logger.error(f"ElevenLabs Quota error: {exc}")
        return JSONResponse(status_code=exc.status_code, content={"error": "فشل توليد المحتوى بسبب انتهاء الكوتة (الرصيد) الخاصة بالـ API"})
    logger.error(f"ElevenLabs error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": f"السيرفر فشل بسبب: {str(exc)}"})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    """Catch-all — log full error, return clean message to client."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": f"السيرفر فشل بسبب: {str(exc)}"},
    )

# ── Mount API Routers ──────────────────────────────────────────────────────
app.include_router(api_v1_router, prefix="/api/v1")

# ── Health Check ───────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def health_check():
    return {
        "status": "operational",
        "service": "Nabda AI Engine",
        "version": "5.0.0",
        "deployment": "VPS/Docker",
        "endpoints": [
            "POST /api/v1/question-bank",
            "POST /api/v1/mindmap",
            "POST /api/v1/media/video/generate",
            "POST /api/v1/media/podcast/generate",
            "GET  /api/v1/media/jobs/{job_id}",
        ],
    }