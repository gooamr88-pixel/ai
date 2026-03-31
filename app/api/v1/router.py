"""
Ruya — API v1 Router Aggregator
================================
Central router that mounts all endpoint modules under /api/v1.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import text, media

api_v1_router = APIRouter()

# ── Text AI Endpoints (quiz, question-bank, mindmap, upload) ────────────
api_v1_router.include_router(text.router, tags=["Text AI"])

# ── Media AI Endpoints (video, podcast) ─────────────────────────────────
api_v1_router.include_router(media.router, prefix="/media", tags=["Media AI"])
