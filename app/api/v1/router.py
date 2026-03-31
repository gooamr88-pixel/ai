"""
Nabda — API v1 Router Aggregator
================================
Central router that mounts all endpoint modules under /api/v1.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import text, media, question_bank, mindmap_endpoint

api_v1_router = APIRouter()

# ── Question Bank (standalone) ──────────────────────────────────────────────
api_v1_router.include_router(question_bank.router, tags=["Question Bank"])

# ── Mind Map (standalone) ───────────────────────────────────────────────────
api_v1_router.include_router(mindmap_endpoint.router, tags=["Mind Map"])

# ── Legacy Text AI (backward compat — educational package) ──────────────────
api_v1_router.include_router(text.router, tags=["Text AI (Legacy)"])

# ── Media AI (video, podcast, job polling) ──────────────────────────────────
api_v1_router.include_router(media.router, prefix="/media", tags=["Media AI"])
