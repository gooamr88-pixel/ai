"""
Ruya — Media Endpoints (Video & Podcast) — Optimized v3
==========================================================
POST /video/generate   → 3-8 min whiteboard video (dynamic from PDF size)
POST /podcast/generate → 3-8 min podcast audio   (dynamic from PDF size)

Smart Config: segment/turn count automatically scales with PDF text size.
"""

import logging
import asyncio
from typing import List

from fastapi import APIRouter, Request, File, UploadFile

from app.schemas.media import VideoResponse, PodcastResponse
from app.services.tts_service import generate_video_segments
from app.services.podcast_service import generate_podcast
from app.services.smart_config import calculate_smart_config
from app.api.v1.utils import resolve_multi_pdf_input
from app.core.limiter import limiter
from app.core.database import supabase

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WHITEBOARD VIDEO (dynamic: 3-8 minutes based on PDF size)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/generate", response_model=VideoResponse)
@limiter.limit("5/minute")
async def create_video(
    request: Request,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files")
):
    """Generate a whiteboard video with Ken Burns effect. Duration scales with PDF size."""

    resolved_text = await resolve_multi_pdf_input(files)

    # Calculate smart config based on extracted text size
    smart_cfg = calculate_smart_config(resolved_text)
    logger.info(
        f"[MEDIA/VIDEO] Smart config: {smart_cfg.tier_name} tier → "
        f"{smart_cfg.video_segments} segments, {smart_cfg.num_chunks} chunks, "
        f"~{smart_cfg.estimated_duration_min}-{smart_cfg.estimated_duration_max} min"
    )

    final_text = (
        f"{resolved_text}\n\n"
        f"STRICT INSTRUCTION: You MUST generate a continuous educational video script "
        f"with {smart_cfg.video_segments} segments, each containing 80-100 words of narration "
        f"and 1 detailed image prompt. Cover ALL the content in the text above comprehensively."
    )

    result = await generate_video_segments(
        final_text,
        smart_cfg=smart_cfg,
    )

    # Convert relative /media/ paths to absolute URLs
    if result.get("final_video_url", "").startswith("/media/"):
        base_url = str(request.base_url).rstrip("/")
        result["final_video_url"] = base_url + result["final_video_url"]

    return VideoResponse(**result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST (dynamic: 3-8 minutes based on PDF size)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/podcast/generate", response_model=PodcastResponse)
@limiter.limit("5/minute")
async def create_podcast(
    request: Request,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files")
):
    """Generate a conversational podcast. Duration scales with PDF size."""

    resolved_text = await resolve_multi_pdf_input(files)

    # Calculate smart config based on extracted text size
    smart_cfg = calculate_smart_config(resolved_text)
    logger.info(
        f"[MEDIA/PODCAST] Smart config: {smart_cfg.tier_name} tier → "
        f"{smart_cfg.podcast_turns} turns, {smart_cfg.num_chunks} chunks, "
        f"~{smart_cfg.estimated_duration_min}-{smart_cfg.estimated_duration_max} min"
    )

    topic = "نقاش تفصيلي وعميق حول محتوى الملفات المرفقة"
    final_text = f"Topic: {topic}\n\n{resolved_text}"

    result = await generate_podcast(
        final_text,
        style="educational",
        smart_cfg=smart_cfg,
    )

    # Convert relative /media/ paths to absolute URLs
    if result.get("final_audio_url", "").startswith("/media/"):
        base_url = str(request.base_url).rstrip("/")
        result["final_audio_url"] = base_url + result["final_audio_url"]

    # Save to database (best-effort)
    if supabase:
        try:
            await asyncio.to_thread(
                lambda: supabase.table("generated_podcasts").insert({
                    "title": result.get("title", "بودكاست تعليمي"),
                    "total_duration_seconds": result.get("total_duration_seconds", 0),
                    "podcast_data": result,
                }).execute()
            )
        except Exception as e:
            logger.error(f"[DB] Insert podcast failed: {e}")

    return PodcastResponse(**result)
