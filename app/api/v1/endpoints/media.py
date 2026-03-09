"""
Ruya — Media API Endpoints (Atomic / Vercel-Safe)
=====================================================
Handles AI Whiteboard Video and Podcast generation.
Returns single atomic JSON responses — NO SSE streaming.

Error handling delegated to FastAPI global exception handlers in main.py.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Form, File, UploadFile

from app.schemas.media import (
    VideoResponse,
    PodcastResponse,
)
from app.api.v1.utils import resolve_text_input
from app.services.tts_service import generate_video_segments
from app.services.podcast_service import generate_podcast
from app.core.limiter import limiter
from app.core.database import supabase

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WHITEBOARD VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/generate", response_model=VideoResponse)
@limiter.limit("5/minute")
async def create_video(
    request: Request,
    num_segments: int = Form(5, ge=1, le=5),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Generate whiteboard video data (script + audio URLs). Returns atomic JSON."""
    resolved_text = await resolve_text_input(text, file)

    result = await generate_video_segments(
        resolved_text,
        num_segments=num_segments,
    )
    
    # Supabase does not have a generated_videos table in the schema given,
    # but I will add it if it existed.
    # We will just return it directly if no table was requested for Videos.
    return VideoResponse(**result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/podcast/generate", response_model=PodcastResponse)
@limiter.limit("5/minute")
async def create_podcast(
    request: Request,
    topic: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Generate conversational podcast (multi-voice audio URLs). Returns atomic JSON."""
    
    if text or file:
        resolved_text = await resolve_text_input(text, file)
        final_text = f"Topic: {topic}\n\n{resolved_text}"
    else:
        final_text = f"Topic: {topic}"

    result = await generate_podcast(
        final_text,
        num_turns=8,
        style="educational",
    )
    
    response_obj = PodcastResponse(**result)
    
    if supabase:
        try:
            db_res = supabase.table("generated_podcasts").insert({
                "title": response_obj.title,
                "description": response_obj.description,
                "total_duration_seconds": response_obj.total_duration_seconds,
                "podcast_data": response_obj.model_dump(exclude={"id"}),
            }).execute()
            if db_res.data:
                response_obj.id = db_res.data[0]["id"]
        except Exception as e:
            logger.error(f"[DB] Insert podcast failed: {e}")
            
    return response_obj
