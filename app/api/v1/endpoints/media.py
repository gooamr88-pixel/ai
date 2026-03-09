"""
Ruya — Media API Endpoints (Atomic / Vercel-Safe)
=====================================================
Handles AI Whiteboard Video and Podcast generation.
Returns single atomic JSON responses — NO SSE streaming.

Error handling delegated to FastAPI global exception handlers in main.py.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.media import (
    VideoRequest,
    VideoResponse,
    PodcastRequest,
    PodcastResponse,
)
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
async def create_video(request: Request, body: VideoRequest):
    """Generate whiteboard video data (script + audio URLs). Returns atomic JSON."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = await generate_video_segments(
        body.text,
        num_segments=body.num_segments,
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
async def create_podcast(request: Request, body: PodcastRequest):
    """Generate conversational podcast (multi-voice audio URLs). Returns atomic JSON."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    result = await generate_podcast(
        body.text,
        num_turns=body.num_turns,
        style=body.style,
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
