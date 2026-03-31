"""
Ruya — Media Endpoints (Video & Podcast)
==========================================
POST /video/generate  → 8-10 min whiteboard video URL
POST /podcast/generate → 7-10 min podcast audio URL
"""

import logging
import os
import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Request, File, UploadFile, BackgroundTasks

from app.schemas.media import VideoResponse, PodcastResponse
from app.services.file_service import extract_text_from_file
from app.services.tts_service import generate_video_segments
from app.services.podcast_service import generate_podcast
from app.core.limiter import limiter
from app.core.database import supabase
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WHITEBOARD VIDEO (8-10 minutes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/generate", response_model=VideoResponse)
@limiter.limit("5/minute")
async def create_video(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files")
):
    """Generate a whiteboard video (8-10 minutes) with Ken Burns effect. Returns URL only."""
    
    extracted_texts = []
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="عفواً، مسموح برفع ملفات PDF فقط.")

        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
            )
            
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file {file.filename} is empty.")

        result_dict = await extract_text_from_file(content, file.filename)
        if not result_dict.get("success"):
            raise HTTPException(status_code=422, detail=f"Text extraction failed for {file.filename}.")
            
        extracted_texts.append(result_dict["text"])

    resolved_text = "\n\n---\n\n".join(extracted_texts)
    
    # Instruction for 8-10 minute video
    final_text = (
        f"{resolved_text}\n\n"
        "STRICT INSTRUCTION: You MUST generate a continuous 8-10 minute educational video script "
        "with 20 segments, each containing 80-100 words of narration and 2 image prompts. "
        "Cover ALL the content in the text above comprehensively."
    )

    result = await generate_video_segments(
        final_text,
        num_segments=20,
    )
    
    # Convert relative /media/ paths to absolute URLs
    if result.get("final_video_url", "").startswith("/media/"):
        base_url = str(request.base_url).rstrip("/")
        result["final_video_url"] = base_url + result["final_video_url"]
    
    return VideoResponse(**result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST (7-10 minutes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/podcast/generate", response_model=PodcastResponse)
@limiter.limit("5/minute")
async def create_podcast(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files")
):
    """Generate a conversational podcast (7-10 minutes). Returns URL only."""
    
    extracted_texts = []
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="عفواً، مسموح برفع ملفات PDF فقط.")

        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
            )
            
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file {file.filename} is empty.")

        result_dict = await extract_text_from_file(content, file.filename)
        if not result_dict.get("success"):
            raise HTTPException(status_code=422, detail=f"Text extraction failed for {file.filename}.")

        extracted_texts.append(result_dict["text"])

    resolved_text = "\n\n---\n\n".join(extracted_texts)
    
    topic = "نقاش تفصيلي وعميق مدته لا تقل عن 8 دقائق حول محتوى الملفات المرفقة"
    final_text = f"Topic: {topic}\n\n{resolved_text}"

    result = await generate_podcast(
        final_text,
        num_turns=35,
        style="educational",
    )
    
    # Convert relative /media/ paths to absolute URLs
    if result.get("final_audio_url", "").startswith("/media/"):
        base_url = str(request.base_url).rstrip("/")
        result["final_audio_url"] = base_url + result["final_audio_url"]
    
    # Save to database (best-effort)
    if supabase:
        try:
            supabase.table("generated_podcasts").insert({
                "title": result.get("title", "بودكاست تعليمي"),
                "total_duration_seconds": result.get("total_duration_seconds", 0),
                "podcast_data": result,
            }).execute()
        except Exception as e:
            logger.error(f"[DB] Insert podcast failed: {e}")
            
    return PodcastResponse(**result)
