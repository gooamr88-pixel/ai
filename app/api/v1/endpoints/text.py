"""
Ruya — Text AI Endpoints (Atomic / Vercel-Safe)
==================================================
Handles Quiz, Question Bank, Mind Map, and File Upload.
All endpoints return atomic JSON — NO SSE streaming.

Error handling is done via FastAPI global exception handlers in main.py.
Endpoints raise ValueError / RuntimeError and the handlers convert to
proper HTTP status codes (422 / 503 / 500).
"""

import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from pydantic import BaseModel, Field

from app.services.ai_engine import (
    generate_quiz,
    generate_question_bank,
    generate_mindmap,
)
from app.services.file_service import extract_text_from_file
from app.api.v1.utils import resolve_text_input
from app.core.config import settings
from app.core.limiter import limiter
from app.core.database import supabase

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. UNIFIED EDUCATIONAL PACKAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/generate-educational-package")
@limiter.limit("5/minute")
async def generate_educational_package(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    num_questions: int = Form(5, ge=1, le=30),
    difficulty: str = Form("medium")
):
    """Generate Quiz, Question Bank, and Mind Map concurrently. Returns atomic JSON."""
    
    # 1. Resolve Text (throws 400 if both empty)
    resolved_text = await resolve_text_input(text, file)
    
    # 2. Concurrently generate all three
    quiz_task = generate_quiz(resolved_text, num_questions, difficulty)
    qb_task = generate_question_bank(resolved_text, num_questions, difficulty)
    mindmap_task = generate_mindmap(resolved_text)
    
    quiz_res, qb_res, mindmap_res = await asyncio.gather(quiz_task, qb_task, mindmap_task)
    
    # 3. Save to database (best-effort)
    if supabase:
        try:
            supabase.table("generated_quizzes").insert([
                {
                    "title": quiz_res.title,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "quiz_data": quiz_res.model_dump(exclude={"id"}),
                    "type": "quiz"
                },
                {
                    "title": qb_res.title,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "quiz_data": qb_res.model_dump(exclude={"id"}),
                    "type": "question-bank"
                }
            ]).execute()
            
            supabase.table("generated_mindmaps").insert({
                "mindmap_data": mindmap_res.model_dump(exclude={"id"})
            }).execute()
        except Exception as e:
            logger.error(f"[DB] Insert package failed: {e}")
            
    return {
        "quiz": quiz_res,
        "question_bank": qb_res,
        "mindmap": mindmap_res
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. FILE UPLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/upload")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a file (PDF/Image) and extract text from it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()

    # Validate size (server-side)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = await extract_text_from_file(content, file.filename)

    if not result.get("success"):
        raise HTTPException(status_code=422, detail="Text extraction failed.")

    return {
        "text": result["text"],
        "success": True,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. YOUTUBE TRANSCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import re as _re

# YouTube URL patterns — supports full URLs, short URLs, and embed URLs
_YT_REGEX = _re.compile(
    r"(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
)


class YouTubeRequest(BaseModel):
    url: str = Field(..., min_length=5, max_length=500, description="YouTube video URL")


@router.post("/text/youtube")
@limiter.limit("10/minute")
async def extract_youtube_transcript(request: Request, body: YouTubeRequest):
    """Extract transcript/subtitles from a YouTube video and return as raw text."""
    # ── Step 1: Validate & extract video ID ──
    match = _YT_REGEX.search(body.url.strip())
    if not match:
        raise HTTPException(
            status_code=400,
            detail="رابط يوتيوب غير صالح. تأكد إنك نسخت الرابط صح.",
        )

    video_id = match.group(1)
    logger.info(f"[YOUTUBE] Extracting transcript for video: {video_id}")

    # ── Step 2: Fetch transcript (Arabic first, then English fallback) ──
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try Arabic first, then English, then any available language
        transcript = None
        for lang_code in ["ar", "en"]:
            try:
                transcript = transcript_list.find_transcript([lang_code])
                break
            except Exception:
                continue

        # If no Arabic/English, try any generated/translated transcript
        if transcript is None:
            try:
                generated = transcript_list.find_generated_transcript(["ar", "en"])
                transcript = generated
            except Exception:
                # Last resort: grab whatever is available
                for t in transcript_list:
                    transcript = t
                    break

        if transcript is None:
            raise HTTPException(
                status_code=422,
                detail="مفيش ترجمة أو نص متاح للفيديو ده. جرب فيديو تاني فيه subtitles.",
            )

        # Fetch the actual transcript data
        fetched = transcript.fetch()
        full_text = " ".join([entry.text for entry in fetched])

        if not full_text.strip():
            raise HTTPException(
                status_code=422,
                detail="النص المستخرج فاضي. الفيديو ده ممكن يكون مفيهوش كلام واضح.",
            )

        logger.info(f"[YOUTUBE] ✓ Extracted {len(full_text)} chars from video {video_id}")

        return {
            "text": full_text.strip(),
            "video_id": video_id,
            "success": True,
        }

    except HTTPException:
        raise  # Re-raise our own HTTPExceptions
    except Exception as e:
        logger.error(f"[YOUTUBE] Failed for {video_id}: {e}")
        raise HTTPException(
            status_code=422,
            detail="مقدرناش نستخرج النص من الفيديو ده. ممكن يكون الفيديو خاص أو الترجمة مقفولة.",
        )
