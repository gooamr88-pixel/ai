import logging

from fastapi import APIRouter, HTTPException, Request, File, UploadFile

from app.schemas.media import (
    VideoResponse,
    PodcastResponse,
)
from app.services.file_service import extract_text_from_file
from app.services.tts_service import generate_video_segments
from app.services.podcast_service import generate_podcast
from app.core.limiter import limiter
from app.core.database import supabase
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WHITEBOARD VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/generate", response_model=VideoResponse)
@limiter.limit("5/minute")
async def create_video(
    request: Request,
    file: UploadFile = File(..., description="Upload PDF file only")
):
    """Generate whiteboard video data (script + audio URLs). Returns atomic JSON."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="عفواً، مسموح برفع ملفات PDF فقط.")

    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
        )
        
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result_dict = await extract_text_from_file(content, file.filename)
    if not result_dict.get("success"):
        raise HTTPException(status_code=422, detail="Text extraction failed.")
        
    resolved_text = result_dict["text"]

    result = await generate_video_segments(
        resolved_text,
        num_segments=5,
    )
    
    return VideoResponse(**result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/podcast/generate", response_model=PodcastResponse)
@limiter.limit("5/minute")
async def create_podcast(
    request: Request,
    file: UploadFile = File(..., description="Upload PDF file only")
):
    """Generate conversational podcast (multi-voice audio URLs). Returns atomic JSON."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="عفواً، مسموح برفع ملفات PDF فقط.")

    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
        )
        
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result_dict = await extract_text_from_file(content, file.filename)
    if not result_dict.get("success"):
        raise HTTPException(status_code=422, detail="Text extraction failed.")

    resolved_text = result_dict["text"]
    topic = "نقاش حول محتوى الملف المرفق"
    final_text = f"Topic: {topic}\n\n{resolved_text}"

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
