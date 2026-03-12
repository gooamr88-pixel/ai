import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request, File, UploadFile, BackgroundTasks

from app.schemas.media import (
    VideoResponse,
    PodcastResponse,
)
from app.services.file_service import extract_text_from_file
from app.services.tts_service import generate_video_segments
from app.services.podcast_service import generate_podcast
from app.core.limiter import limiter
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _extract_texts_from_uploads(files: List[UploadFile]) -> str:
    """Validate, read and extract text from one or more PDF uploads."""
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    extracted_texts = []

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

    return "\n\n---\n\n".join(extracted_texts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WHITEBOARD VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post(
    "/video/generate",
    response_model=VideoResponse,
    summary="Generate Whiteboard Video",
    description=(
        "Upload one or more PDF files. Returns a single stitched MP4 URL and total duration. "
        "FFmpeg merges all segments server-side — the client receives only `final_video_url`."
    ),
)
@limiter.limit("5/minute")
async def create_video(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files"),
):
    """Generate a whiteboard video. Returns: title, total_duration_seconds, final_video_url."""
    resolved_text = await _extract_texts_from_uploads(files)

    # Encourage a thorough long-form script from the AI
    final_text = (
        f"{resolved_text}\n\n"
        "STRICT INSTRUCTION: You MUST generate a continuous 10-minute educational video script "
        "based on the massive concatenated text above."
    )

    result = await generate_video_segments(final_text, num_segments=25)

    # result is already shaped: {title, total_duration_seconds, final_video_url}
    return VideoResponse(**result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post(
    "/podcast/generate",
    response_model=PodcastResponse,
    summary="Generate Podcast",
    description=(
        "Upload one or more PDF files. Returns a single stitched MP3 URL and total duration. "
        "FFmpeg merges all turns server-side — the client receives only `final_audio_url`."
    ),
)
@limiter.limit("5/minute")
async def create_podcast(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files"),
):
    """Generate a conversational podcast. Returns: title, total_duration_seconds, final_audio_url."""
    resolved_text = await _extract_texts_from_uploads(files)

    topic = "نقاش تفصيلي وعميق مدته لا تقل عن 10 دقائق حول محتوى الملفات المرفقة"
    final_text = f"Topic: {topic}\n\n{resolved_text}"

    result = await generate_podcast(final_text, num_turns=30, style="educational")

    # result is already shaped: {title, total_duration_seconds, final_audio_url}
    return PodcastResponse(**result)
