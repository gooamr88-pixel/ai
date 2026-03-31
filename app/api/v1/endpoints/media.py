"""
Nabda — Media Endpoints (Video & Podcast) — v2
==================================================
Async endpoints: return 202 Accepted with job_id → client polls GET /jobs/{id}.
Background pipelines handle: script gen → image gen → TTS → FFmpeg → upload.
"""

import os
import logging
from typing import List

from fastapi import APIRouter, HTTPException, Request, File, UploadFile, BackgroundTasks

from app.schemas.jobs import JobResponse, JobCreatedResponse
from app.services.file_service import extract_text_from_file
from app.services.job_manager import job_manager
from app.services.video_pipeline import run_video_pipeline
from app.services.podcast_pipeline import run_podcast_pipeline
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
# 1. VIDEO GENERATION (Async — 202 + Polling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post(
    "/video/generate",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Generate Whiteboard Video",
    description=(
        "Upload one or more PDF files. Queues a background job that generates "
        "a stitched MP4 video. Poll /jobs/{job_id} for progress and the final video_url."
    ),
)
@limiter.limit("3/minute")
async def create_video(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files"),
    num_segments: int = 25,
):
    """
    Queue a video generation job. Returns 202 with job_id.

    Poll GET /api/v1/media/jobs/{job_id} for progress and final result.
    Final result includes video_url (Supabase MP4 link).
    """
    # Validate and extract text
    resolved_text = await _extract_texts_from_uploads(files)

    # Enforce segment limits
    num_segments = min(max(num_segments, 5), settings.VIDEO_MAX_SEGMENTS)

    # Create job
    job_id = await job_manager.create_job("video")

    # Ensure temp dir exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    # Queue background pipeline
    background_tasks.add_task(run_video_pipeline, job_id, resolved_text, num_segments)

    logger.info(f"[VIDEO] Job {job_id} queued: {num_segments} segments")

    return JobCreatedResponse(
        job_id=job_id,
        status="pending",
        message=f"Video generation queued ({num_segments} segments, ~8 minutes target)",
        poll_url=f"/api/v1/media/jobs/{job_id}",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST GENERATION (Async — 202 + Polling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post(
    "/podcast/generate",
    response_model=JobCreatedResponse,
    status_code=202,
    summary="Generate Podcast",
    description=(
        "Upload one or more PDF files. Queues a background job that generates "
        "a stitched MP3 podcast. Poll /jobs/{job_id} for progress and the final podcast_url."
    ),
)
@limiter.limit("3/minute")
async def create_podcast(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files"),
    num_turns: int = 55,
):
    """
    Queue a podcast generation job. Returns 202 with job_id.

    Poll GET /api/v1/media/jobs/{job_id} for progress and final result.
    Final result includes podcast_url (Supabase MP3 link).
    """
    # Validate and extract text
    resolved_text = await _extract_texts_from_uploads(files)

    topic = "نقاش تفصيلي وعميق مدته لا تقل عن 10 دقائق حول محتوى الملفات المرفقة"
    final_text = f"Topic: {topic}\n\n{resolved_text}"

    # Enforce turn limits
    num_turns = min(max(num_turns, 10), settings.PODCAST_MAX_SEGMENTS)

    # Create job
    job_id = await job_manager.create_job("podcast")

    # Ensure temp dir exists
    os.makedirs(settings.TEMP_DIR, exist_ok=True)

    # Queue background pipeline
    background_tasks.add_task(run_podcast_pipeline, job_id, final_text, num_turns)

    logger.info(f"[PODCAST] Job {job_id} queued: {num_turns} turns")

    return JobCreatedResponse(
        job_id=job_id,
        status="pending",
        message=f"Podcast generation queued ({num_turns} turns, ~8 minutes target)",
        poll_url=f"/api/v1/media/jobs/{job_id}",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. JOB POLLING ENDPOINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Poll the status of a video or podcast generation job.

    Status values:
    - pending: Job is queued but not started
    - processing: Job is actively running (check progress %)
    - completed: Job finished. `result` contains the output data.
    - failed: Job encountered an error. `error` contains details.
    """
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return JobResponse(**job)
