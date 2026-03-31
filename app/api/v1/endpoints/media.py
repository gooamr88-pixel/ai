import logging
import os
import asyncio
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
from app.core.database import supabase
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FFMPEG INTEGRATION UTILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async def merge_media_with_ffmpeg(input_files: List[str], output_file: str):
    """
    FFmpeg integration to merge multiple audio/video segments into a single continuous file.
    Designed for heavy workloads on a VPS.
    Note: input_files are expected to be available locally (downloaded if they are URLs).
    """
    if not input_files:
        return
        
    # Create a simple concat demuxer file
    list_file_path = f"{output_file}_list.txt"
    with open(list_file_path, 'w', encoding='utf-8') as f:
        for file_path in input_files:
            # Escape single quotes in filenames if any
            safe_path = file_path.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
            
    # Run FFmpeg as a subprocess to merge the files continuously
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", output_file,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    
    # Cleanup list file
    if os.path.exists(list_file_path):
        os.remove(list_file_path)
        
    if process.returncode != 0:
        error_msg = stderr.decode()
        logger.error(f"[FFmpeg] Merge failed: {error_msg}")
        raise Exception(f"FFmpeg merge failed: {error_msg}")
        
    logger.info(f"[FFmpeg] Successfully merged {len(input_files)} segments into {output_file}")
    return output_file


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. WHITEBOARD VIDEO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/video/generate", response_model=VideoResponse)
@limiter.limit("5/minute")
async def create_video(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files")
):
    """Generate whiteboard video data (script + audio URLs). Returns atomic JSON."""
    
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
    
    # Force heavy workload / 10-minute video instructions
    final_text = (
        f"{resolved_text}\n\n"
        "STRICT INSTRUCTION: You MUST generate a continuous 10-minute educational video script "
        "based on the massive concatenated text above."
    )

    result = await generate_video_segments(
        final_text,
        num_segments=25,
    )
    
    # [FFmpeg Integration Note] - Ready to call FFmpeg to merge segments into a single file
    # Example placeholder for when actual local files/URLs are downloaded/generated:
    # segment_files = [seg['audio_path'] for seg in result.get('segments', [])] 
    # output_path = f"/tmp/{result.get('video_id', 'output')}_merged.mp4"
    # background_tasks.add_task(merge_media_with_ffmpeg, segment_files, output_path)
    
    return VideoResponse(**result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. PODCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/podcast/generate", response_model=PodcastResponse)
@limiter.limit("5/minute")
async def create_podcast(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload one or multiple PDF files")
):
    """Generate conversational podcast (multi-voice audio URLs). Returns atomic JSON."""
    
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
    
    topic = "نقاش تفصيلي وعميق مدته لا تقل عن 10 دقائق حول محتوى الملفات المرفقة"
    final_text = f"Topic: {topic}\n\n{resolved_text}"

    result = await generate_podcast(
        final_text,
        num_turns=30,
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
            
    # [FFmpeg Integration Note] - Ready to call FFmpeg to merge turns into a single file
    # Example placeholder for background processing:
    # turn_files = [turn['audio_path'] for turn in result.get('turns', [])]
    # output_path = f"/tmp/podcast_{response_obj.id or 'merged'}.mp3"
    # background_tasks.add_task(merge_media_with_ffmpeg, turn_files, output_path)
            
    return response_obj
