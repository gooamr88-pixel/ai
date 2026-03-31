"""
Nabda — Video Generation Pipeline
=====================================
End-to-end: LLM Script → Replicate Images → edge-tTS → FFmpeg Stitch → Upload.
Runs as a FastAPI BackgroundTask, updates job_manager with progress.
"""

import os
import shutil
import logging
import asyncio
import tempfile
import uuid
from typing import List

import httpx

from app.core.config import settings
from app.core.database import supabase
from app.services.job_manager import job_manager
from app.services.ai_engine import generate_video_script
from app.services.tts_provider import generate_tts_audio, get_audio_duration_ffprobe
from app.services.image_service import generate_images_batch, download_image

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUPABASE UPLOAD HELPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def upload_to_supabase(file_path: str, content_type: str) -> str:
    """Upload a local file to Supabase Storage and return its public URL."""
    if not supabase:
        logger.warning("[UPLOAD] Supabase not configured, skipping upload")
        return ""

    ext = os.path.splitext(file_path)[1]
    file_name = f"{uuid.uuid4().hex}{ext}"

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        def _upload():
            supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                path=file_name,
                file=file_bytes,
                file_options={"content-type": content_type},
            )
            return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)

        url = await asyncio.wait_for(asyncio.to_thread(_upload), timeout=30)
        logger.info(f"[UPLOAD] ✓ Uploaded {file_name} → {url[:80]}...")
        return url
    except Exception as e:
        logger.error(f"[UPLOAD] Failed: {e}")
        return ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FFMPEG STITCHING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def concat_audio_files(audio_paths: List[str], output_path: str):
    """Concatenate multiple MP3 files into one using FFmpeg concat demuxer."""
    list_file = output_path + "_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for path in audio_paths:
            safe = path.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    os.remove(list_file)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio concat failed: {stderr.decode()[:500]}")

    logger.info(f"[FFMPEG] ✓ Concatenated {len(audio_paths)} audio files")


async def stitch_video_ffmpeg(
    image_entries: List[dict],  # [{"path": str, "duration": float}]
    audio_path: str,
    output_path: str,
):
    """
    Create video from image slideshow + audio using FFmpeg concat demuxer.
    image_entries: [{"path": "/tmp/img_001.png", "duration": 8.5}, ...]
    """
    list_file = output_path + "_images.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for i, entry in enumerate(image_entries):
            safe = entry["path"].replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
            f.write(f"duration {entry['duration']:.3f}\n")
        # Repeat last image to avoid black frame at end
        if image_entries:
            last_safe = image_entries[-1]["path"].replace("'", "'\\''")
            f.write(f"file '{last_safe}'\n")

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-i", audio_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,"
               "pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    os.remove(list_file)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg video stitch failed: {stderr.decode()[:500]}")

    logger.info(f"[FFMPEG] ✓ Stitched video: {output_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLACEHOLDER IMAGE (when Replicate fails or no token)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_placeholder_image(path: str, text: str = ""):
    """Create a simple white placeholder image when generation fails."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw centered gray text
        if text:
            try:
                draw.text((640, 360), text[:50], fill=(180, 180, 180), anchor="mm")
            except Exception:
                pass
        img.save(path, "PNG")
    except ImportError:
        # Minimal 1x1 white PNG if Pillow not available
        import struct, zlib
        def _min_png(w, h):
            raw = b""
            for _ in range(h):
                raw += b"\x00" + b"\xff\xff\xff" * w
            return (
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR" + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"
                + b"\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00IDAT" + zlib.compress(raw)
                + b"\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
        with open(path, "wb") as f:
            f.write(_min_png(1280, 720))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_video_pipeline(job_id: str, text: str, num_segments: int = 25):
    """
    Full video generation pipeline. Runs as a background task.
    Updates job_manager with progress at each stage.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="nabda_video_", dir=settings.TEMP_DIR)
        os.makedirs(temp_dir, exist_ok=True)

        # ── Stage 1: Generate script (5% → 15%) ──
        await job_manager.update_job(
            job_id, status="processing", progress=5,
            message="جاري إنشاء السيناريو..."
        )
        script = await generate_video_script(text, num_segments=num_segments)
        segments = script.get("segments", [])

        await job_manager.update_job(job_id, progress=15,
            message=f"تم إنشاء سيناريو من {len(segments)} مقطع. جاري إنشاء الصور...")

        # ── Stage 2: Collect all image prompts & generate in parallel (15% → 45%) ──
        all_prompts = []
        prompt_map = []  # (seg_idx, scene_idx)
        for si, seg in enumerate(segments):
            scenes = seg.get("image_scenes", [])
            if not scenes:
                # Fallback: create single scene from segment
                scenes = [{"timestamp_offset": 0, "image_prompt": seg.get("image_prompt", f"educational whiteboard diagram {si}")}]
                seg["image_scenes"] = scenes
            for sci, scene in enumerate(scenes):
                all_prompts.append(scene.get("image_prompt", "whiteboard educational diagram"))
                prompt_map.append((si, sci))

        image_urls = await generate_images_batch(all_prompts, max_concurrent=settings.IMAGE_MAX_CONCURRENT)

        # Assign URLs back to segments
        for idx, (si, sci) in enumerate(prompt_map):
            url = image_urls[idx]
            segments[si]["image_scenes"][sci]["image_url"] = url or ""

        await job_manager.update_job(job_id, progress=45,
            message="تم إنشاء الصور. جاري إنشاء الصوت...")

        # ── Stage 3: Generate TTS audio for each segment (45% → 75%) ──
        audio_paths = []
        segment_durations = []

        for i, seg in enumerate(segments):
            voice = "host" if seg.get("voice_id", 1) == 1 else "expert"
            try:
                audio_bytes, _ = await generate_tts_audio(seg["narration_text"], voice=voice)
            except Exception as e:
                logger.warning(f"[VIDEO] TTS failed for segment {i}: {e}")
                audio_bytes = b""

            audio_path = os.path.join(temp_dir, f"audio_{i:03d}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Get precise duration via ffprobe
            duration = await get_audio_duration_ffprobe(audio_path)
            seg["duration_seconds"] = duration
            segment_durations.append(duration)
            audio_paths.append(audio_path)

            progress = 45 + int((i / len(segments)) * 30)
            await job_manager.update_job(job_id, progress=progress,
                message=f"صوت {i + 1}/{len(segments)}...")

        await job_manager.update_job(job_id, progress=75,
            message="تم إنشاء الصوت. جاري تجميع الفيديو...")

        # ── Stage 4: Download images to local files (75% → 80%) ──
        image_entries = []  # {"path": str, "duration": float}
        img_counter = 0

        for si, seg in enumerate(segments):
            seg_duration = segment_durations[si]
            scenes = seg.get("image_scenes", [])
            num_scenes = max(len(scenes), 1)
            per_image_duration = seg_duration / num_scenes

            for sci, scene in enumerate(scenes):
                img_path = os.path.join(temp_dir, f"img_{img_counter:04d}.png")
                url = scene.get("image_url", "")

                if url and url.startswith("http"):
                    success = await download_image(url, img_path)
                    if not success:
                        create_placeholder_image(img_path, seg.get("title", ""))
                else:
                    create_placeholder_image(img_path, seg.get("title", ""))

                image_entries.append({"path": img_path, "duration": per_image_duration})
                img_counter += 1

        await job_manager.update_job(job_id, progress=80,
            message="جاري دمج الفيديو بالـ FFmpeg...")

        # ── Stage 5: Concatenate audio ──
        combined_audio = os.path.join(temp_dir, "combined_audio.mp3")
        await concat_audio_files(audio_paths, combined_audio)

        # ── Stage 6: FFmpeg stitch (80% → 90%) ──
        output_video = os.path.join(temp_dir, "output.mp4")
        await stitch_video_ffmpeg(image_entries, combined_audio, output_video)

        await job_manager.update_job(job_id, progress=90,
            message="جاري رفع الفيديو...")

        # ── Stage 7: Upload to Supabase (90% → 100%) ──
        video_url = await upload_to_supabase(output_video, "video/mp4")

        total_duration = sum(segment_durations)
        total_images = len(image_entries)

        result = {
            "title": script.get("title", ""),
            "video_url": video_url,
            "total_duration_seconds": round(total_duration, 1),
            "total_segments": len(segments),
            "total_images": total_images,
            "segments": [
                {
                    "id": seg.get("id", i + 1),
                    "title": seg.get("title", ""),
                    "duration_seconds": round(seg.get("duration_seconds", 0), 1),
                    "narration_text": seg.get("narration_text", ""),
                }
                for i, seg in enumerate(segments)
            ],
        }

        await job_manager.complete_job(job_id, result)
        logger.info(
            f"[VIDEO] ✓ Pipeline complete: {total_duration:.0f}s, "
            f"{len(segments)} segments, {total_images} images"
        )

    except Exception as e:
        logger.error(f"[VIDEO] Pipeline failed for job {job_id}: {e}", exc_info=True)
        await job_manager.fail_job(job_id, str(e))

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
