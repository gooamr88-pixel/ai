"""
Ruya — FFmpeg Stitching Service
================================
Provides two functions:
  • stitch_video(segments)  → uploads final_video.mp4 to Supabase, returns public URL
  • stitch_audio(turns)     → uploads final_podcast.mp3 to Supabase, returns public URL

Both functions:
  1. Download individual media files to a unique temp directory
  2. Run FFmpeg via subprocess to concat them into a single file
  3. Upload the result to Supabase Storage and return the public URL
  4. Clean up the temp directory unconditionally (even on error)

If Supabase is unavailable the final file is returned as a base64 data URI.
"""

import asyncio
import base64
import logging
import os
import shutil
import uuid
from typing import List, Dict, Any

import httpx

from app.core.config import settings
from app.core.database import supabase

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _download_file(url: str, dest_path: str) -> bool:
    """Download a URL (http/https or data URI) to dest_path. Returns True on success."""
    if not url:
        return False

    try:
        # Handle base64 data URIs (e.g. data:audio/mp3;base64,...)
        if url.startswith("data:"):
            header, b64data = url.split(",", 1)
            file_bytes = base64.b64decode(b64data)
            with open(dest_path, "wb") as f:
                f.write(file_bytes)
            return True

        # Handle regular HTTP URLs
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True

    except Exception as e:
        logger.warning(f"[FFMPEG] Download failed for {url[:80]}: {e}")
        return False


async def _run_ffmpeg(*args: str) -> None:
    """Run ffmpeg with the given args. Raises RuntimeError on non-zero exit."""
    cmd = ["ffmpeg", "-y", *args]
    logger.debug(f"[FFMPEG] Running: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        err = stderr.decode(errors="replace")
        logger.error(f"[FFMPEG] Command failed (code {process.returncode}):\n{err[-800:]}")
        raise RuntimeError(f"FFmpeg failed: {err[-400:]}")


async def _upload_to_supabase(file_path: str, dest_name: str, content_type: str) -> str:
    """Upload file to Supabase bucket and return public URL. Falls back to base64 on error."""
    if not supabase:
        logger.warning("[FFMPEG] Supabase not configured — returning base64 fallback")
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        mime = content_type
        return f"data:{mime};base64,{data}"

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        def _do_upload():
            supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                path=dest_name,
                file=file_bytes,
                file_options={"content-type": content_type},
            )
            return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(dest_name)

        public_url = await asyncio.wait_for(
            asyncio.to_thread(_do_upload), timeout=30
        )
        logger.info(f"[FFMPEG] ✓ Uploaded {dest_name} → {public_url[:80]}")
        return public_url

    except Exception as e:
        logger.error(f"[FFMPEG] Upload failed: {e}")
        # Fallback: return base64 inline
        with open(file_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{content_type};base64,{data}"


# ── stitch_video ──────────────────────────────────────────────────────────────

async def stitch_video(segments: List[Dict[str, Any]]) -> str:
    """
    Download each segment's image + audio, produce one clip per pair with FFmpeg,
    concatenate all clips into final_video.mp4, upload to Supabase and return URL.

    Expected segment keys: image_url (str), audio_url (str)
    """
    tmp_dir = f"/tmp/ruya_video_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)
    logger.info(f"[FFMPEG/VIDEO] Starting stitch for {len(segments)} segments → {tmp_dir}")

    try:
        clip_paths: List[str] = []

        for idx, seg in enumerate(segments):
            img_url   = seg.get("image_url", "")
            audio_url = seg.get("audio_url", "")

            img_path   = os.path.join(tmp_dir, f"img_{idx}.png")
            audio_path = os.path.join(tmp_dir, f"audio_{idx}.mp3")
            clip_path  = os.path.join(tmp_dir, f"clip_{idx}.mp4")

            # Download image
            img_ok = await _download_file(img_url, img_path)
            # Download audio
            audio_ok = await _download_file(audio_url, audio_path)

            if not audio_ok:
                logger.warning(f"[FFMPEG/VIDEO] Segment {idx}: no audio — skipping clip")
                continue

            if img_ok:
                # Full clip: image + audio → MP4
                await _run_ffmpeg(
                    "-loop", "1",
                    "-framerate", "1",
                    "-i", img_path,
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    clip_path,
                )
            else:
                # Audio-only fallback: silent black video + audio
                logger.warning(f"[FFMPEG/VIDEO] Segment {idx}: no image — using audio-only clip")
                await _run_ffmpeg(
                    "-f", "lavfi",
                    "-i", "color=c=black:s=1280x720:r=1",
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    clip_path,
                )

            if os.path.exists(clip_path):
                clip_paths.append(clip_path)

        if not clip_paths:
            raise RuntimeError("No valid clips were produced — cannot stitch video")

        # Write concat list
        concat_txt = os.path.join(tmp_dir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                safe = cp.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        final_path = os.path.join(tmp_dir, "final_video.mp4")
        await _run_ffmpeg(
            "-f", "concat",
            "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            final_path,
        )

        dest_name = f"videos/final_video_{uuid.uuid4().hex}.mp4"
        url = await _upload_to_supabase(final_path, dest_name, "video/mp4")
        logger.info(f"[FFMPEG/VIDEO] ✓ Final video ready: {url[:80]}")
        return url

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"[FFMPEG/VIDEO] Cleaned up temp dir: {tmp_dir}")


# ── stitch_audio ──────────────────────────────────────────────────────────────

async def stitch_audio(turns: List[Dict[str, Any]]) -> str:
    """
    Download each turn's audio MP3, concatenate them with FFmpeg into
    final_podcast.mp3, upload to Supabase and return the public URL.

    Expected turn key: audio_url (str)
    """
    tmp_dir = f"/tmp/ruya_audio_{uuid.uuid4().hex}"
    os.makedirs(tmp_dir, exist_ok=True)
    logger.info(f"[FFMPEG/AUDIO] Starting stitch for {len(turns)} turns → {tmp_dir}")

    try:
        audio_paths: List[str] = []

        for idx, turn in enumerate(turns):
            audio_url  = turn.get("audio_url", "")
            audio_path = os.path.join(tmp_dir, f"turn_{idx}.mp3")

            ok = await _download_file(audio_url, audio_path)
            if ok and os.path.getsize(audio_path) > 0:
                audio_paths.append(audio_path)
            else:
                logger.warning(f"[FFMPEG/AUDIO] Turn {idx}: download failed or empty — skipping")

        if not audio_paths:
            raise RuntimeError("No valid audio files — cannot stitch podcast")

        # Write concat list
        concat_txt = os.path.join(tmp_dir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for ap in audio_paths:
                safe = ap.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        final_path = os.path.join(tmp_dir, "final_podcast.mp3")
        await _run_ffmpeg(
            "-f", "concat",
            "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            final_path,
        )

        dest_name = f"podcasts/final_podcast_{uuid.uuid4().hex}.mp3"
        url = await _upload_to_supabase(final_path, dest_name, "audio/mpeg")
        logger.info(f"[FFMPEG/AUDIO] ✓ Final podcast ready: {url[:80]}")
        return url

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"[FFMPEG/AUDIO] Cleaned up temp dir: {tmp_dir}")
