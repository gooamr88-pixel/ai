"""
Ruya — FFmpeg Stitching Service (Refactored)
===============================================
Provides:
  • stitch_video(segments) → Ken Burns effect, multi-image per segment, crossfades
  • stitch_audio(turns)    → concat all podcast turn MP3s

Both functions upload to Supabase or save locally, then clean up temp files.
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
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
        if url.startswith("data:"):
            header, b64data = url.split(",", 1)
            file_bytes = base64.b64decode(b64data)
            with open(dest_path, "wb") as f:
                f.write(file_bytes)
            return True

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
    logger.debug(f"[FFMPEG] Running: {' '.join(cmd[:20])}...")

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


MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "media")
os.makedirs(MEDIA_DIR, exist_ok=True)


async def _upload_or_save(file_path: str, dest_name: str, content_type: str) -> str:
    """Upload file to Supabase bucket and return public URL.
    Falls back to saving locally and returning a /media/ path."""

    if supabase:
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
                asyncio.to_thread(_do_upload), timeout=60
            )
            logger.info(f"[FFMPEG] ✓ Uploaded to Supabase: {dest_name} → {public_url[:80]}")
            return public_url

        except Exception as e:
            logger.error(f"[FFMPEG] Supabase upload failed: {e} — falling back to local storage")

    # Fallback: save locally
    local_dest = os.path.join(MEDIA_DIR, dest_name.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_dest), exist_ok=True)

    import shutil as _shutil
    _shutil.copy2(file_path, local_dest)

    local_url = f"/media/{dest_name}"
    logger.info(f"[FFMPEG] ✓ Saved locally: {local_dest} → serving at {local_url}")
    return local_url


# ── Ken Burns Clip Generator ─────────────────────────────────────────────────

async def _make_ken_burns_clip(
    image_path: str,
    audio_path: str,
    output_path: str,
    duration: float,
) -> None:
    """Create a single video clip with Ken Burns (slow zoom) effect on the image."""
    # Calculate zoom parameters
    # Total frames at 25fps
    total_frames = max(int(duration * 25), 25)
    
    await _run_ffmpeg(
        "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-vf", (
            f"scale=1920:1080,zoompan=z='min(zoom+0.0008,1.3)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}:s=1280x720:fps=25"
        ),
        "-c:v", "libx264",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-t", str(duration),
        output_path,
    )


async def _make_multi_image_clip(
    image_paths: List[str],
    audio_path: str,
    output_path: str,
    total_duration: float,
) -> None:
    """Create a clip with multiple images, each showing for equal duration with Ken Burns."""
    if not image_paths:
        return
    
    if len(image_paths) == 1:
        await _make_ken_burns_clip(image_paths[0], audio_path, output_path, total_duration)
        return

    # Each image shows for half the total duration
    dur_per_img = total_duration / len(image_paths)
    tmp_dir = os.path.dirname(output_path)
    sub_clips = []

    for i, img_path in enumerate(image_paths):
        sub_clip = os.path.join(tmp_dir, f"sub_{os.path.basename(output_path)}_{i}.mp4")
        total_frames = max(int(dur_per_img * 25), 25)
        
        # Create silent video with Ken Burns for each image
        await _run_ffmpeg(
            "-loop", "1",
            "-i", img_path,
            "-vf", (
                f"scale=1920:1080,zoompan=z='min(zoom+0.0008,1.3)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={total_frames}:s=1280x720:fps=25"
            ),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-an",
            "-t", str(dur_per_img),
            sub_clip,
        )
        if os.path.exists(sub_clip):
            sub_clips.append(sub_clip)

    if not sub_clips:
        # Fallback to single image
        await _make_ken_burns_clip(image_paths[0], audio_path, output_path, total_duration)
        return

    # Concat the sub-clips
    concat_txt = os.path.join(tmp_dir, f"sub_concat_{os.path.basename(output_path)}.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for sc in sub_clips:
            safe = sc.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    # Merge sub-clips into one video, then add audio
    merged_video = os.path.join(tmp_dir, f"merged_{os.path.basename(output_path)}")
    await _run_ffmpeg(
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        merged_video,
    )

    # Add audio to the merged video
    await _run_ffmpeg(
        "-i", merged_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    )

    # Cleanup sub-clips
    for sc in sub_clips:
        if os.path.exists(sc):
            os.remove(sc)
    if os.path.exists(concat_txt):
        os.remove(concat_txt)
    if os.path.exists(merged_video):
        os.remove(merged_video)


# ── stitch_video ──────────────────────────────────────────────────────────────

async def stitch_video(segments: List[Dict[str, Any]]) -> str:
    """
    Download each segment's images + audio, produce Ken Burns clips,
    concatenate with crossfades into final_video.mp4.

    Expected segment keys: image_urls (List[str]), audio_url (str), duration_seconds (float)
    """
    tmp_dir = tempfile.mkdtemp(prefix="ruya_video_")
    logger.info(f"[FFMPEG/VIDEO] Starting stitch for {len(segments)} segments → {tmp_dir}")

    try:
        clip_paths: List[str] = []

        for idx, seg in enumerate(segments):
            # Get image URLs (support both image_urls list and single image_url)
            image_urls = seg.get("image_urls", [])
            if not image_urls:
                single_url = seg.get("image_url", "")
                image_urls = [single_url] if single_url else []

            audio_url = seg.get("audio_url", "")
            duration = seg.get("duration_seconds", 30.0)
            clip_path = os.path.join(tmp_dir, f"clip_{idx}.mp4")

            # Download audio
            audio_path = os.path.join(tmp_dir, f"audio_{idx}.mp3")
            audio_ok = await _download_file(audio_url, audio_path)

            if not audio_ok:
                logger.warning(f"[FFMPEG/VIDEO] Segment {idx}: no audio — skipping")
                continue

            # Download images
            img_paths = []
            for img_idx, img_url in enumerate(image_urls):
                img_path = os.path.join(tmp_dir, f"img_{idx}_{img_idx}.png")
                if await _download_file(img_url, img_path):
                    img_paths.append(img_path)

            if img_paths:
                # Multi-image Ken Burns clip
                await _make_multi_image_clip(img_paths, audio_path, clip_path, duration)
            else:
                # Audio-only fallback: black background
                logger.warning(f"[FFMPEG/VIDEO] Segment {idx}: no images — audio-only clip")
                await _run_ffmpeg(
                    "-f", "lavfi",
                    "-i", "color=c=black:s=1280x720:r=25",
                    "-i", audio_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
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

        # ── Concatenate all clips ────────────────────────────────────────────
        # Simple concat (crossfade adds too much complexity for many segments)
        concat_txt = os.path.join(tmp_dir, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                safe = cp.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        final_path = os.path.join(tmp_dir, "final_video.mp4")
        
        # Re-encode during concat to ensure uniform format
        await _run_ffmpeg(
            "-f", "concat",
            "-safe", "0",
            "-i", concat_txt,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            final_path,
        )

        dest_name = f"videos/final_video_{uuid.uuid4().hex}.mp4"
        url = await _upload_or_save(final_path, dest_name, "video/mp4")
        logger.info(f"[FFMPEG/VIDEO] ✓ Final video ready: {url[:80]}")
        return url

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"[FFMPEG/VIDEO] Cleaned up temp dir: {tmp_dir}")


# ── stitch_audio ──────────────────────────────────────────────────────────────

async def stitch_audio(turns: List[Dict[str, Any]]) -> str:
    """
    Download each turn's audio MP3, concatenate with FFmpeg into
    final_podcast.mp3, upload to Supabase and return the public URL.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ruya_audio_")
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
                logger.warning(f"[FFMPEG/AUDIO] Turn {idx}: download failed — skipping")

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
        url = await _upload_or_save(final_path, dest_name, "audio/mpeg")
        logger.info(f"[FFMPEG/AUDIO] ✓ Final podcast ready: {url[:80]}")
        return url

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f"[FFMPEG/AUDIO] Cleaned up temp dir: {tmp_dir}")
