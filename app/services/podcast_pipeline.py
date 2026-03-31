"""
Nabda — Podcast Generation Pipeline
=======================================
End-to-end: LLM Script → edge-TTS (batched) → FFmpeg Concat → Upload.
Runs as a FastAPI BackgroundTask, updates job_manager with progress.
"""

import os
import shutil
import logging
import asyncio
import tempfile

from app.core.config import settings
from app.services.job_manager import job_manager
from app.services.ai_engine import generate_podcast_script
from app.services.tts_provider import generate_tts_audio, get_audio_duration_ffprobe
from app.services.video_pipeline import concat_audio_files, upload_to_supabase

logger = logging.getLogger(__name__)


async def run_podcast_pipeline(job_id: str, text: str, num_turns: int = 55):
    """
    Full podcast generation pipeline. Runs as a background task.
    Updates job_manager with progress at each stage.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="nabda_podcast_", dir=settings.TEMP_DIR)
        os.makedirs(temp_dir, exist_ok=True)

        # ── Stage 1: Generate podcast script (5% → 20%) ──
        await job_manager.update_job(
            job_id, status="processing", progress=5,
            message="جاري إنشاء سيناريو البودكاست..."
        )

        script = await generate_podcast_script(text, num_turns=num_turns)
        turns = script.get("turns", [])

        await job_manager.update_job(job_id, progress=20,
            message=f"تم إنشاء {len(turns)} جولة حوار. جاري إنشاء الصوت...")

        # ── Stage 2: Generate TTS audio — batched (20% → 85%) ──
        BATCH_SIZE = settings.PODCAST_TTS_BATCH_SIZE
        audio_paths = []
        turn_durations = []

        for batch_start in range(0, len(turns), BATCH_SIZE):
            batch = turns[batch_start:batch_start + BATCH_SIZE]

            async def _process_turn(turn: dict, index: int) -> float:
                speaker = turn.get("speaker", "Host1")
                if speaker == "Host1":
                    voice_key = "host"
                elif speaker == "Host2":
                    voice_key = "expert"
                else:
                    voice_key = "guest"

                try:
                    audio_bytes, _ = await generate_tts_audio(turn["text"], voice=voice_key)
                except Exception as e:
                    logger.warning(f"[PODCAST] TTS failed for turn {turn.get('id')}: {e}")
                    audio_bytes = b""

                audio_path = os.path.join(temp_dir, f"turn_{index:03d}.mp3")
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)

                duration = await get_audio_duration_ffprobe(audio_path)
                turn["duration_seconds"] = duration
                audio_paths.append(audio_path)
                return duration

            # Process batch concurrently
            batch_indices = range(batch_start, batch_start + len(batch))
            batch_durations = await asyncio.gather(
                *[_process_turn(t, i) for t, i in zip(batch, batch_indices)]
            )
            turn_durations.extend(batch_durations)

            # Small delay between batches to respect rate limits
            if batch_start + BATCH_SIZE < len(turns):
                await asyncio.sleep(0.5)

            # Update progress
            progress = 20 + int(((batch_start + len(batch)) / len(turns)) * 65)
            await job_manager.update_job(job_id, progress=progress,
                message=f"صوت {batch_start + len(batch)}/{len(turns)}...")

        await job_manager.update_job(job_id, progress=85,
            message="جاري دمج الصوت...")

        # ── Stage 3: Concatenate all audio into single MP3 (85% → 92%) ──
        output_audio = os.path.join(temp_dir, "podcast_final.mp3")
        await concat_audio_files(audio_paths, output_audio)

        # Get precise total duration
        total_duration = await get_audio_duration_ffprobe(output_audio)

        await job_manager.update_job(job_id, progress=92,
            message="جاري رفع البودكاست...")

        # ── Stage 4: Upload to Supabase (92% → 100%) ──
        podcast_url = await upload_to_supabase(output_audio, "audio/mpeg")

        # ── Build result ──
        result = {
            "title": script.get("title", ""),
            "description": script.get("description", ""),
            "speakers": script.get("speakers", []),
            "podcast_url": podcast_url,
            "total_duration_seconds": round(total_duration, 1),
            "total_turns": len(turns),
            "turns": [
                {
                    "id": turn.get("id", i + 1),
                    "speaker": turn.get("speaker", ""),
                    "text": turn.get("text", ""),
                    "duration_seconds": round(turn.get("duration_seconds", 0), 1),
                }
                for i, turn in enumerate(turns)
            ],
        }

        await job_manager.complete_job(job_id, result)
        logger.info(
            f"[PODCAST] ✓ Pipeline complete: {total_duration:.0f}s, "
            f"{len(turns)} turns"
        )

    except Exception as e:
        logger.error(f"[PODCAST] Pipeline failed for job {job_id}: {e}", exc_info=True)
        await job_manager.fail_job(job_id, str(e))

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
