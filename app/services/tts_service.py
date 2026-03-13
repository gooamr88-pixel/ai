import base64
import logging
import asyncio
import uuid
import re
from typing import Tuple, List, Dict, Any

import httpx
from elevenlabs.client import ElevenLabs
import google.generativeai as genai

from app.core.config import settings
from app.services.ai_engine import clean_and_parse_json
from app.core.database import supabase
# NOTE: stitch_video imported lazily inside generate_video_segments to avoid
# circular-import issues at module load time.

logger = logging.getLogger(__name__)

# ── Voice Mapping ───────────────────────────────────────────────────────────
VOICES = {
    "default": "onwK4e9ZLuTAKqWW03F9",
    "host":    "onwK4e9ZLuTAKqWW03F9",
    "expert":  "EXAVITQu4vr4xnSDxMaL",
    "guest":   "nPczCjzI2devNBz1zQrb",
}

ELEVENLABS_MODEL = "eleven_multilingual_v2"
TTS_TIMEOUT = 60 # زيادة الوقت لضمان معالجة النصوص الطويلة

# ── TTS Logic ───────────────────────────────────────────────────────────────

def _generate_tts_sync(text: str, voice_id: str) -> bytes:
    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
    audio_iterator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=ELEVENLABS_MODEL,
        output_format="mp3_44100_128",
    )
    return b"".join(audio_iterator)

async def generate_tts_audio(text: str, voice: str = "default") -> Tuple[str, float]:
    logger.info(f"[TTS] Synthesizing {len(text)} chars with voice={voice}")
    try:
        voice_id = VOICES.get(voice, VOICES["default"])
        audio_bytes = await asyncio.wait_for(
            asyncio.to_thread(_generate_tts_sync, text, voice_id),
            timeout=TTS_TIMEOUT,
        )
        
        audio_url = ""
        if supabase:
            file_name = f"audio_{uuid.uuid4().hex}.mp3"
            try:
                def _upload_audio():
                    supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                        path=file_name,
                        file=audio_bytes,
                        file_options={"content-type": "audio/mpeg"}
                    )
                    return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)
                
                audio_url = await asyncio.wait_for(
                    asyncio.to_thread(_upload_audio), timeout=20
                )
            except Exception as e:
                logger.warning(f"[TTS] Storage failed: {e}")

        if not audio_url:
            audio_url = "data:audio/mp3;base64," + base64.b64encode(audio_bytes).decode("utf-8")

        # حسبة دقيقة: الإنسان بيتكلم بمعدل 2 كلمة في الثانية بالعربي التقيل
        word_count = len(text.split())
        duration = word_count / 2.0 
        return audio_url, duration

    except Exception as e:
        logger.error(f"[TTS] Critical Error: {e}")
        raise RuntimeError(f"TTS failed: {e}")

# ── Image Lookup (Lexica.art — fast, reliable, no API key) ──────────────────

async def generate_whiteboard_image(prompt: str) -> str:
    """Search for a relevant image via Lexica.art and upload it to Supabase."""
    if not prompt:
        return ""
    try:
        from urllib.parse import quote
        encoded = quote(prompt, safe="")
        search_url = f"https://lexica.art/api/v1/search?q={encoded}"
        async with httpx.AsyncClient(timeout=30) as client:
            search_resp = await client.get(search_url)
        if search_resp.status_code != 200:
            logger.warning(f"[IMAGE] Lexica search returned {search_resp.status_code}")
            return ""
        img_url = search_resp.json().get("images", [{}])[0].get("src", "")
        if not img_url:
            logger.warning("[IMAGE] Lexica returned no results for prompt.")
            return ""
        # Download the actual image bytes
        async with httpx.AsyncClient(timeout=30) as client:
            img_resp = await client.get(img_url)
        if img_resp.status_code != 200 or not img_resp.content:
            logger.warning(f"[IMAGE] Failed to download image from Lexica: {img_url}")
            return ""
        img_bytes = img_resp.content
        if not supabase:
            return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")
        file_name = f"img_{uuid.uuid4().hex}.jpg"
        def _upload_img():
            supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                path=file_name,
                file=img_bytes,
                file_options={"content-type": "image/jpeg"},
            )
            return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)
        return await asyncio.wait_for(asyncio.to_thread(_upload_img), timeout=20)
    except Exception as e:
        logger.warning(f"[IMAGE] Lexica image lookup failed: {e}")
    return ""


# ── Video Logic (The Forced 5-Minute Refactor) ────────────────────────────────

async def generate_video_segments(text: str, num_segments: int = 10) -> dict:
    """
    Generates exactly 10 segments targeting a ~5-minute video.
    Hard-slices to 10 segments before processing to guarantee the limit.
    """
    num_segments = 10  # Hard cap: 10 segments × ~65 words = ~650 words ≈ 5 min
    logger.info(f"[VIDEO] Generating exactly {num_segments} segments (5-minute target)...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.4,
        },
    )

    prompt = (
        "أنت مُعلم مصري بارع ومُبدع. مهمتك تحويل النص المرفق إلى فيديو تعليمي (Whiteboard Animation).\n\n"
        "قواعد صارمة وإجبارية — لا استثناءات:\n"
        "1. يجب أن تُولِّد 10 شرائح بالضبط (EXACTLY 10 segments). لا أكثر ولا أقل.\n"
        "2. إجمالي عدد الكلمات في جميع حقول 'narration_text' مجتمعةً يجب ألا يتخطى 650 كلمة.\n"
        "3. كل شريحة تحتوي على 55 إلى 65 كلمة في 'narration_text' — لا أقل ولا أكثر.\n"
        "4. هذا الفيديو مُصمَّم بالضبط لمدة 5 دقائق. أي زيادة في الكلمات تُفسد التوقيت.\n\n"
        "قواعد اللغة:\n"
        "- اللغة هي العامية المصرية المثقفة (زي بودكاست علمي).\n"
        "- استخدم تعبيرات مصرية دارجة لجذب الانتباه.\n\n"
        "قواعد الـ TTS:\n"
        "- ممنوع استخدام أي رموز أو ايموجي أو علامات ترقيم غريبة.\n"
        "- اكتب الكلمات الانجليزية بالعربي (مثلاً: 'إيه آي' بدلاً من 'AI').\n\n"
        "Output MUST be valid JSON:\n"
        "{\n"
        '  "title": "عنوان شيق",\n'
        '  "segments": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "title": "عنوان الشريحة",\n'
        '      "bullet_points": ["نقطة 1", "نقطة 2", "نقطة 3"],\n'
        '      "narration_text": "نص الشريحة هنا — بين 55 و65 كلمة بالمصري العامي.",\n'
        '      "voice_id": 1,\n'
        '      "image_prompt": "Clear descriptive English phrase for image search"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"SOURCE TEXT (CONTEXT):\n{text[:18000]}"
    )

    max_retries = 3
    parsed = {"title": "Generated Video", "segments": []}

    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=90,
            )
            raw = response.text.strip()
            parsed = clean_and_parse_json(raw)

            # Guard: accept if we got at least 8 segments with reasonable content
            total_words = sum(len(s.get("narration_text", "").split()) for s in parsed.get("segments", []))
            if len(parsed.get("segments", [])) >= 8 and total_words >= 400:
                logger.info(f"[VIDEO] Success! Total words: {total_words}. Segments: {len(parsed['segments'])}")
                break
            else:
                logger.warning(f"[VIDEO] Attempt {attempt+1} too short ({total_words} words / {len(parsed.get('segments', []))} segs). Retrying...")
        except Exception as e:
            logger.error(f"[VIDEO] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1: raise e

    # ── Hard-slice to exactly 10 segments ────────────────────────────────────
    segments = parsed.get("segments", [])[:10]
    parsed["segments"] = segments
    logger.info(f"[VIDEO] Processing {len(segments)} segments (hard-capped at 10).")

    # Processing images and audio.
    total_duration = 0.0
    for segment in parsed.get("segments", []):
        img_p = segment.get("image_prompt", "")
        segment["image_url"] = await generate_whiteboard_image(img_p) if img_p else ""

        # Sanitise narration_text (same defensive mapping used in podcast_service)
        narration = (
            segment.get("narration_text")
            or segment.get("text")
            or segment.get("content")
            or "..."
        )
        segment["narration_text"] = narration

        voice_key = "host" if segment.get("voice_id", 1) == 1 else "expert"
        try:
            audio_url, duration = await generate_tts_audio(narration, voice=voice_key)
            segment["audio_url"] = audio_url
            segment["duration_seconds"] = duration
            total_duration += duration
        except Exception:
            # TTS failed (e.g. quota exceeded) — estimate duration from word count
            fallback_duration = round(len(narration.split()) / 2.0, 2)
            segment["audio_url"] = ""
            segment["duration_seconds"] = fallback_duration
            total_duration += fallback_duration  # ← CRITICAL: must still accumulate
            logger.warning(f"[VIDEO] TTS failed for segment, estimated {fallback_duration}s from word count")


    parsed["total_duration_seconds"] = total_duration
    logger.info(f"[VIDEO] Per-segment generation done. Duration: {total_duration:.1f}s")

    # ── FFmpeg: stitch all clips into one final_video.mp4 ─────────────────────
    # Lazy import to avoid circular dependency at module load time
    from app.services.ffmpeg_service import stitch_video  # noqa: PLC0415

    logger.info("[VIDEO] Stitching all segments with FFmpeg...")
    try:
        final_video_url = await stitch_video(parsed.get("segments", []))
    except Exception as e:
        logger.error(f"[VIDEO] FFmpeg stitch failed: {e}. Returning empty URL.")
        final_video_url = ""

    logger.info(f"[VIDEO] ✓ Final video URL: {final_video_url[:60]}")
    return {
        "title":                  parsed.get("title", "فيديو تعليمي"),
        "total_duration_seconds": round(total_duration, 2),
        "final_video_url":        final_video_url,
    }
