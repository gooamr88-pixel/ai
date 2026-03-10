"""
Ruya — TTS Service (ElevenLabs / Vercel-Safe)
================================================
Production-grade Text-to-Speech using the ElevenLabs Python SDK.
Generates Base64-encoded MP3 audio from text in-memory.

Architecture:
  - NO disk writes (Vercel read-only filesystem)
  - NO redundant Gemini calls (narration text comes pre-written)
  - Timeouts on ALL external calls via asyncio.wait_for()
  - ElevenLabs handles text normalization natively (no regex sanitizer needed)

Voices (ElevenLabs Arabic — Egyptian accent):
  - Haytham  (Male Egyptian Podcaster — host / video narrator)
  - Hoda     (Female Egyptian — expert / co-host / video narrator)
  - Amr      (Male Egyptian — guest voice)
"""

import base64
import logging
import asyncio
import uuid
from typing import Tuple

import httpx

from elevenlabs.client import ElevenLabs

import google.generativeai as genai

from app.core.config import settings
from app.services.ai_engine import clean_and_parse_json
from app.core.database import supabase

logger = logging.getLogger(__name__)


# ── Voice Mapping (ElevenLabs Arabic Voice IDs) ──────────────────────────────

# ── Voice Mapping (ElevenLabs Default Free Voices) ───────────────────────────
# ── Voice Mapping (ElevenLabs Premium Free Voices) ───────────────────────────

VOICES = {
    "default": "onwK4e9ZLuTAKqWW03F9",   # Daniel (Deep & Professional)
    "host":    "onwK4e9ZLuTAKqWW03F9",   # Daniel (Male Host)
    "expert":  "EXAVITQu4vr4xnSDxMaL",   # Sarah (Female Expert)
    "guest":   "nPczCjzI2devNBz1zQrb",   # Brian (Male Guest)
}

# ElevenLabs model — best quality for Arabic with accent accuracy
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# ── TTS Timeout (seconds) ────────────────────────────────────────────────────
TTS_TIMEOUT = 30  # ElevenLabs is slightly slower than edge-tts, but much higher quality


def _generate_tts_sync(text: str, voice_id: str) -> bytes:
    """
    Synchronous ElevenLabs TTS call.
    Wrapped in asyncio.to_thread() by the async caller.
    """
    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    audio_iterator = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=ELEVENLABS_MODEL,
        output_format="mp3_44100_128",
    )

    # Collect all chunks into bytes
    audio_bytes = b"".join(audio_iterator)
    return audio_bytes


async def generate_tts_audio(text: str, voice: str = "default") -> Tuple[str, float]:
    """
    Generate TTS audio from text using ElevenLabs, upload to Supabase, and return URL.

    Args:
        text: The narration text to synthesize (already written as spoken script).
        voice: Voice key — "default", "host", "expert", "guest".

    Returns:
        Tuple of (audio_url_or_data_uri, estimated_duration_seconds)
    """
    logger.info(f"[TTS] Generating ElevenLabs audio for {len(text)} chars, voice={voice}")

    try:
        voice_id = VOICES.get(voice, VOICES["default"])

        # Run sync SDK call in a thread to avoid blocking the event loop
        audio_bytes = await asyncio.wait_for(
            asyncio.to_thread(_generate_tts_sync, text, voice_id),
            timeout=TTS_TIMEOUT,
        )

        if len(audio_bytes) == 0:
            raise RuntimeError("ElevenLabs returned empty audio")

        audio_url = ""
        # Upload to Supabase
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
                    asyncio.to_thread(_upload_audio),
                    timeout=15,
                )
                logger.info(f"[TTS] ✓ Uploaded to Supabase: {audio_url}")
            except Exception as e:
                logger.warning(f"[TTS] Supabase upload failed, falling back to base64: {e}")

        # Fallback to data URI base64 if upload fails or is unconfigured
        if not audio_url:
            audio_url = "data:audio/mp3;base64," + base64.b64encode(audio_bytes).decode("utf-8")

        # Estimate duration (~2.5 words/sec for Arabic narration)
        word_count = len(text.split())
        estimated_duration = word_count / 2.5

        logger.info(
            f"[TTS] ✓ Audio ready: {len(audio_bytes)} bytes, "
            f"~{estimated_duration:.1f}s, voice={voice_id}"
        )

        return audio_url, estimated_duration

    except asyncio.TimeoutError:
        logger.error(f"[TTS] Timeout after {TTS_TIMEOUT}s for voice={voice}")
        raise RuntimeError(f"TTS generation timed out after {TTS_TIMEOUT}s")
    except Exception as e:
        logger.error(f"[TTS] Failed: {e}")
        raise RuntimeError(f"TTS generation failed: {e}")


# ── Hugging Face Image Generation ────────────────────────────────────────────

HF_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_IMAGE_TIMEOUT = 30  # seconds


async def generate_whiteboard_image(prompt: str) -> str:
    """
    Generate a whiteboard-style image via Hugging Face Inference API.
    Uploads to Supabase and returns URL, or base64 data URI fallback.
    """
    if not settings.HF_API_TOKEN:
        logger.warning("[IMAGE] HF_API_TOKEN not set — skipping image generation")
        return ""

    try:
        async with httpx.AsyncClient(timeout=HF_IMAGE_TIMEOUT) as client:
            response = await client.post(
                f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}",
                headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
                json={"inputs": prompt},
            )

        if response.status_code == 200:
            image_url = ""
            if supabase:
                file_name = f"img_{uuid.uuid4().hex}.png"
                try:
                    def _upload_img():
                        supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                            path=file_name,
                            file=response.content,
                            file_options={"content-type": "image/png"}
                        )
                        return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)
                        
                    image_url = await asyncio.wait_for(
                        asyncio.to_thread(_upload_img),
                        timeout=15,
                    )
                    logger.info(f"[IMAGE] ✓ Uploaded image to Supabase")
                except Exception as e:
                    logger.warning(f"[IMAGE] Supabase upload failed: {e}")
            
            if not image_url:
                image_url = "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")
                
            return image_url
        else:
            logger.warning(f"[IMAGE] HF API returned {response.status_code}: {response.text[:200]}")
            return ""

    except Exception as e:
        logger.warning(f"[IMAGE] Generation failed: {e}")
        return ""


async def generate_video_segments(text: str, num_segments: int = 15) -> dict:
    """
    Generate whiteboard video structure: script + slides + images + narration audio.
    Returns atomic JSON for client-side SmartPlayer rendering.

    Flow:
      1. Gemini generates structured JSON with narration_text per slide
      2. HF Inference API generates whiteboard images from image_prompt
      3. ElevenLabs converts each narration_text to audio and uploads to Supabase
      4. Return complete JSON manifest in one response
    """
    # Force 15 - 20 segment boundaries (Override settings clamping which may be small)
    num_segments = max(15, min(num_segments, 20))
    logger.info(f"[VIDEO] Generating {num_segments} segments...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.3,
            "response_schema": {"type": "object", "properties": {"title": {"type": "string"}, "segments": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"}, "title": {"type": "string"}, "bullet_points": {"type": "array", "items": {"type": "string"}}, "narration_text": {"type": "string"}, "voice_id": {"type": "integer"}, "image_prompt": {"type": "string"}}, "required": ["id", "title", "bullet_points", "narration_text", "voice_id", "image_prompt"]}}}, "required": ["title", "segments"]}
        },
    )

    prompt = (
        "أنت مُعلم مصري متخصص في تبسيط المفاهيم العلمية بأسلوب شيق وممتع.\n"
        f"Create a whiteboard video lesson with exactly {num_segments} slides from the following text.\n\n"
        "DURATION & LENGTH RULES (CRITICAL):\n"
        "- You MUST generate EXACTLY 15 to 20 segments.\n"
        "- Each segment's narration_text MUST be between 40 to 60 words long.\n"
        "- Total word count across all segments MUST be a minimum of 700 words.\n\n"
        "LANGUAGE RULES (CRITICAL):\n"
        "- ALL output MUST be in heavy Egyptian Colloquial Arabic (اللهجة المصرية العامية الدارجة).\n"
        "- ALL dialogue MUST use Egyptian phrasing, idioms, and expressions.\n"
        "- titles, bullet_points, and narration_text MUST all be in natural Egyptian dialect.\n"
        "- If the source text is in English or any other language, TRANSLATE and adapt to Egyptian Arabic.\n"
        "- Use casual, engaging Egyptian expressions (يعني، خلينا نقول، تعالوا نشوف، الموضوع ده، ببساطة كده).\n\n"
        "TTS SCRIPTING RULES (CRITICAL — narration_text will be read aloud by a TTS engine):\n"
        "- NEVER use emojis, markdown, asterisks (*), underscores (_), hashtags (#), or ANY special symbols.\n"
        "- NEVER use bracket stage directions like [يضحك] or (يبتسم) — write ONLY spoken words.\n"
        "- USE frequent punctuation: commas (،) and periods (.) to force the TTS to take natural breathing pauses.\n"
        "- SPELL OUT English acronyms or terms using Arabic letters phonetically. "
        "For example: write 'إتش تي إم إل' instead of 'HTML', 'فيسبوك' instead of 'Facebook', "
        "'إيه آي' instead of 'AI', 'بايثون' instead of 'Python'.\n"
        "- Write in a conversational, calm pace. Short sentences. Natural flow.\n\n"
        "VOICE ASSIGNMENT:\n"
        "- Alternate narration between voice_id 1 (male Egyptian narrator) and voice_id 2 (female Egyptian narrator).\n"
        "- Odd-numbered segments get voice_id 1, even-numbered segments get voice_id 2.\n\n"
        "Output MUST be valid JSON matching this schema:\n"
        "{\n"
        '  "title": "عنوان الدرس",\n'
        '  "segments": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "title": "عنوان الشريحة",\n'
        '      "bullet_points": ["نقطة ١", "نقطة ٢", "نقطة ٣"],\n'
        '      "narration_text": "الكلام اللي الراوي هيقوله بالمصري",\n'
        '      "voice_id": 1,\n'
        '      "image_prompt": "A minimalist continuous line drawing of a glowing brain on a white background, whiteboard sketch style"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "IMAGE PROMPT RULES:\n"
        "- image_prompt MUST be written in ENGLISH (it is for an AI image generator, not for the user).\n"
        "- Write a highly descriptive, concise prompt for a whiteboard-style illustration representing this slide's content.\n"
        "- Style keywords to include: minimalist, clean, whiteboard sketch, line drawing, educational diagram, white background.\n\n"
        "Constraints:\n"
        "- Each slide has 2-4 bullet points\n"
        "- narration_text MUST be written as natural spoken narration in Egyptian Arabic — "
        "conversational, clear, and ready to be read aloud by a text-to-speech engine. "
        "Do NOT use headers, bullet formatting, or symbols in narration_text.\n"
        "- Each narration_text should be 2-3 sentences\n"
        "- Cover ALL major topics progressively\n\n"
        f"SOURCE TEXT:\n{text[:15000]}"
    )

    # ── Step 1: Generate script JSON from Gemini (with timeout and retries) ──
    max_retries = 3
    parsed = {"title": "Generated Video", "segments": []}
    
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=settings.AI_TIMEOUT_SECONDS,
            )
            raw = response.text.strip()
            parsed = clean_and_parse_json(raw)
            
            # Validation Check: Ensure we have at least 15 segments
            if "segments" in parsed and len(parsed["segments"]) >= 15:
                break
            else:
                logger.warning(f"[VIDEO] Validation Warning: Attempt {attempt} generated only {len(parsed.get('segments', []))} segments. Target >= 15. Retrying...")
                
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini script generation timed out after {settings.AI_TIMEOUT_SECONDS}s")
            logger.warning(f"[VIDEO] Attempt {attempt} timed out. Retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to generate video structure: {e}")
            logger.warning(f"[VIDEO] Attempt {attempt} failed: {e}. Retrying...")

    # ── Step 2: Generate images + TTS audio for each segment ──
    total_duration = 0.0
    for segment in parsed.get("segments", []):
        # 2a. Generate whiteboard image from image_prompt
        try:
            image_prompt = segment.get("image_prompt", "")
            if image_prompt:
                segment["image_url"] = await generate_whiteboard_image(image_prompt)
            else:
                segment["image_url"] = ""
        except Exception as e:
            logger.warning(f"[VIDEO] Image generation failed for segment {segment.get('id')}: {e}")
            segment["image_url"] = ""

        # 2b. Generate TTS audio
        try:
            voice_key = "host" if segment.get("voice_id", 1) == 1 else "expert"
            audio_url, duration = await generate_tts_audio(
                segment["narration_text"], voice=voice_key
            )
            segment["audio_url"] = audio_url
            segment["duration_seconds"] = duration
            total_duration += duration
        except Exception as e:
            logger.warning(f"[VIDEO] TTS failed for segment {segment.get('id')}: {e}")
            segment["audio_url"] = ""
            segment["duration_seconds"] = len(segment["narration_text"].split()) / 2.5

    parsed["total_duration_seconds"] = total_duration

    logger.info(
        f"[VIDEO] ✓ Generated {len(parsed.get('segments', []))} segments, "
        f"~{total_duration:.1f}s total"
    )
    return parsed
