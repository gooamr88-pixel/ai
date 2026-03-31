"""
Ruya — TTS & Video Generation Service (Refactored)
=====================================================
Generates 20-segment videos targeting 7-10 minutes.
Each segment has 2 images for visual dynamism.
Image generation: HF primary → Gemini fallback → local placeholder.
"""

import base64
import logging
import asyncio
import uuid
import re
import random
from typing import Tuple, List, Dict, Any
from io import BytesIO

import httpx
from elevenlabs.client import ElevenLabs
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

from app.core.config import settings
from app.services.ai_engine import clean_and_parse_json
from app.core.database import supabase

logger = logging.getLogger(__name__)

# ── Voice Mapping ───────────────────────────────────────────────────────────
VOICES = {
    "default": "onwK4e9ZLuTAKqWW03F9",
    "host":    "onwK4e9ZLuTAKqWW03F9",
    "expert":  "EXAVITQu4vr4xnSDxMaL",
    "guest":   "nPczCjzI2devNBz1zQrb",
}

ELEVENLABS_MODEL = "eleven_multilingual_v2"
TTS_TIMEOUT = 60

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

        word_count = len(text.split())
        duration = word_count / 2.0 
        return audio_url, duration

    except Exception as e:
        logger.error(f"[TTS] Critical Error: {e}")
        raise RuntimeError(f"TTS failed: {e}")


# ── Image Generation (HF → Gemini Fallback → Local Placeholder) ──────────────

_HF_ENDPOINT = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"


def _generate_local_placeholder(prompt: str) -> bytes:
    """Generate a colored placeholder image using Pillow when all APIs fail."""
    # Pick a color based on prompt hash for consistency
    colors = [
        "#1a237e", "#4a148c", "#00695c", "#e65100", 
        "#1b5e20", "#880e4f", "#0d47a1", "#4e342e",
    ]
    bg_color = colors[hash(prompt) % len(colors)]
    
    img = Image.new("RGB", (1280, 720), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Add a subtle gradient overlay effect
    for y in range(720):
        alpha = int(60 * (y / 720))
        draw.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha) if alpha > 0 else bg_color)
    
    # Simple centered text
    text = "Ruya Educational Content"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((1280 - text_w) // 2, (720 - text_h) // 2),
        text, fill="#ffffff", font=font
    )
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def generate_whiteboard_image(prompt: str) -> str:
    """Generate image: HF primary → Gemini fallback → local placeholder."""
    if not prompt:
        prompt = "whiteboard background"

    img_bytes: bytes = b""
    content_type = "image/png"

    # ── Primary: Hugging Face with improved retry ─────────────────────────────
    hf_headers = {"Authorization": f"Bearer {settings.HF_API_TOKEN}"}
    hf_payload = {"inputs": f"Clean educational whiteboard illustration, flat vector style, white background, no text, showing: {prompt}"}
    
    max_attempts = 5
    base_wait = 2

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                hf_resp = await client.post(_HF_ENDPOINT, headers=hf_headers, json=hf_payload)
            
            if hf_resp.status_code == 200:
                if "image" in hf_resp.headers.get("content-type", ""):
                    img_bytes = hf_resp.content
                    logger.info(f"[IMAGE] HF image generated ({len(img_bytes)} bytes) for: {prompt[:60]}")
                    break
                else:
                    logger.warning(f"[IMAGE] Attempt {attempt+1}: HF returned 200 but not image. Body: {hf_resp.text[:200]}")
            
            elif hf_resp.status_code == 503:
                try:
                    error_data = hf_resp.json()
                    wait_time = float(error_data.get("estimated_time", base_wait))
                    wait_time = min(wait_time, 30.0)
                    logger.info(f"[IMAGE] Model loading (503). Waiting {wait_time:.1f}s ({attempt+1}/{max_attempts})...")
                    await asyncio.sleep(wait_time)
                    continue
                except Exception as e:
                    logger.warning(f"[IMAGE] Failed to parse 503 JSON: {e}")
            
            else:
                logger.warning(f"[IMAGE] Attempt {attempt+1}: HF returned {hf_resp.status_code}")
            
        except httpx.TimeoutException:
            logger.warning(f"[IMAGE] Attempt {attempt+1}: HF timed out")
        except Exception as e:
            logger.warning(f"[IMAGE] Attempt {attempt+1}: HF failed: {e}")
        
        # Exponential backoff with jitter
        if attempt < max_attempts - 1:
            wait_time = base_wait * (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)

    # ── Fallback 1: Gemini Imagen ─────────────────────────────────────────────
    if not img_bytes and settings.GOOGLE_API_KEY:
        try:
            logger.info("[IMAGE] HF failed. Trying Gemini Imagen fallback...")
            imagen_model = genai.ImageGenerationModel("imagen-3.0-fast-generate-001")
            
            imagen_prompt = f"Clean educational whiteboard illustration, flat vector style, white background, no text, showing: {prompt}"
            
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    imagen_model.generate_images,
                    prompt=imagen_prompt,
                    number_of_images=1,
                ),
                timeout=30,
            )
            
            if result.images:
                img_bytes = result.images[0]._image_bytes
                content_type = "image/png"
                logger.info(f"[IMAGE] ✓ Gemini Imagen generated ({len(img_bytes)} bytes)")
                
        except Exception as e:
            logger.warning(f"[IMAGE] Gemini Imagen fallback failed: {e}")

    # ── Fallback 2: Local Pillow placeholder ──────────────────────────────────
    if not img_bytes:
        try:
            img_bytes = await asyncio.to_thread(_generate_local_placeholder, prompt)
            logger.info(f"[IMAGE] Using local placeholder ({len(img_bytes)} bytes)")
        except Exception as e:
            logger.error(f"[IMAGE] Even local placeholder failed: {e}")
            return ""

    # ── Upload to Supabase ────────────────────────────────────────────────────
    if not supabase:
        return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")

    file_name = f"img_{uuid.uuid4().hex}.png"
    def _upload_img():
        supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
            path=file_name,
            file=img_bytes,
            file_options={"content-type": content_type},
        )
        return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)

    try:
        url = await asyncio.wait_for(asyncio.to_thread(_upload_img), timeout=20)
        
        # Verify URL is accessible
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                head_resp = await client.head(url)
                if head_resp.status_code >= 400:
                    logger.warning(f"[IMAGE] URL verification failed ({head_resp.status_code}): {url[:80]}")
        except Exception:
            pass  # Non-critical — URL might still work
        
        return url
    except Exception as e:
        logger.error(f"[IMAGE] Supabase upload failed: {e}")
        return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")


# ── Batch Image Generation ───────────────────────────────────────────────────

async def _generate_images_batch(prompts: List[str], batch_size: int = 5) -> List[str]:
    """Generate images in parallel batches to avoid overwhelming APIs."""
    all_urls = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        logger.info(f"[IMAGE] Processing batch {i//batch_size + 1} ({len(batch)} images)...")
        results = await asyncio.gather(
            *[generate_whiteboard_image(p) for p in batch],
            return_exceptions=True,
        )
        for r in results:
            all_urls.append(r if isinstance(r, str) else "")
    return all_urls


# ── Video Generation (20 segments, 2 images each, 7-10 min target) ───────────

async def generate_video_segments(text: str, num_segments: int = 20) -> dict:
    """
    Generates 20 segments targeting an 8-10 minute video.
    Each segment has 2 image prompts for visual dynamism.
    """
    num_segments = min(num_segments, settings.VIDEO_MAX_SEGMENTS)
    logger.info(f"[VIDEO] Generating {num_segments} segments (8-10 min target)...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.4,
            "max_output_tokens": 12000,
        },
    )

    prompt = (
        "أنت مُعلم مصري بارع ومُبدع. مهمتك تحويل النص المرفق إلى فيديو تعليمي (Whiteboard Animation) مدته 8-10 دقائق.\n\n"
        "قواعد صارمة وإجبارية — لا استثناءات:\n"
        f"1. يجب أن تُولِّد {num_segments} شريحة بالضبط (EXACTLY {num_segments} segments). لا أكثر ولا أقل.\n"
        "2. كل شريحة تحتوي على 80 إلى 100 كلمة في 'narration_text'.\n"
        "3. إجمالي كلمات جميع الشرائح مجتمعة يجب أن يكون بين 1600 و 2000 كلمة.\n"
        "4. كل شريحة تحتوي على 2 image prompts بالإنجليزي (مشهدين بصريين مختلفين للشريحة).\n\n"
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
        '      "narration_text": "نص الشريحة هنا — بين 80 و 100 كلمة بالمصري العامي.",\n'
        '      "image_prompts": ["English scene description 1", "English scene description 2"]\n'
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
                timeout=120,
            )
            raw = response.text.strip()
            parsed = clean_and_parse_json(raw)

            # Guard: accept if we got enough quality content
            total_words = sum(len(s.get("narration_text", "").split()) for s in parsed.get("segments", []))
            seg_count = len(parsed.get("segments", []))
            
            if seg_count >= 16 and total_words >= 1200:
                logger.info(f"[VIDEO] ✓ Attempt {attempt+1}: {seg_count} segments, {total_words} words")
                break
            else:
                logger.warning(f"[VIDEO] Attempt {attempt+1} insufficient ({total_words} words / {seg_count} segs). Retrying...")
        except Exception as e:
            logger.error(f"[VIDEO] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1: raise e

    # Hard-cap segments
    segments = parsed.get("segments", [])[:num_segments]
    parsed["segments"] = segments
    logger.info(f"[VIDEO] Processing {len(segments)} segments.")

    # ── Generate all images in parallel batches ──────────────────────────────
    all_image_prompts = []
    prompt_to_segment = []  # Track which segment each prompt belongs to
    
    for idx, seg in enumerate(segments):
        prompts = seg.get("image_prompts", [seg.get("image_prompt", "")])
        # Ensure exactly 2 prompts per segment
        if len(prompts) < 2:
            prompts = prompts + [prompts[0] if prompts else "educational whiteboard"]
        prompts = prompts[:2]
        seg["_image_prompt_count"] = len(prompts)
        all_image_prompts.extend(prompts)
        prompt_to_segment.extend([idx] * len(prompts))

    logger.info(f"[VIDEO] Generating {len(all_image_prompts)} images in batches...")
    all_image_urls = await _generate_images_batch(all_image_prompts, batch_size=5)

    # Assign image URLs back to segments
    url_idx = 0
    for seg in segments:
        count = seg.pop("_image_prompt_count", 2)
        seg["image_urls"] = all_image_urls[url_idx:url_idx + count]
        url_idx += count
        # Keep single image_url for backward compat
        seg["image_url"] = seg["image_urls"][0] if seg["image_urls"] else ""

    # ── Generate TTS audio for each segment ──────────────────────────────────
    total_duration = 0.0
    for segment in segments:
        narration = (
            segment.get("narration_text")
            or segment.get("text")
            or segment.get("content")
            or "..."
        )
        segment["narration_text"] = narration

        voice_key = "host"
        try:
            audio_url, duration = await generate_tts_audio(narration, voice=voice_key)
            segment["audio_url"] = audio_url
            segment["duration_seconds"] = duration
            total_duration += duration
        except Exception:
            fallback_duration = round(len(narration.split()) / 2.0, 2)
            segment["audio_url"] = ""
            segment["duration_seconds"] = fallback_duration
            total_duration += fallback_duration
            logger.warning(f"[VIDEO] TTS failed for segment, estimated {fallback_duration}s")

    parsed["total_duration_seconds"] = total_duration
    logger.info(f"[VIDEO] Per-segment done. Duration: {total_duration:.1f}s")

    # ── FFmpeg: stitch all clips into one final_video.mp4 ─────────────────────
    from app.services.ffmpeg_service import stitch_video

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
