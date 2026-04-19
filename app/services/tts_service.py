"""
Ruya — TTS & Video Generation Service (Refactored)
=====================================================
Generates 20-segment videos targeting 7-10 minutes.
Each segment has 2 images for visual dynamism.
Image generation: HF primary → Gemini fallback → local placeholder.

Architecture:
  Chunked generation — splits input into 3 chunks, generates 7+7+6
  segments sequentially to avoid Groq TPM limits and Gemini output
  truncation. Segments are merged and re-numbered 1-20.
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

from app.core.config import settings
from app.services.ai_engine import clean_and_parse_json, smart_chunk_text, hybrid_call
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

# ── Shared Clients (Connection Pooling) ─────────────────────────────────────

_elevenlabs_client: ElevenLabs | None = None

def _get_elevenlabs_client() -> ElevenLabs:
    """Lazy singleton — avoids creating a new HTTP connection pool per TTS call."""
    global _elevenlabs_client
    if _elevenlabs_client is None:
        _elevenlabs_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
    return _elevenlabs_client

_http_client: httpx.AsyncClient | None = None

def _get_http_client() -> httpx.AsyncClient:
    """Lazy singleton — reuses TCP/TLS connections across image generation calls."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    return _http_client

# ── TTS Logic ───────────────────────────────────────────────────────────────

def _generate_tts_sync(text: str, voice_id: str) -> bytes:
    client = _get_elevenlabs_client()
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
    
    # Add a subtle gradient darkening toward the bottom (proper alpha compositing)
    gradient = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    grad_draw = ImageDraw.Draw(gradient)
    for y in range(720):
        alpha = int(60 * (y / 720))
        grad_draw.line([(0, y), (1280, y)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img.convert("RGBA"), gradient).convert("RGB")
    
    draw = ImageDraw.Draw(img)
    
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
            client = _get_http_client()
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
            
            elif hf_resp.status_code == 402:
                # Payment required — quota exhausted, no point retrying
                logger.warning(f"[IMAGE] HF quota exhausted (402). Skipping remaining attempts.")
                break
            
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

    # ── Fallback 1: Gemini Imagen via REST API ────────────────────────────────
    if not img_bytes and settings.GOOGLE_API_KEY:
        try:
            logger.info("[IMAGE] HF failed. Trying Gemini Imagen REST API fallback...")
            imagen_prompt = f"Clean educational whiteboard illustration, flat vector style, white background, no text, showing: {prompt}"
            
            client = _get_http_client()
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-fast-generate-001:predict?key={settings.GOOGLE_API_KEY}",
                json={
                    "instances": [{"prompt": imagen_prompt}],
                    "parameters": {"sampleCount": 1}
                },
                timeout=30,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                predictions = data.get("predictions", [])
                if predictions:
                    import base64 as b64mod
                    img_b64 = predictions[0].get("bytesBase64Encoded", "")
                    if img_b64:
                        img_bytes = b64mod.b64decode(img_b64)
                        content_type = "image/png"
                        logger.info(f"[IMAGE] ✓ Gemini Imagen generated ({len(img_bytes)} bytes)")
                else:
                    logger.warning(f"[IMAGE] Gemini Imagen returned no predictions")
            else:
                logger.warning(f"[IMAGE] Gemini Imagen returned {resp.status_code}: {resp.text[:200]}")
                
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
            client = _get_http_client()
            head_resp = await client.head(url, timeout=10)
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
    total_failed = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"[IMAGE-BATCH] Processing batch {batch_num} ({len(batch)} images)...")
        results = await asyncio.gather(
            *[generate_whiteboard_image(p) for p in batch],
            return_exceptions=True,
        )
        for j, r in enumerate(results):
            prompt_idx = i + j
            if isinstance(r, Exception):
                total_failed += 1
                logger.error(
                    f"[IMAGE-BATCH] ✗ Image {prompt_idx} FAILED with exception: "
                    f"{type(r).__name__}: {r}  |  Prompt: {prompts[prompt_idx][:80]}"
                )
                all_urls.append("")  # Still append empty to keep index alignment
            elif isinstance(r, str) and r:
                logger.info(f"[IMAGE-BATCH] ✓ Image {prompt_idx} OK ({len(r)} chars URL)")
                all_urls.append(r)
            else:
                total_failed += 1
                logger.error(
                    f"[IMAGE-BATCH] ✗ Image {prompt_idx} returned EMPTY string. "
                    f"Prompt: {prompts[prompt_idx][:80]}"
                )
                all_urls.append("")

    logger.info(
        f"[IMAGE-BATCH] ═══ Complete: {len(all_urls)} total, "
        f"{len(all_urls) - total_failed} succeeded, {total_failed} failed ═══"
    )
    if total_failed == len(all_urls):
        logger.critical(
            f"[IMAGE-BATCH] ALL {len(all_urls)} images failed! "
            f"Check HF_API_TOKEN and GOOGLE_API_KEY validity."
        )
    return all_urls


# ── Video System Prompt (chunk-aware) ─────────────────────────────────────────

VIDEO_SYSTEM_PROMPT = (
    "أنت مُعلم مصري بارع ومُبدع. مهمتك تحويل النص المرفق إلى جزء من فيديو تعليمي (Whiteboard Animation).\n\n"
    "قواعد صارمة وإجبارية — لا استثناءات:\n"
    "1. كل شريحة تحتوي على 80 إلى 100 كلمة في 'narration_text'.\n"
    "2. كل شريحة تحتوي على 2 image prompts بالإنجليزي (مشهدين بصريين مختلفين للشريحة).\n\n"
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
    "}\n"
)


# ── Chunked Video Generation (7+7+6 = 20 segments, 8-10 min) ─────────────────

async def _generate_chunk_segments(
    chunk_text_content: str,
    num_segments: int,
    chunk_index: int,
    total_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Generate a batch of video segments from a single text chunk.
    Uses hybrid_call (Groq primary → Gemini fallback) with safe token limits.
    Always returns the best result — never returns empty if AI responded.
    """
    user_prompt = (
        f"أنت بتولّد الجزء {chunk_index + 1} من {total_chunks} لفيديو تعليمي طويل.\n"
        f"يجب أن تُولِّد بالضبط {num_segments} شريحة (EXACTLY {num_segments} segments).\n"
        f"كل شريحة لازم تحتوي على 80-100 كلمة في 'narration_text' و 2 image prompts بالإنجليزي.\n"
        f"غطي كل المحتوى اللي في النص التالي بالتفصيل.\n\n"
        f"SOURCE TEXT:\n{chunk_text_content}"
    )

    max_retries = 3
    min_words_per_segment = 30  # Lowered from 60 — short input produces shorter narrations
    required_words = num_segments * min_words_per_segment

    # Track the best result across all attempts so we never return empty
    best_segments: List[Dict[str, Any]] = []
    best_word_count = 0

    for attempt in range(max_retries):
        try:
            raw = await hybrid_call(
                system_prompt=VIDEO_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                primary="groq",
                json_mode=True,
                max_tokens=6000,
            )
            parsed = clean_and_parse_json(raw)
            segments = parsed.get("segments", [])

            total_words = sum(len(s.get("narration_text", "").split()) for s in segments)
            avg_words = total_words / len(segments) if segments else 0

            # Track best result
            if total_words > best_word_count and len(segments) > 0:
                best_segments = segments
                best_word_count = total_words

            # Accept if we got enough segments AND enough words
            if len(segments) >= num_segments - 1 and total_words >= required_words:
                logger.info(
                    f"[VIDEO] Chunk {chunk_index + 1}/{total_chunks}: "
                    f"✓ {len(segments)} segments, {total_words} words, "
                    f"avg {avg_words:.0f} wps (attempt {attempt + 1})"
                )
                return segments[:num_segments]
            else:
                logger.warning(
                    f"[VIDEO] Chunk {chunk_index + 1} attempt {attempt + 1}: "
                    f"below threshold — got {len(segments)} segs, {total_words} words "
                    f"(avg {avg_words:.0f} wps), need ≥{num_segments - 1} segs "
                    f"and ≥{required_words} words. "
                    f"{'Retrying...' if attempt < max_retries - 1 else 'Using best result.'}"
                )
        except Exception as e:
            logger.error(
                f"[VIDEO] Chunk {chunk_index + 1} attempt {attempt + 1} EXCEPTION: "
                f"{type(e).__name__}: {e}"
            )
            if attempt == max_retries - 1 and not best_segments:
                raise

    # CRITICAL FIX: Use the best result instead of returning empty.
    # Even "insufficient" content is better than 0 segments which kills the entire pipeline.
    if best_segments:
        best_avg = best_word_count / len(best_segments) if best_segments else 0
        logger.warning(
            f"[VIDEO] Chunk {chunk_index + 1}/{total_chunks}: "
            f"⚠ No attempt met threshold. Using BEST result: "
            f"{len(best_segments)} segments, {best_word_count} words "
            f"(avg {best_avg:.0f} wps)"
        )
        return best_segments[:num_segments]

    logger.error(
        f"[VIDEO] Chunk {chunk_index + 1}/{total_chunks}: "
        f"✗ ALL {max_retries} attempts produced 0 usable segments!"
    )
    return []


async def generate_video_segments(text: str, num_segments: int = 20) -> dict:
    """
    Generates 20 segments targeting an 8-10 minute video using CHUNKED generation.

    Architecture:
      1. Split input text into 3 balanced chunks
      2. Generate 7 + 7 + 6 segments SEQUENTIALLY (respects Groq 12K TPM)
      3. Merge & re-number all segments as 1..20
      4. Generate images in parallel batches
      5. Generate TTS audio per segment
      6. FFmpeg stitch into final video
    """
    num_segments = min(num_segments, settings.VIDEO_MAX_SEGMENTS)
    logger.info(f"[VIDEO] ═══ Starting CHUNKED generation: {num_segments} segments (8-10 min target) ═══")

    # ── Step 1: Split input into 3 chunks ─────────────────────────────────────
    NUM_CHUNKS = 3
    text_chunks = smart_chunk_text(text, num_chunks=NUM_CHUNKS)
    logger.info(f"[VIDEO] Split input into {len(text_chunks)} chunks: {[len(c) for c in text_chunks]}")

    # Distribute segments across chunks: 7 + 7 + 6 = 20
    segments_per_chunk = []
    remaining = num_segments
    for i in range(len(text_chunks)):
        if i < len(text_chunks) - 1:
            count = remaining // (len(text_chunks) - i)
        else:
            count = remaining
        segments_per_chunk.append(count)
        remaining -= count

    logger.info(f"[VIDEO] Segment distribution: {segments_per_chunk} (total={sum(segments_per_chunk)})")

    # ── Step 2: Generate segments SEQUENTIALLY per chunk ──────────────────────
    all_segments: List[Dict[str, Any]] = []
    video_title = "فيديو تعليمي"

    for i, (chunk, seg_count) in enumerate(zip(text_chunks, segments_per_chunk)):
        logger.info(f"[VIDEO] ─── Generating chunk {i + 1}/{len(text_chunks)} ({seg_count} segments) ───")
        chunk_segments = await _generate_chunk_segments(
            chunk_text_content=chunk,
            num_segments=seg_count,
            chunk_index=i,
            total_chunks=len(text_chunks),
        )
        all_segments.extend(chunk_segments)

    # Extract title from the first chunk's response
    if all_segments:
        video_title = all_segments[0].get("title", video_title)

    # ── Step 3: Re-number merged segments (1..N) ─────────────────────────────
    for idx, seg in enumerate(all_segments):
        seg["id"] = idx + 1

    logger.info(
        f"[VIDEO] ═══ Merged {len(all_segments)} segments. "
        f"Total words: {sum(len(s.get('narration_text', '').split()) for s in all_segments)} ═══"
    )

    # Hard-cap segments
    segments = all_segments[:num_segments]
    logger.info(f"[VIDEO] Processing {len(segments)} segments.")

    # ── PRE-FLIGHT: Validate API keys before burning time in loops ────────
    preflight_warnings = []
    if not settings.ELEVENLABS_API_KEY:
        preflight_warnings.append("ELEVENLABS_API_KEY is MISSING — TTS will fail for ALL segments!")
    if not settings.HF_API_TOKEN:
        preflight_warnings.append("HF_API_TOKEN is MISSING — primary image generation will fail!")
    if not settings.GOOGLE_API_KEY:
        preflight_warnings.append("GOOGLE_API_KEY is MISSING — Gemini Imagen fallback will fail!")
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        preflight_warnings.append("SUPABASE credentials missing — uploads will use base64 data URIs.")

    for w in preflight_warnings:
        logger.warning(f"[VIDEO-PREFLIGHT] ⚠ {w}")

    if not settings.ELEVENLABS_API_KEY:
        raise RuntimeError(
            "[VIDEO-PREFLIGHT] FATAL: ELEVENLABS_API_KEY is not set. "
            "TTS will fail for every segment, producing 0 audio files, "
            "which causes FFmpeg to skip all clips. Aborting early."
        )

    # ── Step 4: Generate all images in parallel batches ───────────────────
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

    # ── Step 5: Generate TTS audio in BATCHES of 6 (was sequential → ~50s faster) ─
    TTS_BATCH_SIZE = 6
    total_duration = 0.0
    tts_success_count = 0
    tts_fail_count = 0

    # Pre-process: ensure narration_text exists on all segments
    for segment in segments:
        narration = (
            segment.get("narration_text")
            or segment.get("text")
            or segment.get("content")
            or "..."
        )
        segment["narration_text"] = narration

    async def _process_video_tts(seg_idx: int, seg: dict) -> float:
        """Process TTS for a single video segment. Returns duration."""
        narration = seg["narration_text"]
        try:
            logger.info(
                f"[TTS-LOOP] Segment {seg_idx + 1}/{len(segments)}: "
                f"generating audio for {len(narration)} chars / {len(narration.split())} words..."
            )
            audio_url, duration = await generate_tts_audio(narration, voice="host")
            if not audio_url:
                raise RuntimeError(f"Empty audio_url for segment {seg_idx + 1}")
            seg["audio_url"] = audio_url
            seg["duration_seconds"] = duration
            logger.info(
                f"[TTS-LOOP] ✓ Segment {seg_idx + 1}: audio OK "
                f"({duration:.1f}s, URL length={len(audio_url)})"
            )
            return duration
        except Exception as e:
            logger.error(
                f"[TTS-LOOP] ✗ Segment {seg_idx + 1}/{len(segments)} FAILED! "
                f"{type(e).__name__}: {e} | Preview: {narration[:100]}..."
            )
            fallback_duration = round(len(narration.split()) / 2.0, 2)
            seg["audio_url"] = ""
            seg["duration_seconds"] = fallback_duration
            return fallback_duration

    # FAIL-FAST: test first 2 segments sequentially before batching the rest
    for preflight_idx in range(min(2, len(segments))):
        dur = await _process_video_tts(preflight_idx, segments[preflight_idx])
        total_duration += dur
        if segments[preflight_idx].get("audio_url"):
            tts_success_count += 1
        else:
            tts_fail_count += 1

    if tts_fail_count >= 2 and tts_success_count == 0:
        raise RuntimeError(
            "[TTS-LOOP] FATAL: First 2 TTS calls failed. "
            "ElevenLabs API is likely misconfigured. Aborting."
        )

    # Process remaining segments in parallel batches of 6
    remaining_start = min(2, len(segments))
    for i in range(remaining_start, len(segments), TTS_BATCH_SIZE):
        batch_end = min(i + TTS_BATCH_SIZE, len(segments))
        batch_indices = list(range(i, batch_end))
        logger.info(f"[TTS-LOOP] Batch {(i - remaining_start) // TTS_BATCH_SIZE + 1} ({len(batch_indices)} segments)...")
        durations = await asyncio.gather(
            *[_process_video_tts(idx, segments[idx]) for idx in batch_indices]
        )
        for idx, dur in zip(batch_indices, durations):
            total_duration += dur
            if segments[idx].get("audio_url"):
                tts_success_count += 1
            else:
                tts_fail_count += 1

    logger.info(
        f"[TTS-LOOP] ═══ TTS Complete: {tts_success_count}/{len(segments)} succeeded, "
        f"{tts_fail_count} failed. Total duration: {total_duration:.1f}s ═══"
    )

    if tts_success_count == 0:
        raise RuntimeError(
            f"[TTS-LOOP] FATAL: ALL {len(segments)} TTS generations failed. "
            f"Cannot produce any video clips. Check ElevenLabs API key and quota."
        )

    # ── Step 6: FFmpeg: stitch all clips into one final_video.mp4 ─────────────
    from app.services.ffmpeg_service import stitch_video

    logger.info("[VIDEO] Stitching all segments with FFmpeg...")
    try:
        final_video_url = await stitch_video(segments)
    except Exception as e:
        logger.error(f"[VIDEO] FFmpeg stitch failed: {e}. Returning empty URL.")
        final_video_url = ""

    logger.info(f"[VIDEO] ✓ Final video URL: {final_video_url[:60]}")
    return {
        "title":                  video_title,
        "total_duration_seconds": round(total_duration, 2),
        "final_video_url":        final_video_url,
    }
