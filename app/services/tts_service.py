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

# ── Image Generation ──────────────────────────────────────────────────────────

async def generate_whiteboard_image(prompt: str) -> str:
    if not settings.HF_API_TOKEN: return ""
    try:
        async with httpx.AsyncClient(timeout=40) as client:
            response = await client.post(
                f"https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
                headers={"Authorization": f"Bearer {settings.HF_API_TOKEN}"},
                json={"inputs": prompt},
            )
        if response.status_code == 200:
            if not supabase: return "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")
            file_name = f"img_{uuid.uuid4().hex}.png"
            supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(path=file_name, file=response.content)
            return supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_name)
    except: return ""
    return ""

# ── Video Logic (The Forced 5-Minute Refactor) ────────────────────────────────

async def generate_video_segments(text: str, num_segments: int = 20) -> dict:
    """
    إجبار النظام على توليد 20 مقطع بنصوص طويلة جداً لضمان الوصول لـ 5 دقائق.
    """
    # Force 20 segments to ensure length
    num_segments = 20
    logger.info(f"[VIDEO] TARGETING 5 MINUTES: Generating exactly {num_segments} long segments...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.4, # زيادة شوية عشان ميكررش الكلام
        },
    )

    prompt = (
        "أنت مُعلم مصري بارع ومُبدع. مهمتك تحويل النص المرفق إلى فيديو تعليمي طويل جداً ومفصل (Whiteboard Animation).\n\n"
        f"قواعد طول الفيديو (صارمة - إجباري):\n"
        f"1. يجب أن تقوم بتوليد {num_segments} شريحة بالضبط (20 شريحة).\n"
        "2. كل شريحة في حقل 'narration_text' يجب أن تحتوي على نص طويل جداً (شرح مفصل) يتراوح بين 60 إلى 80 كلمة.\n"
        "3. إجمالي عدد الكلمات في الفيديو بالكامل يجب ألا يقل عن 1200 كلمة لضمان مدة 5 دقائق.\n"
        "4. ممنوع الاختصار نهائياً. اشرح كل نقطة بعمق وتفصيل ممل.\n\n"
        "قواعد اللغة:\n"
        "- اللغة هي العامية المصرية المثقفة (زي بودكاست علمي).\n"
        "- استخدم تعبيرات مصرية دارجة لجذب الانتباه.\n\n"
        "قواعد الـ TTS:\n"
        "- ممنوع استخدام أي رموز أو ايموجي أو علامات ترقيم غريبة.\n"
        "- اكتب الكلمات الانجليزية بالعربي (مثلاً: 'إيه آي' بدلاً من 'AI').\n\n"
        "Output MUST be valid JSON:\n"
        "{\n"
        '  "title": "عنوان طويل وشيق",\n'
        '  "segments": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "title": "عنوان الشريحة",\n'
        '      "bullet_points": ["نقطة 1", "نقطة 2", "نقطة 3"],\n'
        '      "narration_text": "هنا تكتب نص طويل جداً (60-80 كلمة) يشرح الموضوع بتفصيل شديد جداً بالمصري العامي وبدون أي اختصارات.",\n'
        '      "voice_id": 1,\n'
        '      "image_prompt": "Whiteboard line drawing of [subject] on white background"\n'
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
                timeout=90, # وقت أطول للردود الطويلة
            )
            raw = response.text.strip()
            parsed = clean_and_parse_json(raw)
            
            # Guard: لو جيمناي بعت عدد قليل أو نص قصير، نرفض
            total_words = sum(len(s.get("narration_text", "").split()) for s in parsed.get("segments", []))
            if len(parsed.get("segments", [])) >= 15 and total_words >= 800:
                logger.info(f"[VIDEO] Success! Total words: {total_words}. Segments: {len(parsed['segments'])}")
                break
            else:
                logger.warning(f"[VIDEO] Attempt {attempt+1} too short ({total_words} words). Retrying...")
        except Exception as e:
            logger.error(f"[VIDEO] Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1: raise e

    # Processing images and audio
    total_duration = 0.0
    for segment in parsed.get("segments", []):
        img_p = segment.get("image_prompt", "")
        segment["image_url"] = await generate_whiteboard_image(img_p) if img_p else ""
        
        voice_key = "host" if segment.get("voice_id", 1) == 1 else "expert"
        try:
            audio_url, duration = await generate_tts_audio(segment["narration_text"], voice=voice_key)
            segment["audio_url"] = audio_url
            segment["duration_seconds"] = duration
            total_duration += duration
        except:
            segment["audio_url"] = ""
            segment["duration_seconds"] = len(segment["narration_text"].split()) / 2.0

    parsed["total_duration_seconds"] = total_duration
    logger.info(f"[VIDEO] Final Duration: {total_duration:.1f}s")
    return parsed