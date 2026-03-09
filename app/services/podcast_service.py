"""
Ruya — Podcast Service (Atomic / Vercel-Safe)
================================================
Generates AI-powered conversational podcasts from educational text.
Uses three AI speakers (Host1, Host2, Guest) to create engaging dialogue.
Returns Base64-encoded audio segments for each turn via ElevenLabs TTS.

All external calls wrapped with asyncio.wait_for() timeouts.
"""

import re
import logging
import asyncio

import google.generativeai as genai

from app.core.config import settings
from app.services.tts_service import generate_tts_audio
from app.services.ai_engine import clean_and_parse_json

logger = logging.getLogger(__name__)


PODCAST_SYSTEM_PROMPT = (
    "أنت كاتب سيناريو لبودكاست مصري تعليمي ممتع على طريقة التوك شو.\n"
    "Create a lively, natural talk-show conversation between THREE speakers about the given text.\n\n"
    "LANGUAGE RULES (CRITICAL):\n"
    "- ALL dialogue MUST be in heavy Egyptian Colloquial Arabic (اللهجة المصرية العامية الدارجة).\n"
    "- ALL dialogue MUST use Egyptian phrasing, idioms, and expressions.\n"
    "- ALL three speakers, including the Guest, MUST speak in natural Egyptian Arabic.\n"
    "- If the source text is in English or another language, TRANSLATE and adapt to Egyptian Arabic.\n"
    "- Use casual, humorous Egyptian expressions (يعني، أيوه بالظبط، لا خلي بالك، طب سمعني الحتة دي).\n\n"
    "TTS SCRIPTING RULES (CRITICAL — text will be read aloud by a TTS engine):\n"
    "- NEVER use emojis, markdown, asterisks (*), underscores (_), hashtags (#), or ANY special symbols.\n"
    "- NEVER use bracket stage directions like [يضحك] or (يبتسم) — write ONLY spoken words.\n"
    "- USE frequent punctuation: commas (،) and periods (.) to force the TTS to take natural breathing pauses.\n"
    "- SPELL OUT English acronyms or terms using Arabic letters phonetically. "
    "For example: write 'إتش تي إم إل' instead of 'HTML', 'فيسبوك' instead of 'Facebook', "
    "'إيه آي' instead of 'AI', 'بايثون' instead of 'Python'.\n"
    "- Write in a conversational, calm pace. Short sentences. Natural flow.\n\n"
    "Speakers:\n"
    "- Host1: The main host. Leads the conversation, asks interesting questions, uses humor.\n"
    "- Host2: The co-host. Adds commentary, jokes, follow-up questions, and reactions.\n"
    "- Guest: A specialist on the topic. Explains concepts deeply but in Egyptian dialect. Feels like an Egyptian character.\n\n"
    "Output MUST be valid JSON matching this schema:\n"
    "{\n"
    '  "title": "عنوان الحلقة",\n'
    '  "description": "وصف مختصر للحلقة",\n'
    '  "speakers": ["Host1", "Host2", "Guest"],\n'
    '  "turns": [\n'
    "    {\n"
    '      "id": 1,\n'
    '      "speaker": "Host1",\n'
    '      "text": "الكلام اللي المتحدث هيقوله"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Conversation must flow naturally like a real Egyptian talk show\n"
    "- Rotate between all three speakers organically (not strictly alternating)\n"
    "- Cover ALL major topics from the source text\n"
    "- Each turn's text MUST be written as natural spoken dialogue — "
    "conversational, clear, ready to be read aloud by a text-to-speech engine. "
    "Do NOT use formatting, symbols, or bullet points in turn text.\n"
    "- Each turn should be 1-3 sentences\n"
    "- Start with Host1 introducing the topic and welcoming Host2 and Guest\n"
    "- End with Host1 summarizing key takeaways\n"
)


async def generate_podcast(text: str, num_turns: int = 8, style: str = "educational") -> dict:
    """
    Generate a full podcast conversation from educational text.

    1. Uses Gemini to generate conversation script (with timeout)
    2. Uses ElevenLabs to generate audio for each turn (with timeout)
    3. Returns structured response with Base64 audio per turn
    """
    # Enforce hard limit
    num_turns = min(num_turns, settings.PODCAST_MAX_SEGMENTS)
    logger.info(f"[PODCAST] Generating {num_turns}-turn {style} podcast...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.5,  # Slightly creative for natural conversation
        },
    )

    user_prompt = (
        f"Create a {style} podcast conversation with exactly {num_turns} turns.\n\n"
        f"SOURCE TEXT:\n{text[:6000]}"
    )

    full_prompt = f"{PODCAST_SYSTEM_PROMPT}\n\nUser Task:\n{user_prompt}"

    # ── Step 1: Generate script JSON (with timeout) ──
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, full_prompt),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Gemini podcast script generation timed out after {settings.AI_TIMEOUT_SECONDS}s"
        )

    raw = response.text.strip()
    parsed = clean_and_parse_json(raw)

    # ── Step 2: Generate TTS audio for each turn (Parallel) ──
    turns = parsed.get("turns", [])

    async def _process_turn(turn: dict):
        try:
            speaker = turn.get("speaker", "Host1")
            if speaker == "Host1":
                voice_key = "host"
            elif speaker == "Host2":
                voice_key = "expert"
            else:  # Guest
                voice_key = "guest"
            
            audio_url, duration = await generate_tts_audio(
                turn["text"],
                voice=voice_key
            )
            turn["audio_url"] = audio_url
            turn["duration_seconds"] = duration
            return duration
        except Exception as e:
            logger.warning(f"[PODCAST] TTS failed for turn {turn.get('id')}: {e}")
            turn["audio_url"] = ""
            duration = len(turn["text"].split()) / 2.5
            turn["duration_seconds"] = duration
            return duration

    # Run all turns concurrently
    durations = await asyncio.gather(*[_process_turn(t) for t in turns])
    total_duration = sum(durations)

    parsed["total_duration_seconds"] = total_duration

    logger.info(
        f"[PODCAST] ✓ Generated {len(turns)} turns, "
        f"~{total_duration:.1f}s total"
    )
    return parsed
