"""
Ruya — Podcast Service (Atomic / VPS-Ready)
=============================================
Generates AI-powered conversational podcasts from educational text.
Uses three AI speakers (Host1, Host2, Guest) to create engaging dialogue.

Fix: Sanitises AI output BEFORE Pydantic validation to handle misnamed
     narration fields (text / content / narration_text).

Flow:
  1. Gemini → raw JSON script
  2. Sanitise + truncate turns array (Python, pre-validation)
  3. ElevenLabs TTS for each turn (parallel)
  4. FFmpeg stitch all MP3s → final_podcast.mp3
  5. Upload to Supabase → return final_audio_url
"""

import re
import logging
import asyncio

import google.generativeai as genai

from app.core.config import settings
from app.services.tts_service import generate_tts_audio
from app.services.ai_engine import clean_and_parse_json
from app.services.ffmpeg_service import stitch_audio

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
    '      "narration_text": "الكلام اللي المتحدث هيقوله"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Conversation must flow naturally like a real Egyptian talk show\n"
    "- Rotate between all three speakers organically (not strictly alternating)\n"
    "- Cover ALL major topics from the source text\n"
    "- Each turn's narration_text MUST be written as natural spoken dialogue — "
    "conversational, clear, ready to be read aloud by a text-to-speech engine. "
    "Do NOT use formatting, symbols, or bullet points in narration_text.\n"
    "- Each turn should be 1-3 sentences\n"
    "- Start with Host1 introducing the topic and welcoming Host2 and Guest\n"
    "- End with Host1 summarizing key takeaways\n"
)


def _sanitise_turns(raw_turns: list, max_turns: int) -> list:
    """
    Sanitise the AI-generated turns array BEFORE Pydantic sees it.

    Problems fixed:
      1. AI sometimes names the text field 'text' or 'content' instead of 'narration_text'
      2. AI sometimes returns more turns than requested (causes length/timeout issues)

    Returns a clean list with guaranteed 'narration_text' on every item.
    """
    sanitised = []
    for i, turn in enumerate(raw_turns[:max_turns]):
        # Safely extract the spoken text regardless of field name
        narration = (
            turn.get("narration_text")
            or turn.get("text")
            or turn.get("content")
            or turn.get("dialogue")
            or "..."
        )
        # Strip any remaining markdown / symbol artifacts from TTS text
        narration = re.sub(r"[*_#\[\](){}]", "", str(narration)).strip() or "..."

        sanitised.append({
            "id":             turn.get("id", i + 1),
            "speaker":        turn.get("speaker", "Host1"),
            "narration_text": narration,
            "audio_url":      turn.get("audio_url", ""),
            "duration_seconds": turn.get("duration_seconds", 0.0),
        })

    return sanitised


async def generate_podcast(text: str, num_turns: int = 8, style: str = "educational") -> dict:
    """
    Generate a full podcast conversation from educational text.

    Returns a dict with keys:
        title, total_duration_seconds, final_audio_url
    """
    # Enforce hard limit
    num_turns = min(num_turns, settings.PODCAST_MAX_SEGMENTS)
    logger.info(f"[PODCAST] Generating {num_turns}-turn {style} podcast...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.5,
        },
    )

    user_prompt = (
        f"Create a {style} podcast conversation with exactly {num_turns} turns.\n\n"
        f"SOURCE TEXT:\n{text[:6000]}"
    )

    full_prompt = f"{PODCAST_SYSTEM_PROMPT}\n\nUser Task:\n{user_prompt}"

    # ── Step 1: Generate script JSON ──────────────────────────────────────────
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

    # ── Step 2: Sanitise turns BEFORE any Pydantic validation ────────────────
    raw_turns = parsed.get("turns", [])
    turns = _sanitise_turns(raw_turns, max_turns=num_turns)

    if not turns:
        raise RuntimeError("AI returned no valid turns for podcast")

    logger.info(f"[PODCAST] Sanitised {len(turns)} turns (was {len(raw_turns)} raw)")

    # ── Step 3: Generate TTS audio for each turn (Parallel) ──────────────────
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
                turn["narration_text"],
                voice=voice_key
            )
            turn["audio_url"] = audio_url
            turn["duration_seconds"] = duration
            return duration
        except Exception as e:
            logger.warning(f"[PODCAST] TTS failed for turn {turn.get('id')}: {e}")
            turn["audio_url"] = ""
            duration = len(turn["narration_text"].split()) / 2.5
            turn["duration_seconds"] = duration
            return duration

    durations = await asyncio.gather(*[_process_turn(t) for t in turns])
    total_duration = sum(durations)

    # ── Step 4: FFmpeg stitch all MP3s into one final podcast file ────────────
    logger.info("[PODCAST] Stitching all audio turns with FFmpeg...")
    try:
        final_audio_url = await stitch_audio(turns)
    except Exception as e:
        logger.error(f"[PODCAST] FFmpeg stitch failed: {e}. Returning empty URL.")
        final_audio_url = ""

    logger.info(
        f"[PODCAST] ✓ Generated {len(turns)} turns, "
        f"~{total_duration:.1f}s total | URL: {final_audio_url[:60]}"
    )

    return {
        "title":                  parsed.get("title", "بودكاست تعليمي"),
        "total_duration_seconds": round(total_duration, 2),
        "final_audio_url":        final_audio_url,
    }

