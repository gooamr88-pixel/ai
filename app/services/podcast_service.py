"""
Ruya — Podcast Service (Refactored for 7-10 min duration)
===========================================================
Generates AI-powered conversational podcasts from educational text.
Uses three AI speakers (Host1, Host2, Guest) with longer, richer turns.

Flow:
  1. Gemini → raw JSON script (35 turns, 60-120 words each)
  2. Sanitise + truncate turns array
  3. ElevenLabs TTS in batches of 6 (avoid rate limiting)
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
    "Create a lively, natural, LONG talk-show conversation between THREE speakers about the given text.\n\n"
    "DURATION TARGET: The podcast MUST be at least 7-10 minutes long when read aloud.\n"
    "To achieve this, each turn must be a FULL PARAGRAPH of spoken dialogue (4-8 sentences, 60-120 words per turn).\n\n"
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
    "- Write in a conversational, calm pace. Natural flow.\n\n"
    "CRITICAL TURN LENGTH RULES:\n"
    "- Each turn MUST be a FULL PARAGRAPH — 4 to 8 sentences.\n"
    "- Each turn MUST contain 60 to 120 words.\n"
    "- Short 1-2 sentence turns are NOT acceptable. Make each turn substantial and rich.\n"
    "- Think of each turn as a full speaking block, not a quick back-and-forth.\n\n"
    "Speakers:\n"
    "- Host1: The main host. Leads the conversation, asks interesting questions, uses humor.\n"
    "- Host2: The co-host. Adds commentary, jokes, follow-up questions, and reactions.\n"
    "- Guest: A specialist on the topic. Explains concepts deeply but in Egyptian dialect.\n\n"
    "Output MUST be valid JSON matching this schema:\n"
    "{\n"
    '  "title": "عنوان الحلقة",\n'
    '  "description": "وصف مختصر للحلقة",\n'
    '  "speakers": ["Host1", "Host2", "Guest"],\n'
    '  "turns": [\n'
    "    {\n"
    '      "id": 1,\n'
    '      "speaker": "Host1",\n'
    '      "narration_text": "فقرة كاملة من الكلام — 4-8 جمل، 60-120 كلمة"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Conversation must flow naturally like a real Egyptian talk show\n"
    "- Rotate between all three speakers organically\n"
    "- Cover ALL major topics from the source text\n"
    "- Each turn MUST be 60-120 words (full paragraph), NOT short sentences\n"
    "- Start with Host1 introducing the topic and welcoming Host2 and Guest\n"
    "- End with Host1 summarizing key takeaways\n"
)


def _sanitise_turns(raw_turns: list, max_turns: int) -> list:
    """Sanitise AI-generated turns BEFORE Pydantic validation."""
    sanitised = []
    for i, turn in enumerate(raw_turns[:max_turns]):
        narration = (
            turn.get("narration_text")
            or turn.get("text")
            or turn.get("content")
            or turn.get("dialogue")
            or "..."
        )
        narration = re.sub(r"[*_#\[\](){}]", "", str(narration)).strip() or "..."

        sanitised.append({
            "id":             turn.get("id", i + 1),
            "speaker":        turn.get("speaker", "Host1"),
            "narration_text": narration,
            "audio_url":      turn.get("audio_url", ""),
            "duration_seconds": turn.get("duration_seconds", 0.0),
        })

    return sanitised


async def generate_podcast(text: str, num_turns: int = 35, style: str = "educational") -> dict:
    """
    Generate a full 7-10 minute podcast from educational text.
    TTS calls are batched in groups of 6 to avoid ElevenLabs rate limiting.
    """
    num_turns = min(num_turns, settings.PODCAST_MAX_SEGMENTS)
    logger.info(f"[PODCAST] Generating {num_turns}-turn {style} podcast (7-10 min target)...")

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.5,
            "max_output_tokens": 12000,
        },
    )

    user_prompt = (
        f"Create a {style} podcast conversation with exactly {num_turns} turns.\n"
        f"IMPORTANT: Each turn MUST be 60-120 words (a full paragraph).\n"
        f"The total word count should be at least 2000 words.\n\n"
        f"SOURCE TEXT:\n{text[:12000]}"
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

    # ── Step 2: Sanitise turns ────────────────────────────────────────────────
    raw_turns = parsed.get("turns", [])
    turns = _sanitise_turns(raw_turns, max_turns=num_turns)

    if not turns:
        raise RuntimeError("AI returned no valid turns for podcast")

    logger.info(f"[PODCAST] Sanitised {len(turns)} turns (was {len(raw_turns)} raw)")

    # ── Step 3: Generate TTS audio in BATCHES of 6 ────────────────────────────
    BATCH_SIZE = 6
    total_duration = 0.0

    async def _process_turn(turn: dict) -> float:
        try:
            speaker = turn.get("speaker", "Host1")
            if speaker == "Host1":
                voice_key = "host"
            elif speaker == "Host2":
                voice_key = "expert"
            else:
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

    # Process in batches
    for i in range(0, len(turns), BATCH_SIZE):
        batch = turns[i:i + BATCH_SIZE]
        logger.info(f"[PODCAST] TTS batch {i//BATCH_SIZE + 1} ({len(batch)} turns)...")
        durations = await asyncio.gather(*[_process_turn(t) for t in batch])
        total_duration += sum(durations)

    # ── Step 4: FFmpeg stitch ─────────────────────────────────────────────────
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
