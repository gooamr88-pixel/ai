"""
Ruya — Podcast Service (Refactored for 7-10 min duration)
===========================================================
Generates AI-powered conversational podcasts from educational text.
Uses three AI speakers (Host1, Host2, Guest) with longer, richer turns.

Architecture:
  Chunked generation — splits input into 3 chunks, generates 12+12+11
  turns sequentially to avoid Groq TPM limits and Gemini output
  truncation. Turns are merged and re-numbered 1-35.

Flow:
  1. smart_chunk_text → split input into 3 balanced chunks
  2. hybrid_call × 3 → generate 12+12+11 turns SEQUENTIALLY (JSON)
  3. Merge & re-number all turns
  4. Sanitise + truncate turns array
  5. ElevenLabs TTS in batches of 6 (avoid rate limiting)
  6. FFmpeg stitch all MP3s → final_podcast.mp3
  7. Upload to Supabase → return final_audio_url
"""

import re
import logging
import asyncio
from typing import List, Dict, Any


from app.core.config import settings
from app.services.tts_service import generate_tts_audio
from app.services.ai_engine import clean_and_parse_json, smart_chunk_text, hybrid_call
from app.services.ffmpeg_service import stitch_audio

logger = logging.getLogger(__name__)


PODCAST_SYSTEM_PROMPT = (
    "أنت كاتب سيناريو لبودكاست مصري تعليمي ممتع على طريقة التوك شو.\n"
    "Create a lively, natural, LONG talk-show conversation between THREE speakers about the given text.\n\n"
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


# ── Chunked Podcast Generation (12+12+11 = 35 turns) ─────────────────────────

async def _generate_chunk_turns(
    chunk_text_content: str,
    num_turns: int,
    chunk_index: int,
    total_chunks: int,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate a batch of podcast turns from a single text chunk.
    Uses hybrid_call (Gemini primary → Groq fallback) with safe token limits.
    """
    context_hint = ""
    if is_first_chunk:
        context_hint = (
            "هذا هو الجزء الأول من البودكاست. "
            "ابدأ بتقديم Host1 للموضوع والترحيب بـ Host2 و Guest.\n"
        )
    elif is_last_chunk:
        context_hint = (
            "هذا هو الجزء الأخير من البودكاست. "
            "اختم بتلخيص Host1 لأهم النقاط والتوديع.\n"
        )
    else:
        context_hint = (
            "هذا جزء وسط من البودكاست. "
            "استمر في النقاش بشكل طبيعي بدون مقدمة أو خاتمة.\n"
        )

    user_prompt = (
        f"أنت بتولّد الجزء {chunk_index + 1} من {total_chunks} لبودكاست تعليمي طويل.\n"
        f"{context_hint}"
        f"يجب أن تُولِّد بالضبط {num_turns} turn (EXACTLY {num_turns} turns).\n"
        f"كل turn لازم يحتوي على 60-120 كلمة (فقرة كاملة).\n"
        f"غطي كل المحتوى اللي في النص التالي بالتفصيل.\n\n"
        f"SOURCE TEXT:\n{chunk_text_content}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = await hybrid_call(
                system_prompt=PODCAST_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                primary="gemini",
                json_mode=True,
                max_tokens=6000,  # ~12 turns fit comfortably in 6K output tokens
            )
            parsed = clean_and_parse_json(raw)
            turns = parsed.get("turns", [])

            # Validate: accept if we got enough content
            total_words = sum(len(t.get("narration_text", "").split()) for t in turns)
            if len(turns) >= num_turns - 2 and total_words >= (num_turns * 40):
                logger.info(
                    f"[PODCAST] Chunk {chunk_index + 1}/{total_chunks}: "
                    f"✓ {len(turns)} turns, {total_words} words (attempt {attempt + 1})"
                )
                return turns[:num_turns], parsed.get("title", "")
            else:
                logger.warning(
                    f"[PODCAST] Chunk {chunk_index + 1} attempt {attempt + 1}: "
                    f"insufficient ({len(turns)} turns, {total_words} words). Retrying..."
                )
        except Exception as e:
            logger.error(
                f"[PODCAST] Chunk {chunk_index + 1} attempt {attempt + 1} failed: {e}"
            )
            if attempt == max_retries - 1:
                raise

    return [], ""


async def generate_podcast(text: str, num_turns: int = 35, style: str = "educational") -> dict:
    """
    Generate a full 7-10 minute podcast from educational text using CHUNKED generation.

    Architecture:
      1. Split input text into 3 balanced chunks
      2. Generate 12 + 12 + 11 turns SEQUENTIALLY (respects Groq 12K TPM)
      3. Merge & re-number all turns
      4. Sanitise turns
      5. ElevenLabs TTS in batches of 6
      6. FFmpeg stitch into final podcast
    """
    num_turns = min(num_turns, settings.PODCAST_MAX_SEGMENTS)
    logger.info(f"[PODCAST] ═══ Starting CHUNKED generation: {num_turns}-turn {style} podcast (7-10 min target) ═══")

    # ── Step 1: Split input into 3 chunks ─────────────────────────────────────
    NUM_CHUNKS = 3
    text_chunks = smart_chunk_text(text, num_chunks=NUM_CHUNKS)
    logger.info(f"[PODCAST] Split input into {len(text_chunks)} chunks: {[len(c) for c in text_chunks]}")

    # Distribute turns across chunks: 12 + 12 + 11 = 35
    turns_per_chunk = []
    remaining = num_turns
    for i in range(len(text_chunks)):
        if i < len(text_chunks) - 1:
            count = remaining // (len(text_chunks) - i)
        else:
            count = remaining
        turns_per_chunk.append(count)
        remaining -= count

    logger.info(f"[PODCAST] Turn distribution: {turns_per_chunk} (total={sum(turns_per_chunk)})")

    # ── Step 2: Generate turns SEQUENTIALLY per chunk ─────────────────────────
    all_turns: List[Dict[str, Any]] = []
    podcast_title = "بودكاست تعليمي"

    for i, (chunk, turn_count) in enumerate(zip(text_chunks, turns_per_chunk)):
        logger.info(f"[PODCAST] ─── Generating chunk {i + 1}/{len(text_chunks)} ({turn_count} turns) ───")
        chunk_turns, chunk_title = await _generate_chunk_turns(
            chunk_text_content=chunk,
            num_turns=turn_count,
            chunk_index=i,
            total_chunks=len(text_chunks),
            is_first_chunk=(i == 0),
            is_last_chunk=(i == len(text_chunks) - 1),
        )
        all_turns.extend(chunk_turns)

        # Use title from the first chunk
        if i == 0 and chunk_title:
            podcast_title = chunk_title

    # ── Step 3: Re-number & sanitise merged turns ────────────────────────────
    for idx, turn in enumerate(all_turns):
        turn["id"] = idx + 1

    turns = _sanitise_turns(all_turns, max_turns=num_turns)

    if not turns:
        raise RuntimeError("AI returned no valid turns for podcast")

    logger.info(
        f"[PODCAST] ═══ Merged & sanitised {len(turns)} turns "
        f"(total words: {sum(len(t.get('narration_text', '').split()) for t in turns)}) ═══"
    )

    # ── Step 4: Generate TTS audio in BATCHES of 6 ────────────────────────────
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

    # ── Step 5: FFmpeg stitch ─────────────────────────────────────────────────
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
        "title":                  podcast_title,
        "total_duration_seconds": round(total_duration, 2),
        "final_audio_url":        final_audio_url,
    }
