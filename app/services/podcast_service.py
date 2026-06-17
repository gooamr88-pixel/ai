"""
Ruya — Podcast Service (Optimized v3 — Smart Duration)
=========================================================
Generates AI-powered conversational podcasts from educational text.
Uses three AI speakers (Host1, Host2, Guest) with longer, richer turns.
Duration dynamically calculated from PDF size (3-8 min target).

Architecture:
  Chunked generation — splits input into 1-3 chunks based on text size,
  generates turns sequentially to avoid Groq TPM limits.
  Turns are merged and re-numbered.

Flow:
  1. calculate_smart_config → determine optimal turn count
  2. smart_chunk_text → split input into balanced chunks
  3. hybrid_call × N → generate turns SEQUENTIALLY (JSON)
  4. Merge & re-number all turns
  5. Sanitise + truncate turns array
  6. ElevenLabs TTS in batches of 6 (avoid rate limiting)
  7. FFmpeg stitch all MP3s → final_podcast.mp3
  8. Upload to Supabase → return final_audio_url
"""

import re
import math
import logging
import asyncio
from typing import List, Dict, Any


from app.core.config import settings
from app.services.tts_service import generate_tts_audio, voice_id_for_speaker, SPEAKERS, DEFAULT_SPEAKER
from app.services.ai_engine import clean_and_parse_json, smart_chunk_text, hybrid_call
from app.services.smart_config import (
    calculate_smart_config,
    GenerationConfig,
    estimate_clip_seconds,
    TARGET_DURATION_MIN_SEC,
    TARGET_DURATION_MAX_SEC,
    MAX_TOPUP_ROUNDS,
    HARD_MAX_TURNS,
)
from app.services.ffmpeg_service import stitch_audio

logger = logging.getLogger(__name__)


PODCAST_SYSTEM_PROMPT = (
    "أنت كاتب سيناريو لبودكاست مصري تعليمي ممتع على طريقة التوك شو.\n"
    "Create a lively, natural, LONG talk-show conversation between THREE speakers about the given text.\n\n"
    "STRICT GROUNDING & ANTI-HALLUCINATION RULES (100% ACCURACY):\n"
    "- You MUST strictly adhere ONLY to the facts, details, definitions, and concepts provided in the source text.\n"
    "- NEVER invent, assume, or introduce any external facts, details, stats, figures, product names, or ideas not directly written in the source text.\n"
    "- If you need to expand/elaborate to meet the requested turn count, explain the existing concepts in the source text in more detail, break down the text's ideas step-by-step, or use hypothetical examples that directly illustrate the facts in the text. DO NOT import any external information.\n\n"
    "LANGUAGE RULES (CRITICAL):\n"
    "- ALL dialogue MUST be in heavy Egyptian Colloquial Arabic (اللهجة المصرية العامية الدارجة).\n"
    "- ALL dialogue MUST use Egyptian phrasing, idioms, and expressions.\n"
    "- All speakers MUST speak in natural Egyptian Arabic.\n"
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
    "- Each turn MUST be a FULL PARAGRAPH — 4 to 6 sentences.\n"
    "- Each turn MUST contain 85 to 110 words (target ~95 words).\n"
    "- Short 1-2 sentence turns are NOT acceptable. Make each turn substantial and rich.\n"
    "- Think of each turn as a full speaking block, not a quick back-and-forth.\n"
    "- الحلقة كلها لازم تطلع مدة تشغيل بين 6 و 8 دقايق، فوزّع الكلام عشان يملأ المدة دي.\n\n"
    "المتحدثون (3 أصدقاء بيتكلموا بشكل طبيعي بدون ألقاب): شريف، عبدالله، فريدة.\n"
    "- شريف: بيقود النقاش بطريقة شيقة وبخفة دم ولغة عامية مصرية.\n"
    "- عبدالله: بيعلّق ويطرح أسئلة متابعة وبيوضّح النقط الصعبة ببساطة.\n"
    "- فريدة: بتشارك بآراء وأمثلة وبتسأل وتوضّح بحماس.\n\n"
    "قاعدة الكلام المحايد بدون ضمائر (إلزامية وصارمة 100% — ممنوع أي استثناء):\n"
    "- ممنوع منعاً باتاً استخدام أي ضمير أو فعل موجّه لشخص بصيغة بتفرّق بين مذكّر ومؤنث. يعني ممنوع تماماً: إنتَ، إنتِ، قلتَ، قلتِ، عملتَ، عملتِ، عارف، عارفة، شايف، شايفة، فاهم، فاهمة، وأي صيغة 'يا فلان إنتَ/إنتِ'.\n"
    "- لما أي متحدث يرد على غيره، ينادي عليه بالاسم بس من غير ما يتبعه فعل أو ضمير يدل على النوع. مثال صح: (كلام مظبوط يا عبدالله)، (نقطة مهمة يا فريدة)، (تمام يا شريف). مثال غلط: (إنتَ قلت يا عبدالله)، (إنتِ قلتِ يا فريدة).\n"
    "- استخدم دايماً صيغة الجمع أو المحايد بدل المفرد المذكّر/المؤنث: قول (إحنا شايفين، نقدر نقول، خلينا نوضّح، الكلام ده معناه، ممكن نقول، تعالوا نشوف) بدل (أنا شايف / أنا شايفة).\n"
    "- لو محتاج تنسب كلام أو فكرة لحد، قول (الكلام اللي اتقال، النقطة اللي فاتت، زي ما اتقال) من غير ضمير بيدل على نوع.\n"
    "- خلّي الحوار عن المحتوى والأفكار نفسها أكتر من مخاطبة الأشخاص بشكل شخصي.\n"
    "- الأسماء (يا شريف، يا عبدالله، يا فريدة) للنداء بس، وممنوع الألقاب (دكتور، مهندس، أستاذ، حضرتك).\n\n"
    "Output MUST be valid JSON matching this schema:\n"
    "{\n"
    '  "title": "عنوان الحلقة",\n'
    '  "description": "وصف مختصر للحلقة",\n'
    '  "speakers": ["شريف", "عبدالله", "فريدة"],\n'
    '  "turns": [\n'
    "    {\n"
    '      "id": 1,\n'
    '      "speaker": "شريف",\n'
    '      "narration_text": "فقرة كاملة من الكلام — 4-6 جمل، 85-110 كلمة"\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- The 'speaker' value of EVERY turn MUST be exactly one of: شريف، عبدالله، فريدة (no other names, no titles).\n"
    "- Conversation must flow naturally like a real Egyptian talk show\n"
    "- Rotate between the three speakers organically\n"
    "- Cover ALL major topics from the source text\n"
    "- Each turn MUST be 85-110 words (full paragraph), NOT short sentences\n"
    "- التزم بقاعدة الكلام المحايد بدون ضمائر مذكّر/مؤنث موجّهة للأشخاص في كل turn من غير أي استثناء\n"
)


def _normalise_speaker(raw_speaker: str) -> str:
    """Map any model-emitted speaker name to a known registry speaker.

    Keeps gender/voice consistent (Mandate 2): an unrecognised or garbled
    name can never slip through and get the default/opposite-gender voice.
    """
    speaker = (raw_speaker or "").strip()
    if speaker in SPEAKERS:
        return speaker
    for name in SPEAKERS:
        if name in speaker:
            return name
    return DEFAULT_SPEAKER


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
            "speaker":        _normalise_speaker(turn.get("speaker")),
            "narration_text": narration,
            "audio_url":      turn.get("audio_url", ""),
            "duration_seconds": turn.get("duration_seconds", 0.0),
        })

    return sanitised


# ── Chunked Podcast Generation (dynamic turns) ─────────────────────────────

async def _generate_chunk_turns(
    chunk_text_content: str,
    num_turns: int,
    chunk_index: int,
    total_chunks: int,
    is_first_chunk: bool = False,
    is_last_chunk: bool = False,
    words_per_turn: int = 95,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Generate a batch of podcast turns from a single text chunk.
    Uses hybrid_call (Gemini primary → Groq fallback) with safe token limits.

    `words_per_turn` is the duration-anchored narration budget (Mandate 1).
    """
    word_lo = max(int(words_per_turn * 0.85), 70)
    word_hi = int(words_per_turn * 1.2)

    context_hint = ""
    if is_first_chunk:
        context_hint = (
            "هذا هو الجزء الأول من البودكاست. "
            "ابدأ بتقديم شريف للموضوع والترحيب بـ عبدالله وفريدة.\n"
        )
    elif is_last_chunk:
        context_hint = (
            "هذا هو الجزء الأخير من البودكاست. "
            "اختم بتلخيص شريف لأهم النقاط والتوديع، وشارك عبدالله وفريدة في الختام.\n"
        )
    else:
        context_hint = (
            "هذا جزء وسط من البودكاست. "
            "استمر في النقاش بين شريف وعبدالله وفريدة بشكل طبيعي بدون مقدمة أو خاتمة.\n"
        )

    user_prompt = (
        f"أنت بتولّد الجزء {chunk_index + 1} من {total_chunks} لبودكاست تعليمي طويل ومفصل وعميق.\n"
        f"{context_hint}"
        f"يجب أن تُولِّد بالضبط {num_turns} turn (EXACTLY {num_turns} turns) في مصفوفة الـ 'turns'. لا تولد أقل من ذلك تحت أي ظرف!\n"
        f"وزّع الأدوار بين المتحدثين الثلاثة (شريف، عبدالله، فريدة) بشكل طبيعي، والتزم بقواعد الضمائر: فريدة مؤنثة، وشريف وعبدالله مذكّران.\n"
        f"إذا كان النص المرفق (SOURCE TEXT) قصيرًا، يجب عليك التوسع في شرح المفاهيم بالتفصيل، وإعطاء أمثلة توضيحية وتطبيقات عملية، وإثراء الحوار والنقاش التفاعلي بين المتحدثين لملء الـ {num_turns} أدوار المطلوبة بالكامل. وإذا كان طويلاً، لخّص وادمج الأفكار.\n"
        f"كل turn لازم يحتوي على {word_lo}-{word_hi} كلمة (الهدف ~{words_per_turn} كلمة) في فقرة كاملة من 4-6 جمل.\n"
        f"غطي كل المحتوى اللي في النص التالي بالتفصيل وبشكل وافٍ.\n\n"
        f"SOURCE TEXT:\n{chunk_text_content}"
    )

    max_retries = 3

    # Track the best result across all attempts so we never return empty
    best_turns: List[Dict[str, Any]] = []
    best_title = ""
    best_word_count = 0

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

            # Track best result
            if total_words > best_word_count and len(turns) > 0:
                best_turns = turns
                best_title = parsed.get("title", "")
                best_word_count = total_words

            if len(turns) >= num_turns - 2 and total_words >= (num_turns * max(int(words_per_turn * 0.6), 45)):
                logger.info(
                    f"[PODCAST] Chunk {chunk_index + 1}/{total_chunks}: "
                    f"✓ {len(turns)} turns, {total_words} words (attempt {attempt + 1})"
                )
                return turns[:num_turns], parsed.get("title", "")
            else:
                logger.warning(
                    f"[PODCAST] Chunk {chunk_index + 1} attempt {attempt + 1}: "
                    f"insufficient ({len(turns)} turns, {total_words} words). "
                    f"{'Retrying...' if attempt < max_retries - 1 else 'Using best result.'}"
                )
        except Exception as e:
            logger.error(
                f"[PODCAST] Chunk {chunk_index + 1} attempt {attempt + 1} failed: {e}"
            )
            if attempt == max_retries - 1 and not best_turns:
                raise

    # Use the best result instead of returning empty
    if best_turns:
        best_avg = best_word_count / len(best_turns) if best_turns else 0
        logger.warning(
            f"[PODCAST] Chunk {chunk_index + 1}/{total_chunks}: "
            f"⚠ No attempt met threshold. Using BEST result: "
            f"{len(best_turns)} turns, {best_word_count} words "
            f"(avg {best_avg:.0f} wps)"
        )
        return best_turns[:num_turns], best_title

    return [], ""


# ── TTS synthesis helper (reused by main pass + top-up loop) ────────────────

async def _synthesize_turns(turns: List[Dict[str, Any]], batch_size: int = 6) -> None:
    """Synthesize TTS audio for each turn in place, in batches.

    Uses the speaker registry so فريدة → female voice and شريف/عبدالله → male
    voices (Mandate 2). On failure, sets an empty audio_url + estimated duration.
    """
    async def _process_turn(turn: dict) -> None:
        try:
            voice_id = voice_id_for_speaker(turn.get("speaker", DEFAULT_SPEAKER))
            audio_url, duration = await generate_tts_audio(turn["narration_text"], voice_id=voice_id)
            turn["audio_url"] = audio_url
            turn["duration_seconds"] = duration
        except Exception as e:
            logger.warning(f"[PODCAST] TTS failed for turn {turn.get('id')}: {e}")
            turn["audio_url"] = ""
            turn["duration_seconds"] = estimate_clip_seconds(len(turn["narration_text"].split()))

    for i in range(0, len(turns), batch_size):
        batch = turns[i:i + batch_size]
        logger.info(f"[PODCAST] TTS batch {i // batch_size + 1} ({len(batch)} turns)...")
        await asyncio.gather(*[_process_turn(t) for t in batch])


async def _generate_extra_turns(
    text_chunks: List[str],
    words_per_turn: int,
    extra_count: int,
    round_index: int,
) -> List[Dict[str, Any]]:
    """Generate `extra_count` additional middle-of-podcast turns for the top-up
    loop, drawn from a chunk chosen round-robin so we keep covering the text."""
    chunk = text_chunks[round_index % len(text_chunks)]
    new_turns, _ = await _generate_chunk_turns(
        chunk_text_content=chunk,
        num_turns=extra_count,
        chunk_index=0,
        total_chunks=1,
        is_first_chunk=False,
        is_last_chunk=False,
        words_per_turn=words_per_turn,
    )
    return new_turns


async def generate_podcast(text: str, num_turns: int = None, style: str = "educational", smart_cfg: GenerationConfig = None) -> dict:
    """
    Generate a podcast (3-8 min) from educational text using CHUNKED generation.
    Duration is dynamically calculated from PDF text size.

    Architecture:
      1. Calculate smart config from text size
      2. Split input text into balanced chunks
      3. Generate turns SEQUENTIALLY per chunk
      4. Merge & re-number all turns
      5. Sanitise turns
      6. ElevenLabs TTS in batches of 6
      7. FFmpeg stitch into final podcast
    """
    # Use smart config if not provided
    if smart_cfg is None:
        smart_cfg = calculate_smart_config(text)
    if num_turns is None:
        num_turns = smart_cfg.podcast_turns
    num_turns = min(num_turns, settings.PODCAST_MAX_SEGMENTS)
    logger.info(f"[PODCAST] ═══ Starting CHUNKED generation: {num_turns}-turn {style} podcast ({smart_cfg.tier_name} tier, ~{smart_cfg.estimated_duration_min}-{smart_cfg.estimated_duration_max} min target) ═══")

    # ── PRE-FLIGHT: Validate ElevenLabs API key ─────────────────────────────
    if not settings.ELEVENLABS_API_KEY:
        raise RuntimeError(
            "[PODCAST-PREFLIGHT] FATAL: ELEVENLABS_API_KEY is not set. "
            "TTS will fail for every turn, producing 0 audio files, "
            "which causes FFmpeg to skip all clips. Aborting early."
        )

    # ── Step 1: Split input into dynamic chunks based on text size ──────────
    NUM_CHUNKS = smart_cfg.num_chunks
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
            words_per_turn=smart_cfg.words_per_turn,
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

    # ── Step 4: Generate TTS audio in batches ─────────────────────────────────
    await _synthesize_turns(turns)

    # ── Step 5: FFmpeg stitch (returns REAL probed duration) ──────────────────
    logger.info("[PODCAST] Stitching all audio turns with FFmpeg...")
    try:
        final_audio_url, real_duration = await stitch_audio(turns)
    except Exception as e:
        logger.error(f"[PODCAST] FFmpeg stitch failed: {e}")
        raise RuntimeError(f"Podcast audio generation failed: {e}")

    if not final_audio_url:
        raise RuntimeError("Podcast audio generation failed: no final audio URL was produced.")

    # ── Step 6: TOP-UP loop — extend until we clear the 6-min floor ───────────
    rounds = 0
    while (
        real_duration < TARGET_DURATION_MIN_SEC
        and rounds < MAX_TOPUP_ROUNDS
        and len(turns) < HARD_MAX_TURNS
    ):
        rounds += 1
        avg_per_turn = (real_duration / len(turns)) if turns else 20.0
        avg_per_turn = max(avg_per_turn, 8.0)  # guard against tiny averages
        deficit = TARGET_DURATION_MIN_SEC - real_duration
        extra = max(2, math.ceil(deficit / avg_per_turn) + 1)
        extra = min(extra, HARD_MAX_TURNS - len(turns))
        logger.warning(
            f"[PODCAST] ⏳ Top-up round {rounds}: {real_duration:.0f}s < "
            f"{TARGET_DURATION_MIN_SEC}s floor → generating {extra} more turns..."
        )

        new_turns = await _generate_extra_turns(text_chunks, smart_cfg.words_per_turn, extra, rounds)
        new_turns = _sanitise_turns(new_turns, max_turns=extra)
        if not new_turns:
            logger.warning("[PODCAST] Top-up produced no turns — stopping early.")
            break

        await _synthesize_turns(new_turns)
        turns.extend(new_turns)
        for idx, t in enumerate(turns):
            t["id"] = idx + 1

        final_audio_url, real_duration = await stitch_audio(turns)

    # ── Step 7: TRIM if we overshot the 8-min ceiling ─────────────────────────
    if real_duration > TARGET_DURATION_MAX_SEC and len(turns) > 4:
        avg_per_turn = max(real_duration / len(turns), 8.0)
        keep = max(4, int(TARGET_DURATION_MAX_SEC / avg_per_turn))
        if keep < len(turns):
            logger.info(
                f"[PODCAST] ✂ {real_duration:.0f}s over ceiling → trimming "
                f"{len(turns)} → {keep} turns."
            )
            turns = turns[:keep]
            final_audio_url, real_duration = await stitch_audio(turns)

    # ── Duration-window check (Mandate 1: target 6–8 min) ────────────────────
    if real_duration < TARGET_DURATION_MIN_SEC:
        logger.warning(
            f"[PODCAST] ⚠ Final duration {real_duration:.0f}s still BELOW 6-min floor "
            f"after {rounds} top-up round(s) — source text likely too short."
        )
    elif real_duration > TARGET_DURATION_MAX_SEC:
        logger.warning(f"[PODCAST] ⚠ Final duration {real_duration:.0f}s exceeds 8-min ceiling.")
    else:
        logger.info(f"[PODCAST] ✓ Final duration {real_duration:.0f}s is within the 6-8 min window.")

    logger.info(
        f"[PODCAST] ✓ Generated {len(turns)} turns, "
        f"{real_duration:.1f}s real | URL: {final_audio_url[:60]}"
    )

    return {
        "title":                  podcast_title,
        "total_duration_seconds": round(real_duration, 2),
        "final_audio_url":        final_audio_url,
    }
