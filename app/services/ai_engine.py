"""
Nabda — AI Engine (Hybrid Groq + Gemini)
============================================
Core LLM orchestration: system prompts, JSON enforcement, hybrid failover.
Supports: Question Bank, Mindmap (Mermaid), Video Script, Podcast Script.
"""

import json
import re
import logging
import asyncio
from typing import Optional, Dict, Any

import google.generativeai as genai
from groq import AsyncGroq

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Clients Initialization ────────────────────────────────────────────────────
logger.info(f"[INIT] AI_PROVIDER set to: {settings.AI_PROVIDER}")

groq_client: Optional[AsyncGroq] = None
if settings.GROQ_API_KEY:
    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    logger.info("[INIT] ✓ Groq client initialized")
else:
    logger.warning("[INIT] ✗ Groq API key missing")

if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY, transport="rest")
    logger.info("[INIT] ✓ Gemini client initialized")
else:
    logger.warning("[INIT] ✗ Google API key missing")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GROUNDING_PREAMBLE = (
    "CRITICAL RULES:\n"
    "1. You MUST strictly adhere to the provided document/text.\n"
    "2. If information is NOT present in the text, refuse to answer or mark as 'insufficient information'.\n"
    "3. Do NOT use any external knowledge beyond the given text.\n"
    "4. Do NOT hallucinate or invent facts.\n"
    "5. Your ENTIRE output (JSON values, dialogue, titles, descriptions) MUST be in Arabic, "
    "regardless of the input document's original language. Translate all content to Arabic.\n\n"
)

# ── 1. Question Bank (Triple-Layer JSON Enforcement) ─────────────────────────

QUESTION_BANK_SYSTEM_PROMPT = (
    "You are a JSON-only API. You NEVER output anything except a single valid JSON object.\n"
    "No markdown. No code fences. No ```json. No explanatory text. No greetings. No apologies.\n"
    "Your entire response is one JSON object. Nothing before it, nothing after it.\n"
    "The FIRST character of your response MUST be { and the LAST character MUST be }.\n\n"
    + GROUNDING_PREAMBLE
    + "You are an expert educational assessment designer.\n"
    "Create a question bank based ONLY on the user's provided text using Bloom's Taxonomy.\n"
    "Output MUST be valid JSON matching this EXACT schema:\n"
    "{\n"
    '  "title": "string (Arabic)",\n'
    '  "questions": [\n'
    "    {\n"
    '      "question": "string (Arabic)",\n'
    '      "options": [\n'
    '        { "text": "string", "is_correct": false },\n'
    '        { "text": "string", "is_correct": false },\n'
    '        { "text": "string", "is_correct": true },\n'
    '        { "text": "string", "is_correct": false }\n'
    "      ],\n"
    '      "correct_answer_index": 2,\n'
    '      "explanation": "string (cite from source text)",\n'
    '      "difficulty": "easy|medium|hard"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Constraints:\n"
    "- Exactly 4 options per question. Exactly 1 correct answer per question.\n"
    "- correct_answer_index is the 0-based index of the correct option in the options array.\n"
    "- Explanations MUST reference the source text directly.\n"
    "- ALL text values MUST be in Arabic.\n"
)

# ── 2. Mindmap (Mermaid Syntax Output) ───────────────────────────────────────

MINDMAP_MERMAID_SYSTEM_PROMPT = (
    "You are a Mermaid diagram generator. Output ONLY valid Mermaid mindmap syntax.\n"
    "No markdown fences. No explanation. No ```mermaid wrapper. Just the raw Mermaid code.\n"
    "The FIRST word of your response MUST be 'mindmap'.\n\n"
    + GROUNDING_PREAMBLE
    + "Create a comprehensive, visually appealing mind map from the given text.\n\n"
    "Rules:\n"
    "- Start with 'mindmap' on the first line\n"
    "- Use proper Mermaid mindmap indentation (2 spaces per level)\n"
    "- Root node uses double parentheses: root((Topic Name))\n"
    "- Branch nodes are plain text with proper indentation\n"
    "- 3-4 levels of depth maximum\n"
    "- Max 6 words per node label\n"
    "- ALL labels MUST be in Arabic\n"
    "- Cover ALL major topics from the text\n"
    "- Create 4-7 main branches from the root\n\n"
    "Example output format:\n"
    "mindmap\n"
    "  root((الموضوع الرئيسي))\n"
    "    الفرع الأول\n"
    "      نقطة فرعية أ\n"
    "      نقطة فرعية ب\n"
    "    الفرع الثاني\n"
    "      نقطة فرعية ج\n"
)

# ── 3. Video Script (Word-Budget Enforced) ───────────────────────────────────

VIDEO_SCRIPT_SYSTEM_PROMPT = (
    "You are a JSON-only API. Output ONLY valid JSON. No markdown. No fences.\n"
    "The FIRST character MUST be {{ and the LAST MUST be }}.\n\n"
    "أنت مُعلم مصري متخصص في تبسيط المفاهيم العلمية بأسلوب شيق وممتع.\n"
    "Create a whiteboard video lesson with exactly {{num_segments}} segments.\n\n"
    "STRICT DURATION RULES (THIS IS CRITICAL):\n"
    "- Total narration word count MUST be between {min_words} and {max_words} Arabic words.\n"
    "- Each segment's narration_text MUST be 45-55 Arabic words (6-8 sentences).\n"
    "- You MUST generate exactly {{num_segments}} segments. No more, no less.\n"
    "- COUNT YOUR WORDS. If total is under {min_words}, add more detail.\n\n"
    "LANGUAGE RULES (CRITICAL):\n"
    "- ALL output MUST be in heavy Egyptian Colloquial Arabic.\n"
    "- Use casual Egyptian expressions (يعني، خلينا نقول، تعالوا نشوف، ببساطة كده).\n"
    "- If source is English, TRANSLATE to Egyptian Arabic.\n\n"
    "TTS SCRIPTING RULES (CRITICAL):\n"
    "- NEVER use emojis, markdown, asterisks, underscores, hashtags, or special symbols.\n"
    "- NEVER use bracket stage directions like [يضحك] or (يبتسم).\n"
    "- USE frequent commas (،) and periods (.) for natural TTS pauses.\n"
    "- SPELL OUT English acronyms phonetically in Arabic.\n\n"
    "VOICE ASSIGNMENT:\n"
    "- Odd segments: voice_id 1 (male). Even segments: voice_id 2 (female).\n\n"
    "IMAGE SCENES (CRITICAL FOR VISUAL VARIETY):\n"
    "- Each segment MUST have exactly {images_per_segment} image_scenes.\n"
    "- Each scene has a timestamp_offset (0, 8, 16 seconds) and an English image_prompt.\n"
    "- image_prompt: descriptive, concise, whiteboard style, educational, white background.\n\n"
    "Output JSON schema:\n"
    '{{\n'
    '  "title": "عنوان الدرس",\n'
    '  "segments": [\n'
    "    {{\n"
    '      "id": 1,\n'
    '      "title": "عنوان الشريحة",\n'
    '      "bullet_points": ["نقطة ١", "نقطة ٢", "نقطة ٣"],\n'
    '      "narration_text": "النص المسموع بالمصري — 45 to 55 words",\n'
    '      "voice_id": 1,\n'
    '      "image_scenes": [\n'
    '        {{"timestamp_offset": 0, "image_prompt": "English prompt for scene 1"}},\n'
    '        {{"timestamp_offset": 8, "image_prompt": "English prompt for scene 2"}},\n'
    '        {{"timestamp_offset": 16, "image_prompt": "English prompt for scene 3"}}\n'
    "      ]\n"
    "    }}\n"
    "  ]\n"
    "}}\n"
)

# ── 4. Podcast Script (Word-Budget Enforced) ─────────────────────────────────

PODCAST_SCRIPT_SYSTEM_PROMPT = (
    "You are a JSON-only API. Output ONLY valid JSON. No markdown. No fences.\n"
    "The FIRST character MUST be {{ and the LAST MUST be }}.\n\n"
    "أنت كاتب سيناريو لبودكاست مصري تعليمي ممتع على طريقة التوك شو.\n"
    "Create a lively talk-show conversation between THREE speakers.\n\n"
    "STRICT DURATION RULES (THIS IS CRITICAL):\n"
    "- Total dialogue word count MUST be between {min_words} and {max_words} Arabic words.\n"
    "- Generate exactly {{num_turns}} conversation turns.\n"
    "- VARY the length naturally:\n"
    "  * Short reactions: 8-12 words (e.g., أيوه بالظبط، ده اللي كنت عايز أقوله)\n"
    "  * Normal dialogue: 20-30 words\n"
    "  * Extended explanations (Guest): 40-60 words\n"
    "- COUNT YOUR WORDS. If total is under {min_words}, Guest should elaborate more.\n\n"
    "LANGUAGE RULES (CRITICAL):\n"
    "- ALL dialogue MUST be in heavy Egyptian Colloquial Arabic.\n"
    "- Use casual Egyptian expressions (يعني، أيوه بالظبط، طب سمعني الحتة دي).\n"
    "- ALL speakers including Guest MUST speak Egyptian Arabic.\n\n"
    "TTS SCRIPTING RULES (CRITICAL):\n"
    "- NEVER use emojis, markdown, asterisks, underscores, hashtags, or special symbols.\n"
    "- NEVER use bracket stage directions like [يضحك] or (يبتسم).\n"
    "- USE frequent commas (،) and periods (.) for TTS pauses.\n"
    "- SPELL OUT English acronyms phonetically in Arabic.\n\n"
    "Speakers:\n"
    "- Host1: Main host. Leads, asks questions, uses humor.\n"
    "- Host2: Co-host. Commentary, jokes, follow-ups.\n"
    "- Guest: Topic specialist. Deep explanations in Egyptian dialect.\n\n"
    "Output JSON schema:\n"
    '{{\n'
    '  "title": "عنوان الحلقة",\n'
    '  "description": "وصف مختصر للحلقة",\n'
    '  "speakers": ["Host1", "Host2", "Guest"],\n'
    '  "turns": [\n'
    "    {{\n"
    '      "id": 1,\n'
    '      "speaker": "Host1",\n'
    '      "text": "الكلام اللي المتحدث هيقوله"\n'
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Constraints:\n"
    "- Rotate between all three speakers organically\n"
    "- Cover ALL major topics from the source text\n"
    "- Start with Host1 introducing the topic and welcoming everyone\n"
    "- End with Host1 summarizing key takeaways\n"
)

# Legacy prompts (backward compatibility for text.py)
QUIZ_SYSTEM_PROMPT = QUESTION_BANK_SYSTEM_PROMPT
MINDMAP_SYSTEM_PROMPT = MINDMAP_MERMAID_SYSTEM_PROMPT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON ENFORCEMENT & PARSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor. Strips markdown code fences, preambles,
    and any text outside the JSON object.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty AI response received")

    cleaned = raw_text.strip()

    # Strip BOM
    cleaned = cleaned.lstrip('\ufeff')

    # Strategy 1: Remove ```json ... ``` wrapper
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 2: Find the first JSON boundary
    if not cleaned.startswith("{") and not cleaned.startswith("["):
        # Find first { or [
        obj_idx = cleaned.find("{")
        arr_idx = cleaned.find("[")

        if obj_idx == -1 and arr_idx == -1:
            raise ValueError("No JSON object found in AI response")

        if obj_idx == -1:
            start = arr_idx
        elif arr_idx == -1:
            start = obj_idx
        else:
            start = min(obj_idx, arr_idx)

        # Find corresponding closing bracket
        if cleaned[start] == "{":
            end = cleaned.rfind("}")
        else:
            end = cleaned.rfind("]")

        if end == -1:
            raise ValueError("Unclosed JSON in AI response")

        cleaned = cleaned[start:end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed. Raw (first 500 chars): {raw_text[:500]}")
        raise ValueError(f"AI returned invalid JSON: {str(e)}")


def enforce_pure_json(raw_text: str, expected_keys: list = None) -> Dict[str, Any]:
    """
    Triple-layer JSON enforcement: parse + validate structure.
    Raises ValueError with actionable error for retry logic.
    """
    parsed = clean_and_parse_json(raw_text)

    if expected_keys:
        missing = [k for k in expected_keys if k not in parsed]
        if missing:
            raise ValueError(f"JSON missing required keys: {missing}")

    return parsed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEXT CHUNKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_text(text: str, chunk_size: int = None) -> list[str]:
    """Split text into chunks respecting sentence boundaries."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current = ""
    sentences = re.split(r'(?<=[.!?؟。])\s+', text)

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > chunk_size and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE LLM CALLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _call_groq(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
    """Call Groq (Llama 3) with anti-hallucination settings."""
    if not groq_client:
        raise ValueError("Groq API Key missing")

    logger.info(f"Calling Groq ({settings.GROQ_MODEL})...")
    try:
        completion = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"} if json_mode else None,
                temperature=0,
                max_tokens=8000,
            ),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Groq call timed out after {settings.AI_TIMEOUT_SECONDS}s")
    result = completion.choices[0].message.content
    logger.info("✓ Groq call succeeded")
    return result


async def _call_gemini(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
    """Call Gemini with anti-hallucination settings."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Google API Key missing")

    logger.info(f"Calling Gemini ({settings.GEMINI_MODEL})...")
    config = {}
    if json_mode:
        config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={**config, "temperature": 0},
    )
    full_prompt = f"{system_prompt}\n\nUser Task:\n{user_prompt}"
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, full_prompt),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Gemini timed out after {settings.AI_TIMEOUT_SECONDS}s")
    logger.info("✓ Gemini call succeeded")
    return response.text


async def _hybrid_call(
    system_prompt: str,
    user_prompt: str,
    primary: str = "groq",
    json_mode: bool = True,
) -> str:
    """Execute with failover. If primary fails, try the other."""
    provider = settings.AI_PROVIDER

    if provider == "groq":
        callers = [("Groq", _call_groq)]
    elif provider == "gemini":
        callers = [("Gemini", _call_gemini)]
    else:  # hybrid
        if primary == "groq":
            callers = [("Groq", _call_groq), ("Gemini", _call_gemini)]
        else:
            callers = [("Gemini", _call_gemini), ("Groq", _call_groq)]

    last_error = None
    for name, caller in callers:
        try:
            return await caller(system_prompt, user_prompt, json_mode)
        except Exception as e:
            last_error = e
            logger.warning(f"{name} failed: {str(e)[:200]}. Trying next...")

    raise RuntimeError(f"All AI providers failed. Last error: {str(last_error)}")


async def _call_gemini_raw(prompt: str, json_mode: bool = True, temperature: float = 0.3) -> str:
    """Direct Gemini call with custom prompt (no system/user split). Used by pipelines."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Google API Key missing")

    config = {}
    if json_mode:
        config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={**config, "temperature": temperature},
    )

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Gemini timed out after {settings.AI_TIMEOUT_SECONDS}s")

    return response.text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HIGH-LEVEL GENERATORS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def generate_question_bank(
    text: str, num_questions: int = 50, difficulty: str = "medium"
) -> Dict[str, Any]:
    """Generate question bank with retry-on-validation-failure."""
    logger.info(f"[QBANK] Starting: {num_questions} questions, difficulty={difficulty}")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:5])[:settings.INPUT_TEXT_CAP]

    user_prompt = (
        f"Generate exactly {num_questions} {difficulty} questions from this text. "
        f"Cover ALL major topics comprehensively.\n\n{source_text}"
    )

    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            raw = await _hybrid_call(QUESTION_BANK_SYSTEM_PROMPT, user_prompt, primary="groq")
            parsed = enforce_pure_json(raw, expected_keys=["title", "questions"])
            logger.info(f"[QBANK] ✓ Generated {len(parsed.get('questions', []))} questions")
            return parsed
        except (ValueError, KeyError) as e:
            last_error = e
            logger.warning(f"[QBANK] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                user_prompt = (
                    f"Your previous output was INVALID: {str(e)[:200]}. "
                    f"Fix and retry. Output ONLY valid JSON.\n\n{user_prompt}"
                )

    raise ValueError(f"Question bank generation failed after {max_retries} attempts: {last_error}")


async def generate_mindmap_mermaid(text: str) -> str:
    """Generate Mermaid mindmap syntax from text. Returns raw Mermaid code."""
    logger.info("[MINDMAP] Starting Mermaid generation...")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:3])[:settings.INPUT_TEXT_CAP]

    user_prompt = f"Create a comprehensive mind map for:\n\n{source_text}"

    raw = await _hybrid_call(
        MINDMAP_MERMAID_SYSTEM_PROMPT, user_prompt, primary="gemini", json_mode=False
    )

    # Clean: strip any markdown fences
    mermaid_code = raw.strip()
    fence_match = re.search(r"```(?:mermaid)?\s*\n?(.*?)\n?\s*```", mermaid_code, re.DOTALL)
    if fence_match:
        mermaid_code = fence_match.group(1).strip()

    # Validate it starts with 'mindmap'
    if not mermaid_code.lower().startswith("mindmap"):
        mermaid_code = f"mindmap\n  root((الخريطة الذهنية))\n{mermaid_code}"

    logger.info(f"[MINDMAP] ✓ Generated {len(mermaid_code)} chars of Mermaid code")
    return mermaid_code


async def generate_video_script(text: str, num_segments: int = 25) -> Dict[str, Any]:
    """Generate video script JSON with word-budget enforcement."""
    logger.info(f"[VIDEO-SCRIPT] Generating {num_segments}-segment script...")

    prompt = VIDEO_SCRIPT_SYSTEM_PROMPT.format(
        num_segments=num_segments,
        min_words=settings.VIDEO_MIN_TOTAL_WORDS,
        max_words=settings.VIDEO_MAX_TOTAL_WORDS,
        images_per_segment=settings.VIDEO_IMAGES_PER_SEGMENT,
    )

    source = text[:settings.INPUT_TEXT_CAP]
    full_prompt = f"{prompt}\n\nSOURCE TEXT:\n{source}"

    raw = await _call_gemini_raw(full_prompt, json_mode=True, temperature=0.3)
    parsed = enforce_pure_json(raw, expected_keys=["title", "segments"])

    seg_count = len(parsed.get("segments", []))
    logger.info(f"[VIDEO-SCRIPT] ✓ Generated {seg_count} segments")
    return parsed


async def generate_podcast_script(text: str, num_turns: int = 55) -> Dict[str, Any]:
    """Generate podcast dialogue JSON with word-budget enforcement."""
    logger.info(f"[PODCAST-SCRIPT] Generating {num_turns}-turn script...")

    prompt = PODCAST_SCRIPT_SYSTEM_PROMPT.format(
        num_turns=num_turns,
        min_words=settings.PODCAST_MIN_TOTAL_WORDS,
        max_words=settings.PODCAST_MAX_TOTAL_WORDS,
    )

    source = text[:settings.INPUT_TEXT_CAP]
    full_prompt = f"{prompt}\n\nSOURCE TEXT:\n{source}"

    raw = await _call_gemini_raw(full_prompt, json_mode=True, temperature=0.5)
    parsed = enforce_pure_json(raw, expected_keys=["title", "turns"])

    turn_count = len(parsed.get("turns", []))
    logger.info(f"[PODCAST-SCRIPT] ✓ Generated {turn_count} turns")
    return parsed


# ── Legacy generators (backward compatibility for text.py) ────────────────────

async def generate_quiz(text: str, num_questions: int = 5, difficulty: str = "medium"):
    """Legacy quiz generator. Kept for backward compat."""
    from app.schemas.quiz import QuizResponse
    result = await generate_question_bank(text, num_questions, difficulty)
    return QuizResponse(**result)


async def generate_mindmap(text: str):
    """Legacy mindmap generator. Returns MindMapResponse (JSON tree)."""
    from app.schemas.mindmap import MindMapResponse
    logger.info("[MINDMAP-LEGACY] Starting generation...")
    chunks = chunk_text(text)
    source_text = " ".join(chunks[:3])

    legacy_prompt = (
        GROUNDING_PREAMBLE
        + "Summarize the text into a hierarchical Mind Map structure.\n"
        "Output MUST be valid JSON matching this exact schema:\n"
        "{\n"
        '  "root_node": {\n'
        '    "id": "string", "label": "string", "children": [\n'
        '      { "id": "string", "label": "string", "children": [...] }\n'
        "    ]\n"
        "  }\n"
        "}\n"
    )
    user_prompt = f"Create a comprehensive mind map for:\n\n{source_text}"
    raw = await _hybrid_call(legacy_prompt, user_prompt, primary="gemini")
    parsed = clean_and_parse_json(raw)
    return MindMapResponse(**parsed)
