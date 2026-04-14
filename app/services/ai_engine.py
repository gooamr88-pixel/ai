"""
Ruya — AI Engine (Refactored)
===============================
Hybrid AI engine with Groq primary + Gemini fallback.
Provides: generate_question_bank(), generate_mindmap()
"""

import json
import re
import logging
import asyncio
from typing import Optional, Dict, Any

import google.generativeai as genai
from groq import AsyncGroq

from app.core.config import settings
from app.schemas.quiz import QuestionBankResponse
from app.schemas.mindmap import MindMapResponse

logger = logging.getLogger(__name__)

# ── Clients Initialization ────────────────────────────────────────────────────
logger.info(f"[INIT] AI_PROVIDER set to: {settings.AI_PROVIDER}")

groq_client: Optional[AsyncGroq] = None
if settings.GROQ_API_KEY:
    groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    logger.info("[INIT] ✓ Groq client initialized")
else:
    logger.warning("[INIT] ✗ Groq API key missing")


from google.generativeai import types 

if settings.GOOGLE_API_KEY:
   
    genai.configure(api_key=settings.GOOGLE_API_KEY, transport="rest")
    logger.info("[INIT] ✓ Gemini client initialized")


else:
    logger.warning("[INIT] ✗ Google API key missing")


# ── Anti-Hallucination System Prompts ─────────────────────────────────────────

GROUNDING_PREAMBLE = (
    "CRITICAL RULES:\n"
    "1. You MUST strictly adhere to the provided document/text.\n"
    "2. If information is NOT present in the text, refuse to answer or mark as 'insufficient information'.\n"
    "3. Do NOT use any external knowledge beyond the given text.\n"
    "4. Do NOT hallucinate or invent facts.\n"
    "5. Your ENTIRE output (JSON values, dialogue, titles, descriptions) MUST be in English, "
    "regardless of the input document's original language. Translate all content to English.\n\n"
)

# ── Question Bank System Prompt (Nested JSON Schema) ──────────────────────────

QUESTION_BANK_SYSTEM_PROMPT = (
    GROUNDING_PREAMBLE +
    "You are an expert educational assessment designer. "
    "Create a question bank based ONLY on the user's provided text using Bloom's Taxonomy.\n\n"
    "You MUST generate exactly 50 questions:\n"
    "- 30 Multiple Choice Questions (type: 'MCQ')\n"
    "- 20 True/False Questions (type: 'TF')\n\n"
    "For MCQ questions:\n"
    "- Provide exactly 4 options, each with 'text' and 'isCorrect' (boolean)\n"
    "- Exactly 1 option must have isCorrect: true\n"
    "- All 4 options must be plausible and educational\n\n"
    "For True/False questions:\n"
    "- type MUST be 'TF'\n"
    "- Provide exactly 2 options ONLY:\n"
    "  - { \"text\": \"True\", \"isCorrect\": true/false }\n"
    "  - { \"text\": \"False\", \"isCorrect\": true/false }\n"
    "- Do NOT add any extra options beyond these two\n\n"
    "Output MUST be valid JSON matching this EXACT schema:\n"
    "{\n"
    '  "questions": [\n'
    "    {\n"
    '      "text": "A multiple choice question",\n'
    '      "type": "MCQ",\n'
    '      "options": [\n'
    '        { "text": "First option", "isCorrect": true },\n'
    '        { "text": "Second option", "isCorrect": false },\n'
    '        { "text": "Third option", "isCorrect": false },\n'
    '        { "text": "Fourth option", "isCorrect": false }\n'
    "      ]\n"
    "    },\n"
    "    {\n"
    '      "text": "A true or false question",\n'
    '      "type": "TF",\n'
    '      "options": [\n'
    '        { "text": "True", "isCorrect": true },\n'
    '        { "text": "False", "isCorrect": false }\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Exactly 50 questions total (30 MCQ + 20 TF)\n"
    "- MCQ questions MUST have exactly 4 options\n"
    "- TF questions MUST have exactly 2 options (True and False only)\n"
    "- Exactly 1 option per question must have isCorrect: true\n"
    "- All text MUST be in English\n"
    "- Questions must cover ALL major topics in the text comprehensively\n"
    "- Mix difficulty levels: easy, medium, hard (using Bloom's Taxonomy)\n"
    "- Do NOT add any extra fields beyond the schema above"
)

MINDMAP_SYSTEM_PROMPT = (
    GROUNDING_PREAMBLE +
    "Summarize the text into a hierarchical Mind Map structure.\n"
    "Output MUST be valid JSON matching this exact schema:\n"
    "{\n"
    '  "root_node": {\n'
    '    "id": "string", "label": "string", "children": [\n'
    '      { "id": "string", "label": "string", "children": [...] }\n'
    "    ]\n"
    "  }\n"
    "}\n"
    "Constraints: 2-4 levels depth. Max 6 words per label. Unique IDs (use kebab-case like 'node-1').\n"
    "Create meaningful hierarchical groupings that reflect the document's structure."
)


# ── Helper: Robust JSON Recovery ──────────────────────────────────────────────

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Robust JSON extractor. Strips markdown code fences, preambles, and
    any text outside the JSON object. Uses regex to find the first valid
    JSON object in the response.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty AI response received")

    cleaned = raw_text.strip()

    # Strategy 1: Remove ```json ... ``` wrapper
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 2: Find the first { ... } block (greedy from first { to last })
    if not cleaned.startswith("{"):
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Strategy 3: Try repairing truncated JSON before giving up
        try:
            repaired = repair_truncated_json(cleaned)
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parse failed even after repair. Raw text (first 500 chars): {raw_text[:500]}")
            raise ValueError(f"AI returned invalid JSON: {str(e)}")


def repair_truncated_json(raw: str) -> str:
    """
    Attempt to repair JSON that was truncated mid-generation by Gemini.
    Handles:
      - Missing closing brackets/braces
      - Trailing commas before closing brackets
      - Incomplete key-value pairs or strings cut mid-way
    """
    if not raw or not raw.strip():
        raise ValueError("Empty string cannot be repaired")

    text = raw.strip()

    # Strip trailing incomplete string value (e.g., '"narration_text": "some text that got cu')
    # Find the last properly closed string
    # Remove any trailing content after the last complete JSON value
    # Look for patterns like: ,"key": "incomplete... (no closing quote)
    trailing_incomplete = re.search(
        r',\s*"[^"]*"\s*:\s*"[^"]*$', text
    )
    if trailing_incomplete:
        text = text[:trailing_incomplete.start()]
        logger.info("[JSON-REPAIR] Stripped trailing incomplete key-value pair")

    # Also handle truncated object inside an array: {..., "key": "val
    trailing_incomplete_obj = re.search(
        r',\s*\{[^}]*$', text
    )
    if trailing_incomplete_obj:
        text = text[:trailing_incomplete_obj.start()]
        logger.info("[JSON-REPAIR] Stripped trailing incomplete object")

    # Remove trailing commas (invalid JSON)
    text = re.sub(r',\s*$', '', text)

    # Count open vs close brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    # Close any unclosed brackets/braces
    missing_brackets = open_brackets - close_brackets
    missing_braces = open_braces - close_braces

    if missing_brackets > 0 or missing_braces > 0:
        # Remove any trailing comma before we close
        text = re.sub(r',\s*$', '', text.rstrip())
        text += ']' * missing_brackets
        text += '}' * missing_braces
        logger.info(
            f"[JSON-REPAIR] Closed {missing_brackets} brackets and {missing_braces} braces"
        )

    return text


# ── Text Chunking for Large Documents ─────────────────────────────────────────

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


def smart_chunk_text(text: str, num_chunks: int = 3) -> list[str]:
    """
    Split input text into N balanced chunks for the chunked generation pipeline.
    Each chunk targets CHUNK_TARGET_CHARS (~4K tokens) and respects sentence
    boundaries so no sentence is split mid-way.

    Unlike chunk_text() which splits by max size, this function splits into
    exactly `num_chunks` parts of roughly equal length.
    """
    if not text or not text.strip():
        return [text or ""]

    text = text.strip()
    target_per_chunk = max(len(text) // num_chunks, 1000)

    # Also enforce hard cap from settings
    target_per_chunk = min(target_per_chunk, settings.CHUNK_TARGET_CHARS)

    sentences = re.split(r'(?<=[.!?؟。،;])\s+', text)

    # If very few sentences, fall back to simple slicing
    if len(sentences) <= num_chunks:
        # Just do even character splits
        chunk_len = len(text) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_len
            end = (i + 1) * chunk_len if i < num_chunks - 1 else len(text)
            chunks.append(text[start:end].strip())
        return [c for c in chunks if c]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 > target_per_chunk and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    # If we ended up with more chunks than requested, merge the smallest trailing ones
    while len(chunks) > num_chunks and len(chunks) > 1:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    # If we ended up with fewer chunks, that's fine — just means the text was short
    logger.info(
        f"[CHUNK] Split {len(text)} chars into {len(chunks)} chunks: "
        f"{[len(c) for c in chunks]}"
    )
    return chunks if chunks else [text]


# ── Core: Call Groq ───────────────────────────────────────────────────────────

async def _call_groq(system_prompt: str, user_prompt: str, json_mode: bool = True, max_tokens: int = 8000) -> str:
    """Call Groq (Llama 3) with anti-hallucination settings and timeout."""
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
                max_tokens=max_tokens,
            ),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Groq call timed out after {settings.AI_TIMEOUT_SECONDS}s")
    result = completion.choices[0].message.content
    logger.info("✓ Groq call succeeded")
    return result


# ── Core: Call Gemini ─────────────────────────────────────────────────────────

async def _call_gemini(system_prompt: str, user_prompt: str, json_mode: bool = True, max_tokens: int = 8000) -> str:
    """Call Gemini with anti-hallucination settings and timeout."""
    if not settings.GOOGLE_API_KEY:
        raise ValueError("Google API Key missing")

    logger.info(f"Calling Gemini ({settings.GEMINI_MODEL})...")
    config = {}
    if json_mode:
        config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={**config, "temperature": 0, "max_output_tokens": max_tokens},
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


# ── Hybrid Call with Failover ─────────────────────────────────────────────────

async def hybrid_call(
    system_prompt: str,
    user_prompt: str,
    primary: str = "groq",
    json_mode: bool = True,
    max_tokens: int = 8000,
) -> str:
    """
    Execute with failover. If primary fails in hybrid mode, try the other.
    primary can be 'groq' or 'gemini'.
    """
    provider = settings.AI_PROVIDER

    # Determine call order
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
            return await caller(system_prompt, user_prompt, json_mode, max_tokens)
        except Exception as e:
            last_error = e
            logger.warning(f"{name} failed: {str(e)[:200]}. Trying next provider...")

    raise RuntimeError(f"All AI providers failed. Last error: {str(last_error)}")


# Backward-compatible alias
_hybrid_call = hybrid_call


# ── Question Bank Generation (Flat JSON) ──────────────────────────────────────

async def generate_question_bank(text: str, num_questions: int = 50) -> QuestionBankResponse:
    """Generate a flat question bank with 50 questions (30 MCQ + 20 T/F)."""
    logger.info(f"[QBANK] Starting: {num_questions} questions (30 MCQ + 20 T/F)")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:5])  # Use more text for comprehensive coverage

    user_prompt = (
        f"Generate exactly {num_questions} questions from this text. "
        f"30 must be Multiple Choice Questions and 20 must be True/False questions. "
        f"Cover ALL major topics and subtopics comprehensively.\n\n{source_text}"
    )

    raw = await hybrid_call(
        QUESTION_BANK_SYSTEM_PROMPT,
        user_prompt,
        primary="gemini",  # Gemini handles large structured output better
        max_tokens=16000,   # 50 questions need more tokens
    )
    parsed = clean_and_parse_json(raw)
    return QuestionBankResponse(**parsed)


# ── Mind Map Generation ───────────────────────────────────────────────────────

async def generate_mindmap(text: str) -> MindMapResponse:
    """Generate mind map using Hybrid AI: Gemini primary, Groq fallback."""
    logger.info("[MINDMAP] Starting generation...")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:3])

    user_prompt = f"Create a comprehensive mind map for:\n\n{source_text}"

    raw = await hybrid_call(MINDMAP_SYSTEM_PROMPT, user_prompt, primary="gemini")
    parsed = clean_and_parse_json(raw)
    return MindMapResponse(**parsed)
