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
    "5. Your ENTIRE output (JSON values, dialogue, titles, descriptions) MUST be in Arabic, "
    "regardless of the input document's original language. Translate all content to Arabic.\n\n"
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
    "- Provide exactly 4 options:\n"
    "  - First option: { \"text\": \"صح\", \"isCorrect\": true/false }\n"
    "  - Second option: { \"text\": \"خطأ\", \"isCorrect\": true/false }\n"
    "  - Third and fourth: plausible distractor statements with isCorrect: false\n\n"
    "Output MUST be valid JSON matching this EXACT schema:\n"
    "{\n"
    '  "questions": [\n'
    "    {\n"
    '      "text": "نص السؤال بالعربي",\n'
    '      "type": "MCQ",\n'
    '      "options": [\n'
    '        { "text": "الخيار الأول", "isCorrect": true },\n'
    '        { "text": "الخيار الثاني", "isCorrect": false },\n'
    '        { "text": "الخيار الثالث", "isCorrect": false },\n'
    '        { "text": "الخيار الرابع", "isCorrect": false }\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Constraints:\n"
    "- Exactly 50 questions total (30 MCQ + 20 TF)\n"
    "- Each question MUST have exactly 4 options\n"
    "- Exactly 1 option per question must have isCorrect: true\n"
    "- All text MUST be in Arabic\n"
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
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed. Raw text (first 500 chars): {raw_text[:500]}")
        raise ValueError(f"AI returned invalid JSON: {str(e)}")


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

async def _hybrid_call(
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

    raw = await _hybrid_call(
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

    raw = await _hybrid_call(MINDMAP_SYSTEM_PROMPT, user_prompt, primary="gemini")
    parsed = clean_and_parse_json(raw)
    return MindMapResponse(**parsed)
