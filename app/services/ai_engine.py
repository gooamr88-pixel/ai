import json
import re
import logging
import asyncio
from typing import Optional, Dict, Any

import google.generativeai as genai
from groq import AsyncGroq

from app.core.config import settings
from app.core.config import settings
from app.schemas.quiz import QuizResponse, QuestionBank
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

QUIZ_SYSTEM_PROMPT = (
    GROUNDING_PREAMBLE +
    "You are an expert educational assessment designer. "
    "Create a quiz based ONLY on the user's provided text using Bloom's Taxonomy.\n"
    "Output MUST be valid JSON matching this exact schema:\n"
    "{\n"
    '  "title": "string",\n'
    '  "questions": [\n'
    "    {\n"
    '      "question": "string",\n'
    '      "options": [{ "text": "string", "is_correct": boolean }],\n'
    '      "explanation": "string (cite from the source text)",\n'
    '      "difficulty": "easy|medium|hard"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Constraints: Exactly 4 options per question. Exactly 1 correct answer per question. "
    "Explanations must reference the source text directly."
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

async def _call_groq(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
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
                max_tokens=8000,
            ),
            timeout=settings.AI_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Groq call timed out after {settings.AI_TIMEOUT_SECONDS}s")
    result = completion.choices[0].message.content
    logger.info("✓ Groq call succeeded")
    return result


# ── Core: Call Gemini ─────────────────────────────────────────────────────────

async def _call_gemini(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
    """Call Gemini with anti-hallucination settings and timeout."""
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


# ── Hybrid Call with Failover ─────────────────────────────────────────────────

async def _hybrid_call(
    system_prompt: str,
    user_prompt: str,
    primary: str = "groq",
    json_mode: bool = True,
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
            return await caller(system_prompt, user_prompt, json_mode)
        except Exception as e:
            last_error = e
            logger.warning(f"{name} failed: {str(e)[:200]}. Trying next provider...")

    raise RuntimeError(f"All AI providers failed. Last error: {str(last_error)}")


# ── Quiz Generation ───────────────────────────────────────────────────────────

async def generate_quiz(text: str, num_questions: int = 5, difficulty: str = "medium") -> QuizResponse:
    """Generate quiz using Hybrid AI: Groq primary, Gemini fallback."""
    logger.info(f"[QUIZ] Starting: {num_questions} questions, difficulty={difficulty}")

    # Chunk if needed — use first chunk for quiz generation
    chunks = chunk_text(text)
    source_text = chunks[0] if len(chunks) == 1 else " ".join(chunks[:3])

    user_prompt = f"Generate exactly {num_questions} {difficulty} questions from this text:\n\n{source_text}"

    raw = await _hybrid_call(QUIZ_SYSTEM_PROMPT, user_prompt, primary="groq")
    parsed = clean_and_parse_json(raw)
    return QuizResponse(**parsed)


# ── Question Bank Generation ──────────────────────────────────────────────────

async def generate_question_bank(text: str, num_questions: int = 50, difficulty: str = "medium") -> QuestionBank:
    """Generate a large question bank. Exactly 50 questions, STRICTLY English."""
    logger.info(f"[QBANK] Starting: {num_questions} questions, difficulty={difficulty}")

    chunks = chunk_text(text)
    source_text = " ".join(chunks[:5])  # Use more text for bigger question banks

    user_prompt = (
        f"The source document is in Arabic. You MUST TRANSLATE the information and generate the ENTIRE JSON response STRICTLY IN ENGLISH ONLY. Zero Arabic characters are allowed.\n"
        f"You MUST generate EXACTLY 30 Multiple Choice Questions (MCQ). Every single MCQ MUST have EXACTLY 4 options.\n"
        f"You MUST generate EXACTLY 20 True/False questions. Every single T/F question MUST have EXACTLY 2 options ('True' and 'False').\n"
        f"Cover ALL major topics and subtopics in the text comprehensively.\n\nSOURCE_DOCUMENT:\n{source_text}"
    )

    # For strict adherence, passing Pydantic schema to Gemini
    raw = await _hybrid_call(QUIZ_SYSTEM_PROMPT, user_prompt, primary="gemini")
    parsed_data = clean_and_parse_json(raw)

    # Manual Python Filtering & Truncation (Safety Guard)
    mcqs = []
    tfs = []
    
    questions = parsed_data.get("questions", [])
    for q in questions:
        opts = q.get("options", [])
        if len(opts) == 4:
            mcqs.append(q)
        elif len(opts) == 2:
            tfs.append(q)
        
    final_mcqs = mcqs[:30]
    final_tfs = tfs[:20]
    
    parsed_data["questions"] = final_mcqs + final_tfs

    return QuestionBank(**parsed_data)


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

