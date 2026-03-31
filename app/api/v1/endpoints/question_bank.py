"""
Nabda — Question Bank Endpoint
=================================
POST /api/v1/question-bank
Triple-layer JSON enforcement: strict prompt + response_format + server-side validation.
Returns PURE JSON — no markdown, no wrappers, no conversational text.
"""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from app.services.ai_engine import generate_question_bank
from app.api.v1.utils import resolve_text_input
from app.core.limiter import limiter
from app.core.database import supabase

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/question-bank")
@limiter.limit("5/minute")
async def create_question_bank(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    num_questions: int = Form(50),
    difficulty: str = Form("medium"),
):
    """
    Generate a question bank with PURE JSON output.

    Triple-layer enforcement:
    1. System prompt strictly forbids non-JSON output
    2. LLM API response_format = json_object
    3. Server-side enforce_pure_json() strips any remaining artifacts

    Returns raw JSON: {"title": "...", "questions": [...]}
    """
    # 1. Resolve input text
    resolved_text = await resolve_text_input(text, file)

    # 2. Build instruction overlay for 30 MCQ + 20 T/F
    instructions = (
        f"{resolved_text}\n\n"
        f"STRICT INSTRUCTION: Generate exactly {num_questions} questions. "
    )
    if num_questions >= 50:
        instructions += (
            "Specifically: 30 Multiple Choice Questions (MCQs) and 20 True/False questions. "
            "For True/False, options should be: True, False, and two dummy options."
        )

    # 3. Generate with retry-on-failure
    result = await generate_question_bank(
        instructions,
        num_questions=num_questions,
        difficulty=difficulty,
    )

    # 4. Save to DB (best-effort)
    if supabase:
        try:
            supabase.table("generated_quizzes").insert({
                "title": result.get("title", ""),
                "difficulty": difficulty,
                "num_questions": len(result.get("questions", [])),
                "quiz_data": result,
                "type": "question-bank",
            }).execute()
        except Exception as e:
            logger.error(f"[DB] Insert question bank failed: {e}")

    # 5. Return PURE JSON (JSONResponse ensures no Pydantic wrapping)
    return JSONResponse(
        content=result,
        media_type="application/json",
    )
