"""
Ruya — Text AI Endpoints (Atomic / Vercel-Safe)
==================================================
Handles Quiz, Question Bank, Mind Map, and File Upload.
All endpoints return atomic JSON — NO SSE streaming.

Error handling is done via FastAPI global exception handlers in main.py.
Endpoints raise ValueError / RuntimeError and the handlers convert to
proper HTTP status codes (422 / 503 / 500).
"""

import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request

from app.services.ai_engine import (
    generate_quiz,
    generate_question_bank,
    generate_mindmap,
)
from app.api.v1.utils import resolve_text_input
from app.core.limiter import limiter
from app.core.database import supabase

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. UNIFIED EDUCATIONAL PACKAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/generate-educational-package")
@limiter.limit("5/minute")
async def generate_educational_package(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    num_questions: int = Form(5, ge=1, le=30),
    difficulty: str = Form("medium")
):
    """Generate Quiz, Question Bank, and Mind Map concurrently. Returns atomic JSON."""
    
    # 1. Resolve Text (throws 400 if both empty)
    resolved_text = await resolve_text_input(text, file)
    
    # 2. Concurrently generate all three
    quiz_task = generate_quiz(resolved_text, num_questions, difficulty)
    qb_task = generate_question_bank(resolved_text, num_questions, difficulty)
    mindmap_task = generate_mindmap(resolved_text)
    
    quiz_res, qb_res, mindmap_res = await asyncio.gather(quiz_task, qb_task, mindmap_task)
    
    # 3. Save to database (best-effort)
    if supabase:
        try:
            supabase.table("generated_quizzes").insert([
                {
                    "title": quiz_res.title,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "quiz_data": quiz_res.model_dump(exclude={"id"}),
                    "type": "quiz"
                },
                {
                    "title": qb_res.title,
                    "difficulty": difficulty,
                    "num_questions": num_questions,
                    "quiz_data": qb_res.model_dump(exclude={"id"}),
                    "type": "question-bank"
                }
            ]).execute()
            
            supabase.table("generated_mindmaps").insert({
                "mindmap_data": mindmap_res.model_dump(exclude={"id"})
            }).execute()
        except Exception as e:
            logger.error(f"[DB] Insert package failed: {e}")
            
    return {
        "quiz": quiz_res,
        "question_bank": qb_res,
        "mindmap": mindmap_res
    }
