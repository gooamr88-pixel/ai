import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Request

from app.services.ai_engine import (
    generate_question_bank,
    generate_mindmap,
)
from app.api.v1.utils import resolve_file_input, resolve_text_input
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
    file: UploadFile = File(...)
):
    """Generate Question Bank, and Mind Map concurrently. Returns atomic JSON."""
    
    # 1. Resolve Text from uploaded file (throws 400 if invalid)
    resolved_text = await resolve_file_input(file)
    
    # Strictly enforce 50 questions (30 MCQ, 20 T/F)
    qb_instructions = (
        f"{resolved_text}\n\n"
        "STRICT INSTRUCTION: You MUST generate exactly 50 questions based on the text above. "
        "Specifically: 30 Multiple Choice Questions (MCQs) and 20 True/False questions. "
        "For True/False questions, the options should be True, False, and two dummy options to satisfy the 4-option schema."
    )
    
    # 2. Concurrently generate Question Bank and Mindmap
    qb_task = generate_question_bank(qb_instructions, num_questions=50, difficulty="medium")
    mindmap_task = generate_mindmap(resolved_text)
    
    qb_res, mindmap_res = await asyncio.gather(qb_task, mindmap_task)
    
    # 3. Save to database (best-effort)
    if supabase:
        try:
            supabase.table("generated_quizzes").insert({
                "title": qb_res.title,
                "difficulty": "medium",
                "num_questions": 50,
                "quiz_data": qb_res.model_dump(exclude={"id"}),
                "type": "question-bank"
            }).execute()
            
            supabase.table("generated_mindmaps").insert({
                "mindmap_data": mindmap_res.model_dump(exclude={"id"})
            }).execute()
        except Exception as e:
            logger.error(f"[DB] Insert package failed: {e}")
            
    return {
        "question_bank": qb_res,
        "mindmap": mindmap_res
    }
