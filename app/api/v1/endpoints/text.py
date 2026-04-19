"""
Ruya — Text Endpoints (Question Bank + Mindmap)
==================================================
Two separate endpoints:
  POST /text/generate-question-bank → flat JSON question bank (50 questions)
  POST /text/generate-mindmap       → mindmap image URL
"""

import logging
import asyncio
from fastapi import APIRouter, UploadFile, File, Request

from app.services.ai_engine import generate_question_bank, generate_mindmap
from app.services.mindmap_renderer import render_mindmap_image
from app.api.v1.utils import resolve_file_input
from app.core.limiter import limiter
from app.core.database import supabase

logger = logging.getLogger(__name__)

router = APIRouter()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. QUESTION BANK — Flat JSON (50 questions: 30 MCQ + 20 T/F)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/generate-question-bank")
@limiter.limit("5/minute")
async def generate_question_bank_endpoint(
    request: Request,
    file: UploadFile = File(...)
):
    """Generate a flat question bank (50 questions: 30 MCQ + 20 True/False).
    
    Returns:
        { "questions": [{ "question", "option_a", "option_b", "option_c", "option_d", "correct" }] }
    """
    
    # 1. Extract text from uploaded file
    resolved_text = await resolve_file_input(file)
    
    # 2. Build instructions for 50 questions
    qb_instructions = (
        f"{resolved_text}\n\n"
        "STRICT INSTRUCTION: You MUST generate exactly 50 questions based on the text above. "
        "Specifically: 30 Multiple Choice Questions (MCQs) and 20 True/False questions. "
        "For True/False questions, option_a='True', option_b='False', and option_c/option_d must be "
        "plausible distractor statements related to the question topic."
    )
    
    # 3. Generate question bank
    qb_res = await generate_question_bank(qb_instructions, num_questions=50)
    
    # 4. Save to database (best-effort)
    if supabase:
        try:
            await asyncio.to_thread(
                lambda: supabase.table("generated_quizzes").insert({
                    "title": "Question Bank",
                    "difficulty": "mixed",
                    "num_questions": len(qb_res.questions),
                    "quiz_data": qb_res.model_dump(),
                    "type": "question-bank"
                }).execute()
            )
        except Exception as e:
            logger.error(f"[DB] Insert question bank failed: {e}")
            
    return qb_res


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MINDMAP — Returns Professional Image URL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/text/generate-mindmap")
@limiter.limit("5/minute")
async def generate_mindmap_endpoint(
    request: Request,
    file: UploadFile = File(...)
):
    """Generate a professional mindmap image from uploaded file.
    
    Returns:
        { "mindmap_image_url": "https://..." }
    """
    
    # 1. Extract text from uploaded file
    resolved_text = await resolve_file_input(file)
    
    # 2. Generate mindmap tree structure (JSON)
    mindmap_res = await generate_mindmap(resolved_text)
    
    # 3. Render tree as professional image
    image_url = await render_mindmap_image(mindmap_res.root_node)
    
    # Convert relative /media/ paths to absolute URLs
    if image_url.startswith("/media/"):
        base_url = str(request.base_url).rstrip("/")
        image_url = base_url + image_url
    
    # 4. Save to database (best-effort)
    if supabase:
        try:
            await asyncio.to_thread(
                lambda: supabase.table("generated_mindmaps").insert({
                    "mindmap_data": mindmap_res.model_dump(exclude={"id"}),
                    "image_url": image_url
                }).execute()
            )
        except Exception as e:
            logger.error(f"[DB] Insert mindmap failed: {e}")
    
    return {"mindmap_image_url": image_url}
