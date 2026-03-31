"""
Nabda — Mindmap Endpoint
===========================
POST /api/v1/mindmap
Generates Mermaid mindmap syntax → encodes → returns mermaid.ink image URL.
"""

import base64
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Request

from app.services.ai_engine import generate_mindmap_mermaid
from app.api.v1.utils import resolve_text_input
from app.core.limiter import limiter
from app.core.database import supabase
from app.schemas.mindmap import MindMapImageResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def mermaid_to_image_url(mermaid_code: str) -> str:
    """
    Convert Mermaid syntax to a direct image URL via mermaid.ink.

    mermaid.ink renders Mermaid diagrams server-side and returns a PNG.
    URL format: https://mermaid.ink/img/{base64_encoded_code}
    """
    encoded = base64.urlsafe_b64encode(
        mermaid_code.encode("utf-8")
    ).decode("utf-8")
    return f"https://mermaid.ink/img/{encoded}"


@router.post("/mindmap", response_model=MindMapImageResponse)
@limiter.limit("5/minute")
async def create_mindmap(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Generate a professional mind map image.

    Workflow:
    1. LLM generates Mermaid mindmap syntax
    2. Server base64-encodes the Mermaid code
    3. Returns a mermaid.ink URL that directly renders as a PNG image

    Response: {"image_url": "https://mermaid.ink/img/...", "mermaid_code": "mindmap\n  ..."}
    """
    # 1. Resolve input
    resolved_text = await resolve_text_input(text, file)

    # 2. Generate Mermaid syntax from LLM
    mermaid_code = await generate_mindmap_mermaid(resolved_text)

    # 3. Build image URL
    image_url = mermaid_to_image_url(mermaid_code)

    # 4. Optionally save to Supabase
    supabase_url = None
    if supabase:
        try:
            db_result = supabase.table("generated_mindmaps").insert({
                "mermaid_code": mermaid_code,
                "image_url": image_url,
            }).execute()
            if db_result.data:
                logger.info("[MINDMAP] ✓ Saved to database")
        except Exception as e:
            logger.error(f"[DB] Insert mindmap failed: {e}")

    return MindMapImageResponse(
        image_url=image_url,
        mermaid_code=mermaid_code,
        supabase_url=supabase_url,
    )
