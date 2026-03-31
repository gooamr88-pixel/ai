from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request ──────────────────────────────────────────────────────────────────

class MindMapRequest(BaseModel):
    """Request body for mind map generation."""
    text: str = Field(..., min_length=3, max_length=50000, description="Educational text to summarize as a mind map")


# ── Response ─────────────────────────────────────────────────────────────────

class MindMapNode(BaseModel):
    """A single node in the mind map tree (recursive)."""
    id: str
    label: str
    children: Optional[List[MindMapNode]] = Field(default_factory=list)


class MindMapResponse(BaseModel):
    """Full mind map response returned to the client (legacy JSON tree)."""
    id: Optional[str] = None
    root_node: MindMapNode


class MindMapImageResponse(BaseModel):
    """Mind map response with rendered image URL (new endpoint)."""
    image_url: str = Field(..., description="Direct URL to rendered PNG image (mermaid.ink)")
    mermaid_code: str = Field(..., description="Raw Mermaid mindmap source code")
    supabase_url: Optional[str] = Field(None, description="Permanent Supabase URL if uploaded")
