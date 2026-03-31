"""
Ruya — Unified Schema Package
===============================
Single source of truth for all Pydantic models.
Import from here: `from app.schemas import QuestionBankQuestion, MindMapNode, ...`
"""

from app.schemas.quiz import (
    QuestionBankQuestion,
    QuestionBankResponse,
)
from app.schemas.mindmap import (
    MindMapRequest,
    MindMapNode,
    MindMapResponse,
)
from app.schemas.media import (
    VideoResponse,
    PodcastResponse,
)

__all__ = [
    # Question Bank
    "QuestionBankQuestion",
    "QuestionBankResponse",
    # MindMap
    "MindMapRequest",
    "MindMapNode",
    "MindMapResponse",
    # Media
    "VideoResponse",
    "PodcastResponse",
]
