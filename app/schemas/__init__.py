"""
Ruya — Unified Schema Package
===============================
Single source of truth for all Pydantic models.
Import from here: `from app.schemas import QuizRequest, MindMapNode, ...`
"""

from app.schemas.quiz import (
    Difficulty,
    QuizRequest,
    QuizOption,
    QuizQuestion,
    QuizResponse,
)
from app.schemas.mindmap import (
    MindMapRequest,
    MindMapNode,
    MindMapResponse,
)
from app.schemas.media import (
    AudioSegment,
    VideoRequest,
    VideoSegment,
    VideoResponse,
    PodcastRequest,
    PodcastLine,
    PodcastResponse,
)

__all__ = [
    # Quiz
    "Difficulty",
    "QuizRequest",
    "QuizOption",
    "QuizQuestion",
    "QuizResponse",
    # MindMap
    "MindMapRequest",
    "MindMapNode",
    "MindMapResponse",
    # Media
    "AudioSegment",
    "VideoRequest",
    "VideoSegment",
    "VideoResponse",
    "PodcastRequest",
    "PodcastLine",
    "PodcastResponse",
]
