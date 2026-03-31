"""
Nabda — Unified Schema Package
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
    MindMapImageResponse,
)
from app.schemas.media import (
    AudioSegment,
    VideoImageScene,
    VideoRequest,
    VideoSegment,
    VideoResponse,
    PodcastRequest,
    PodcastLine,
    PodcastResponse,
)
from app.schemas.jobs import (
    JobResponse,
    JobCreatedResponse,
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
    "MindMapImageResponse",
    # Media
    "AudioSegment",
    "VideoImageScene",
    "VideoRequest",
    "VideoSegment",
    "VideoResponse",
    "PodcastRequest",
    "PodcastLine",
    "PodcastResponse",
    # Jobs
    "JobResponse",
    "JobCreatedResponse",
]
