"""
Ruya — Media Schemas (Video & Podcast)
========================================
Simplified response models — return URLs only, no segment/turn data.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIDEO (WHITEBOARD)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoResponse(BaseModel):
    """Simplified video response — URL only."""
    title: str = Field(..., description="Video title")
    total_duration_seconds: float = Field(default=0.0, description="Estimated total duration")
    final_video_url: str = Field(default="", description="URL of the final stitched video")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PODCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PodcastResponse(BaseModel):
    """Simplified podcast response — URL only."""
    title: str = Field(..., description="Podcast episode title")
    total_duration_seconds: float = Field(default=0.0, description="Total podcast duration")
    final_audio_url: str = Field(default="", description="URL of the final stitched audio")
