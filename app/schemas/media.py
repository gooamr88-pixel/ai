"""
Ruya — Media Schemas (Video & Podcast)
========================================
Request/response models for the Ruya Vision module.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SHARED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AudioSegment(BaseModel):
    """A single audio chunk encoded as Base64 data URI."""
    id: int
    text: str = Field(..., description="Script text narrated in this segment")
    audio_url: str = Field(default="", description="URL of the generated audio")
    duration_seconds: float = Field(..., description="Duration of this audio chunk")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VIDEO (WHITEBOARD)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoRequest(BaseModel):
    """Request body for whiteboard video generation."""
    text: str = Field(..., min_length=3, max_length=50000, description="Educational text to convert into video")
    num_segments: int = Field(default=5, ge=1, le=5, description="Number of video segments (hard limit: 5)")

class VideoSegment(BaseModel):
    """A single whiteboard slide with narration."""
    id: int
    title: str = Field(..., description="Slide headline")
    bullet_points: List[str] = Field(..., description="Key points to display on whiteboard")
    narration_text: str = Field(..., description="Script for narration audio")
    voice_id: int = Field(default=1, description="Voice ID: 1=male narrator, 2=female narrator")
    image_prompt: str = Field(default="", description="English prompt for AI whiteboard image generation")
    image_url: str = Field(default="", description="URL of the generated whiteboard image")
    audio_url: str = Field(default="", description="URL of the generated narration audio")
    duration_seconds: float = Field(default=0.0)

class VideoResponse(BaseModel):
    """Full whiteboard video response."""
    id: Optional[str] = None
    title: str
    segments: List[VideoSegment] = Field(default_factory=list, description="Individual video segments (empty when final_video_url is provided)")
    total_duration_seconds: float = 0.0
    final_video_url: str = Field(default="", description="URL or base64 data URI of the final stitched video")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PODCAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PodcastRequest(BaseModel):
    """Request body for podcast generation."""
    text: str = Field(..., min_length=3, max_length=50000, description="Educational text to convert into podcast")
    num_turns: int = Field(default=8, ge=2, le=20, description="Number of conversation turns")
    style: str = Field(default="educational", description="Conversation style: educational, casual, debate")

class PodcastTurn(BaseModel):
    """A single conversation turn in the podcast."""
    id: int
    speaker: str = Field(..., description="Speaker name (e.g. 'Host', 'Expert')")
    text: str = Field(..., description="What this speaker says")
    audio_url: str = Field(default="", description="URL of the generated speech audio")
    duration_seconds: float = Field(default=0.0)

class PodcastResponse(BaseModel):
    """Full podcast response."""
    id: Optional[str] = None
    title: str
    description: str
    speakers: List[str] = Field(..., description="List of speaker names")
    turns: List[PodcastTurn]
    total_duration_seconds: float = 0.0
