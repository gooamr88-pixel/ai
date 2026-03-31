"""
Nabda — Media Schemas (Video & Podcast) — v2
================================================
Updated for multi-image-per-segment video and word-budget podcast.
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
# VIDEO (WHITEBOARD) — v2 with multi-image scenes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VideoImageScene(BaseModel):
    """A single image within a video segment, shown at a specific timestamp."""
    timestamp_offset: float = Field(0.0, description="Seconds after segment start to show this image")
    image_prompt: str = Field(default="", description="English prompt for AI image generation")
    image_url: str = Field(default="", description="URL of the generated image")


class VideoRequest(BaseModel):
    """Request body for whiteboard video generation."""
    text: str = Field(..., min_length=3, max_length=50000, description="Educational text to convert into video")
    num_segments: int = Field(default=25, ge=5, le=30, description="Number of video segments")


class VideoSegment(BaseModel):
    """A single whiteboard slide with narration and multiple image scenes."""
    id: int
    title: str = Field(..., description="Slide headline")
    bullet_points: List[str] = Field(default_factory=list, description="Key points to display on whiteboard")
    narration_text: str = Field(..., description="Script for narration audio")
    voice_id: int = Field(default=1, description="Voice ID: 1=male narrator, 2=female narrator")
    image_scenes: List[VideoImageScene] = Field(default_factory=list)
    audio_url: str = Field(default="")
    duration_seconds: float = Field(default=0.0)


class VideoResponse(BaseModel):
    """Full whiteboard video response (returned from completed job)."""
    id: Optional[str] = None
    title: str
    video_url: str = Field(default="", description="URL of the final stitched MP4")
    segments: List[VideoSegment] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_images: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PODCAST — v2 with word count tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PodcastRequest(BaseModel):
    """Request body for podcast generation."""
    text: str = Field(..., min_length=3, max_length=50000, description="Educational text to convert into podcast")
    num_turns: int = Field(default=55, ge=10, le=65, description="Number of conversation turns")
    style: str = Field(default="educational", description="Style: educational, casual, debate")


class PodcastLine(BaseModel):
    """A single conversation turn in the podcast."""
    id: int
    speaker: str = Field(..., description="Speaker name (e.g. 'Host1', 'Host2', 'Guest')")
    text: str = Field(..., description="Spoken dialogue")
    audio_url: str = Field(default="")
    duration_seconds: float = Field(default=0.0)
    word_count: int = Field(default=0, description="Word count for duration tracking")


class PodcastResponse(BaseModel):
    """Full podcast response (returned from completed job)."""
    id: Optional[str] = None
    title: str
    description: str = ""
    speakers: List[str] = Field(default_factory=list)
    podcast_url: str = Field(default="", description="URL of the final concatenated MP3")
    turns: List[PodcastLine] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    total_turns: int = 0
