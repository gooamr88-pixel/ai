"""
Nabda — Application Configuration
====================================
All settings loaded from environment variables / .env file.
Optimized for VPS/Docker deployment (no Vercel constraints).
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── AI Providers ──────────────────────────────────────────────────────────
    AI_PROVIDER: str = "hybrid"

    @field_validator("AI_PROVIDER")
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        allowed = {"hybrid", "groq", "gemini", "openai"}
        if v.lower() not in allowed:
            raise ValueError(f"AI_PROVIDER must be one of {allowed}, got '{v}'")
        return v.lower()

    # Groq (Llama 3 - High Speed)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Google (Gemini - Complex Reasoning)
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"

    # ── TTS Provider ──────────────────────────────────────────────────────────
    # Primary: "elevenlabs" (premium). Fallback: "edge" (free).
    TTS_PROVIDER: str = "elevenlabs"
    ELEVENLABS_API_KEY: Optional[str] = None

    # ── Image Generation (Replicate SDXL) ─────────────────────────────────────
    REPLICATE_API_TOKEN: Optional[str] = None
    IMAGE_MAX_CONCURRENT: int = 5          # parallel Replicate calls
    IMAGE_TIMEOUT_SECONDS: int = 60        # per-image timeout

    # ── Hugging Face (Legacy Fallback) ────────────────────────────────────────
    HF_API_TOKEN: Optional[str] = None

    # ── Video Generation ──────────────────────────────────────────────────────
    VIDEO_MAX_SEGMENTS: int = 30           # 30 segments for full video
    VIDEO_TARGET_DURATION_SECONDS: int = 480   # 8 min target
    VIDEO_IMAGES_PER_SEGMENT: int = 3      # 3 image scenes per segment
    VIDEO_WORDS_PER_SEGMENT: int = 50      # ~20s narration per segment
    VIDEO_MIN_TOTAL_WORDS: int = 1200      # floor for 7-min video
    VIDEO_MAX_TOTAL_WORDS: int = 1500      # ceiling for 10-min video

    # ── Podcast Generation ────────────────────────────────────────────────────
    PODCAST_MAX_SEGMENTS: int = 65         # max conversation turns
    PODCAST_TARGET_TURNS: int = 55         # default turns to request
    PODCAST_TARGET_DURATION_SECONDS: int = 480   # 8 min target
    PODCAST_MAX_DURATION_SECONDS: int = 660      # 11 min hard ceiling
    PODCAST_MIN_TOTAL_WORDS: int = 1000
    PODCAST_MAX_TOTAL_WORDS: int = 1300
    PODCAST_TTS_BATCH_SIZE: int = 10       # process 10 turns at a time

    # ── Limits ────────────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20
    CHUNK_SIZE: int = 12000                # chars per chunk (increased for VPS)
    AI_TIMEOUT_SECONDS: int = 120          # VPS has no 60s ceiling
    INPUT_TEXT_CAP: int = 12000            # max chars sent to LLM

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_STORAGE_BUCKET: str = "ruya-media"

    # ── Temp Directory (FFmpeg workspace) ─────────────────────────────────────
    TEMP_DIR: str = "/tmp/nabda_media"

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
