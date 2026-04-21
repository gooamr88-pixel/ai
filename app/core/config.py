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

    # OpenAI (Legacy/Fallback)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo-0125"

    # Groq (Llama 3 - High Speed)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Google (Gemini - Complex Reasoning)
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"

    # ── ElevenLabs TTS ─────────────────────────────────────────────────────────
    ELEVENLABS_API_KEY: Optional[str] = None

    # ── Hugging Face (Image Generation) ───────────────────────────────────────
    HF_API_TOKEN: Optional[str] = None

    # ── Generation Limits (VPS-tuned) ─────────────────────────────────────────
    VIDEO_MAX_SEGMENTS: int = 14       # Dynamic: 6-12 segments based on PDF size ≈ 3-8 min video
    PODCAST_MAX_SEGMENTS: int = 20     # Dynamic: 10-18 turns based on PDF size ≈ 3-8 min podcast
    PODCAST_MAX_DURATION_SECONDS: int = 480  # 8 min max podcast

    # ── Limits ────────────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20
    CHUNK_SIZE: int = 8000  # chars per chunk for large PDFs
    CHUNK_TARGET_CHARS: int = 10000  # ~4K tokens per chunk for chunked generation
    AI_TIMEOUT_SECONDS: int = 120  # VPS — no serverless limit

    # ── Supabase ──────────────────────────────────────────────────────────────
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_STORAGE_BUCKET: str = "ruya-media"

    # ── Auth ──────────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = None  # Single master key (set via env)

    # ── Core ──────────────────────────────────────────────────────────────────
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
