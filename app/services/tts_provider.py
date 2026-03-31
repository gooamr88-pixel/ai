"""
Nabda — Swappable TTS Provider
=================================
Primary: ElevenLabs (eleven_multilingual_v2) for premium Arabic audio.
Backup:  edge-tts (free, Microsoft Edge Neural voices).

Set TTS_PROVIDER=elevenlabs in .env for ElevenLabs (default).
Set TTS_PROVIDER=edge in .env to fall back to free edge-tts.

Usage:
    from app.services.tts_provider import tts_engine
    audio_bytes, duration = await tts_engine.synthesize("مرحبا", voice="host")
"""

import io
import os
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import edge_tts
from elevenlabs import AsyncElevenLabs

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── TTS Timeout (per-call) ───────────────────────────────────────────────────
TTS_TIMEOUT = 45  # seconds — ElevenLabs multilingual can be slower on long text


# ── Voice Mappings ────────────────────────────────────────────────────────────

EDGE_VOICES = {
    "host":    "ar-EG-ShakirNeural",    # Male Egyptian narrator
    "expert":  "ar-EG-SalmaNeural",     # Female Egyptian narrator
    "guest":   "ar-SA-HamedNeural",     # Male Saudi (variety)
    "default": "ar-EG-ShakirNeural",
}

ELEVENLABS_VOICES = {
    "host":    "onwK4e9ZLuTAKqWW03F9",  # Daniel — Deep, professional male
    "expert":  "EXAVITQu4vr4xnSDxMaL",  # Sarah — Warm, natural female
    "guest":   "nPczCjzI2devNBz1zQrb",  # Brian — Distinct male guest
    "default": "onwK4e9ZLuTAKqWW03F9",
}

# ElevenLabs model — best quality for Arabic with natural pronunciation
ELEVENLABS_MODEL = "eleven_multilingual_v2"
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"  # 128kbps, 44.1kHz MP3


# ── Abstract Provider ─────────────────────────────────────────────────────────

class TTSProvider(ABC):
    """Base class for all TTS providers. Implement synthesize()."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str = "default") -> Tuple[bytes, float]:
        """
        Convert text to speech audio.

        Returns:
            Tuple of (mp3_audio_bytes, estimated_duration_seconds)
        """
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ELEVENLABS PROVIDER (Primary — Premium)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ElevenLabsTTSProvider(TTSProvider):
    """
    Production-grade TTS via ElevenLabs AsyncElevenLabs client.

    Features:
    - Uses eleven_multilingual_v2 for natural Arabic pronunciation
    - Native async streaming (no thread pool needed)
    - Per-call timeout protection via asyncio.wait_for()
    - MP3 128kbps @ 44.1kHz output
    """

    def __init__(self, api_key: str):
        self._client = AsyncElevenLabs(api_key=api_key)
        logger.info("[TTS:ElevenLabs] ✓ Async client initialized")

    async def synthesize(self, text: str, voice: str = "default") -> Tuple[bytes, float]:
        voice_id = ELEVENLABS_VOICES.get(voice, ELEVENLABS_VOICES["default"])
        logger.info(
            f"[TTS:ElevenLabs] Generating: {len(text)} chars, "
            f"voice={voice} ({voice_id}), model={ELEVENLABS_MODEL}"
        )

        try:
            # ── Call ElevenLabs async API with timeout ──
            audio_generator = await asyncio.wait_for(
                self._client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=ELEVENLABS_MODEL,
                    output_format=ELEVENLABS_OUTPUT_FORMAT,
                ),
                timeout=TTS_TIMEOUT,
            )

            # ── Collect streamed audio chunks ──
            audio_chunks = []
            async for chunk in audio_generator:
                audio_chunks.append(chunk)

            audio_bytes = b"".join(audio_chunks)

            if not audio_bytes or len(audio_bytes) < 100:
                raise RuntimeError(
                    f"ElevenLabs returned insufficient audio data "
                    f"({len(audio_bytes)} bytes) for {len(text)} chars"
                )

            # ── Estimate duration from MP3 byte size ──
            # MP3 @ 128kbps = 16,000 bytes/second
            estimated_duration = len(audio_bytes) / 16_000.0

            logger.info(
                f"[TTS:ElevenLabs] ✓ {len(audio_bytes):,} bytes, "
                f"~{estimated_duration:.1f}s, voice={voice}"
            )
            return audio_bytes, estimated_duration

        except asyncio.TimeoutError:
            logger.error(f"[TTS:ElevenLabs] Timeout after {TTS_TIMEOUT}s for voice={voice}")
            raise RuntimeError(f"ElevenLabs TTS timed out after {TTS_TIMEOUT}s")
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"[TTS:ElevenLabs] Failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"ElevenLabs TTS failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDGE-TTS PROVIDER (Backup — Free)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EdgeTTSProvider(TTSProvider):
    """Free TTS via Microsoft Edge Neural voices. No API key required."""

    async def synthesize(self, text: str, voice: str = "default") -> Tuple[bytes, float]:
        voice_name = EDGE_VOICES.get(voice, EDGE_VOICES["default"])
        logger.info(f"[TTS:Edge] {len(text)} chars, voice={voice_name}")

        try:
            communicate = edge_tts.Communicate(text, voice_name)
            audio_buffer = io.BytesIO()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])

            audio_bytes = audio_buffer.getvalue()
            if not audio_bytes:
                raise RuntimeError("edge-tts returned empty audio")

            # edge-tts MP3 ≈ 48kbps mono → ~6000 bytes/second
            estimated_duration = len(audio_bytes) / 6000.0

            logger.info(
                f"[TTS:Edge] ✓ {len(audio_bytes):,} bytes, "
                f"~{estimated_duration:.1f}s"
            )
            return audio_bytes, estimated_duration

        except Exception as e:
            logger.error(f"[TTS:Edge] Failed: {e}")
            raise RuntimeError(f"Edge-TTS generation failed: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROVIDER FACTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _create_provider() -> TTSProvider:
    """
    Select TTS provider based on TTS_PROVIDER env var.

    Values:
      - "elevenlabs" → Premium ElevenLabs (requires ELEVENLABS_API_KEY)
      - "edge"       → Free Microsoft Edge Neural TTS

    Falls back to edge-tts if ElevenLabs is selected but API key is missing.
    """
    provider_name = settings.TTS_PROVIDER.lower()

    if provider_name == "elevenlabs":
        if not settings.ELEVENLABS_API_KEY:
            logger.warning(
                "[TTS] ElevenLabs selected but ELEVENLABS_API_KEY is missing. "
                "Falling back to edge-tts."
            )
            return EdgeTTSProvider()

        logger.info(
            f"[TTS] ✓ Using ElevenLabs provider "
            f"(model={ELEVENLABS_MODEL}, format={ELEVENLABS_OUTPUT_FORMAT})"
        )
        return ElevenLabsTTSProvider(api_key=settings.ELEVENLABS_API_KEY)

    if provider_name == "edge":
        logger.info("[TTS] ✓ Using Edge-TTS provider (free)")
        return EdgeTTSProvider()

    # Unknown provider → default to edge-tts with warning
    logger.warning(f"[TTS] Unknown TTS_PROVIDER='{provider_name}'. Falling back to edge-tts.")
    return EdgeTTSProvider()


# ── Module-Level Singleton ────────────────────────────────────────────────────
tts_engine = _create_provider()


# ── Convenience Function ─────────────────────────────────────────────────────

async def generate_tts_audio(text: str, voice: str = "default") -> Tuple[bytes, float]:
    """Generate TTS audio using the configured provider."""
    return await tts_engine.synthesize(text, voice)


# ── FFprobe Duration Helper ──────────────────────────────────────────────────

async def get_audio_duration_ffprobe(file_path: str) -> float:
    """Get precise audio duration using ffprobe. Use after saving to file."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip())
    except Exception as e:
        logger.warning(f"[FFprobe] Failed for {file_path}: {e}, estimating from file size")
        size = os.path.getsize(file_path)
        return size / 16_000.0  # 128kbps MP3 = 16KB/s
