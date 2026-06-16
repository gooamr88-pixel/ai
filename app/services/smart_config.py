"""
Ruya — Smart Configuration Calculator
========================================
Dynamically adjusts video/podcast generation parameters
based on the size of the input PDF text.

DURATION ANCHORING (Mandate 1):
  • Both Video and Podcast target a 6:00–8:00 min playback window.
  • This target is INDEPENDENT of the uploaded PDF size:
      - Short PDFs  → the model is told to expand/scaffold concepts.
      - Long PDFs   → the model is told to synthesise/compress.
  • Structural counts (segments/turns) are FIXED so duration stays stable;
    only the number of input chunks scales with PDF size to avoid LLM
    context/output-token overflow.
  • Word budgets are derived mathematically from a target WPM so the
    runtime lands inside the window.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Duration anchoring constants (Mandate 1) ──────────────────────────────────
TARGET_DURATION_MIN_SEC = 360      # 6:00 — hard lower bound of the playback window
TARGET_DURATION_MAX_SEC = 480      # 8:00 — hard upper bound of the playback window
TARGET_DURATION_CENTER_SEC = 420   # 7:00 — word budgets aim here

# Speaking-rate assumptions for Arabic TTS (ElevenLabs multilingual v2).
#   CONTENT_WPM  → how many words we ask the model to WRITE (the budget).
#   CLIP_WPM     → intentionally lower (more generous) so the per-clip
#                  duration estimate used for ffmpeg `-t` is >= the real
#                  audio length and never truncates the audio tail.
CONTENT_WPM = 135
CLIP_WPM = 120

# Fixed structural counts — the Mandate 1 invariant (PDF-size independent).
# Sized GENEROUSLY because (a) the LLM reliably under-delivers vs the asked
# word count and (b) Arabic TTS with breathing pauses runs slower than the
# naive estimate. A top-up loop trims/extends to land inside the window.
VIDEO_SEGMENTS = 12
PODCAST_TURNS = 14

# Fixed per-block word budgets (NOT divided by count — keeping a high floor
# is what actually drives runtime, since the model shortens each block).
WORDS_PER_SEGMENT = 110
WORDS_PER_TURN = 100

# Top-up loop bounds (Mandate 1 guarantee).
MAX_TOPUP_ROUNDS = 2
HARD_MAX_SEGMENTS = 20
HARD_MAX_TURNS = 26


def estimate_clip_seconds(word_count: int) -> float:
    """Generous per-narration-block duration estimate, in seconds.

    Uses CLIP_WPM (lower than CONTENT_WPM) so the estimate is >= the real
    synthesized audio, preventing ffmpeg `-t` from cutting the audio short.
    Single shared source of truth for both video segments and podcast turns.
    """
    if word_count <= 0:
        return 0.0
    return round(word_count / CLIP_WPM * 60.0, 2)


@dataclass
class GenerationConfig:
    """Dynamic generation parameters calculated from input text size."""
    video_segments: int
    podcast_turns: int
    num_chunks: int
    images_per_segment: int
    tier_name: str
    estimated_duration_min: float
    estimated_duration_max: float
    words_per_segment: int   # per video segment narration budget (Mandate 1)
    words_per_turn: int      # per podcast turn narration budget (Mandate 1)


def calculate_smart_config(text: str) -> GenerationConfig:
    """
    Calculate optimal video/podcast parameters.

    Structural counts are FIXED to anchor duration into the 6–8 min window:
      - Video:   9 segments
      - Podcast: 10 turns

    Per-block word budgets are derived from CONTENT_WPM so total runtime
    centres on ~7 minutes regardless of PDF size. Only num_chunks scales
    with the input size to keep each LLM call within context/token limits.
    """
    char_count = len(text.strip()) if text else 0

    # Dynamically determine the number of chunks to prevent context overflow,
    # but keep segments/turns fixed.
    if char_count < 3000:
        num_chunks = 1
        tier_name = "small"
    elif char_count < 8000:
        num_chunks = 2
        tier_name = "medium"
    elif char_count < 20000:
        num_chunks = 2
        tier_name = "large"
    else:
        num_chunks = 3
        tier_name = "xlarge"

    config = GenerationConfig(
        video_segments=VIDEO_SEGMENTS,
        podcast_turns=PODCAST_TURNS,
        num_chunks=num_chunks,
        images_per_segment=1,
        tier_name=tier_name,
        estimated_duration_min=round(TARGET_DURATION_MIN_SEC / 60.0, 1),
        estimated_duration_max=round(TARGET_DURATION_MAX_SEC / 60.0, 1),
        words_per_segment=WORDS_PER_SEGMENT,
        words_per_turn=WORDS_PER_TURN,
    )

    logger.info(
        f"[SMART-CONFIG] Input: {char_count} chars → "
        f"Tier: {config.tier_name} | "
        f"Video: {config.video_segments} segments × ~{WORDS_PER_SEGMENT}w | "
        f"Podcast: {config.podcast_turns} turns × ~{WORDS_PER_TURN}w | "
        f"Chunks: {config.num_chunks} | "
        f"Window: {config.estimated_duration_min}-{config.estimated_duration_max} min"
    )

    return config
