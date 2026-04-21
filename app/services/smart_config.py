"""
Ruya — Smart Configuration Calculator
========================================
Dynamically adjusts video/podcast generation parameters
based on the size of the input PDF text.

This ensures:
  • Small PDFs → short, focused content (3-4 min)
  • Large PDFs → comprehensive content (7-8 min)
  • Never exceeds 8 minutes to keep generation fast
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


def calculate_smart_config(text: str) -> GenerationConfig:
    """
    Calculate optimal video/podcast parameters based on input text length.
    
    The key insight: larger PDFs have more content to cover, so they need
    more segments/turns. But we cap at ~8 minutes to keep generation time
    reasonable (each segment = ~1 TTS call + 1 image + FFmpeg work).
    
    Podcast is FIXED at 20 turns (~8 minutes) regardless of PDF size.
    Video scales dynamically based on text length.
    
    Tiers:
      Small  (< 3K chars,  ~1-2 pages):  6 segments, 20 turns
      Medium (3-8K chars,   ~3-6 pages):  8 segments, 20 turns
      Large  (8-20K chars,  ~6-15 pages): 10 segments, 20 turns
      XLarge (20K+ chars,   ~15+ pages):  12 segments, 20 turns
    """
    char_count = len(text.strip()) if text else 0

    if char_count < 3000:
        config = GenerationConfig(
            video_segments=6,
            podcast_turns=20,
            num_chunks=1,
            images_per_segment=1,
            tier_name="small",
            estimated_duration_min=3.0,
            estimated_duration_max=4.0,
        )
    elif char_count < 8000:
        config = GenerationConfig(
            video_segments=8,
            podcast_turns=20,
            num_chunks=2,
            images_per_segment=1,
            tier_name="medium",
            estimated_duration_min=5.0,
            estimated_duration_max=6.0,
        )
    elif char_count < 20000:
        config = GenerationConfig(
            video_segments=10,
            podcast_turns=20,
            num_chunks=2,
            images_per_segment=1,
            tier_name="large",
            estimated_duration_min=6.0,
            estimated_duration_max=7.0,
        )
    else:
        config = GenerationConfig(
            video_segments=12,
            podcast_turns=20,
            num_chunks=3,
            images_per_segment=1,
            tier_name="xlarge",
            estimated_duration_min=7.0,
            estimated_duration_max=8.0,
        )

    logger.info(
        f"[SMART-CONFIG] Input: {char_count} chars → "
        f"Tier: {config.tier_name} | "
        f"Video: {config.video_segments} segments | "
        f"Podcast: {config.podcast_turns} turns | "
        f"Chunks: {config.num_chunks} | "
        f"Est: {config.estimated_duration_min}-{config.estimated_duration_max} min"
    )

    return config
