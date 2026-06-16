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
    Calculate optimal video/podcast parameters.
    
    The durations are now fixed:
      - Video: always 9 segments (~6.7 minutes, target 6-8 minutes)
      - Podcast: always 12 turns (~9.0 minutes, target 8-10 minutes)
      
    We still dynamically adjust the number of chunks (num_chunks) based on 
    the input PDF text size to ensure we process large documents reliably 
    without hitting LLM context/output token limits.
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
        video_segments=9,          # Fixed: ~6.7 mins (Target: 6-8 mins)
        podcast_turns=10,          # Fixed: ~7.5 mins (Target: 6-8 mins)
        num_chunks=num_chunks,
        images_per_segment=1,
        tier_name=tier_name,
        estimated_duration_min=6.0,
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
