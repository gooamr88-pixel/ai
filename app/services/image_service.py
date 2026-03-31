"""
Nabda — Image Generation Service (Replicate SDXL)
=====================================================
Parallel batch image generation for the video pipeline.
Supports concurrency control to respect API rate limits.

Usage:
    urls = await generate_images_batch(["prompt1", "prompt2", ...])
"""

import logging
import asyncio
from typing import List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

REPLICATE_API_URL = "https://api.replicate.com/v1/models/stability-ai/sdxl/predictions"


# ── Single Image Generation ──────────────────────────────────────────────────

async def generate_image(
    prompt: str,
    width: int = 1280,
    height: int = 720,
    style_suffix: str = (
        ", minimalist whiteboard sketch style, clean educational diagram, "
        "white background, continuous line drawing, professional illustration"
    ),
) -> Optional[str]:
    """
    Generate a single image via Replicate SDXL.
    Returns the URL of the generated image, or None on failure.
    """
    if not settings.REPLICATE_API_TOKEN:
        logger.warning("[IMAGE] REPLICATE_API_TOKEN not set — skipping")
        return None

    full_prompt = f"{prompt}{style_suffix}"
    headers = {
        "Authorization": f"Bearer {settings.REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # Replicate sync mode — waits up to 60s
    }

    try:
        async with httpx.AsyncClient(timeout=settings.IMAGE_TIMEOUT_SECONDS) as client:
            # Create prediction (sync mode with Prefer: wait)
            resp = await client.post(
                REPLICATE_API_URL,
                headers=headers,
                json={
                    "input": {
                        "prompt": full_prompt,
                        "width": width,
                        "height": height,
                        "num_outputs": 1,
                        "scheduler": "K_EULER",
                        "num_inference_steps": 25,
                        "guidance_scale": 7.5,
                    }
                },
            )
            resp.raise_for_status()
            result = resp.json()

            # Sync mode returns completed prediction directly
            if result.get("status") == "succeeded":
                output = result.get("output", [])
                if output:
                    logger.info("[IMAGE] ✓ Generated image")
                    return output[0]

            # If sync mode didn't complete, fall back to polling
            if result.get("status") in ("starting", "processing"):
                return await _poll_prediction(client, result, headers)

            logger.warning(f"[IMAGE] Unexpected status: {result.get('status')}")
            return None

    except httpx.TimeoutException:
        logger.error(f"[IMAGE] Timeout after {settings.IMAGE_TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        logger.error(f"[IMAGE] Generation failed: {e}")
        return None


async def _poll_prediction(
    client: httpx.AsyncClient,
    prediction: dict,
    headers: dict,
    max_polls: int = 30,
    interval: float = 2.0,
) -> Optional[str]:
    """Poll a Replicate prediction until completion."""
    poll_url = prediction.get("urls", {}).get("get", "")
    if not poll_url:
        return None

    for _ in range(max_polls):
        await asyncio.sleep(interval)
        try:
            resp = await client.get(poll_url, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            if result["status"] == "succeeded":
                output = result.get("output", [])
                return output[0] if output else None
            elif result["status"] == "failed":
                logger.error(f"[IMAGE] Prediction failed: {result.get('error')}")
                return None
        except Exception as e:
            logger.warning(f"[IMAGE] Poll error: {e}")

    logger.error("[IMAGE] Prediction timed out during polling")
    return None


# ── Batch Image Generation ───────────────────────────────────────────────────

async def generate_images_batch(
    prompts: List[str],
    max_concurrent: int = None,
) -> List[Optional[str]]:
    """
    Generate multiple images in parallel with concurrency limiting.
    Returns list of URLs (None for each failure).
    """
    max_concurrent = max_concurrent or settings.IMAGE_MAX_CONCURRENT
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _limited(prompt: str) -> Optional[str]:
        async with semaphore:
            return await generate_image(prompt)

    logger.info(f"[IMAGE] Batch: {len(prompts)} images, concurrency={max_concurrent}")

    results = await asyncio.gather(*[_limited(p) for p in prompts])

    success = sum(1 for r in results if r is not None)
    logger.info(f"[IMAGE] Batch complete: {success}/{len(prompts)} succeeded")
    return list(results)


# ── Download Helper ──────────────────────────────────────────────────────────

async def download_image(url: str, dest_path: str) -> bool:
    """Download an image URL to a local file. Returns True on success."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        logger.warning(f"[IMAGE] Download failed for {url}: {e}")
        return False
