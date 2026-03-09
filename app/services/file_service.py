import io
import logging
import asyncio

import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


async def extract_text_from_file(file_content: bytes, filename: str) -> dict:
    """
    Professional file processing with PyMuPDF (PDF) and Gemini Vision (Images).
    Returns: {"text": str, "success": bool}
    """
    filename = filename.lower()

    # ── Validate file size ────────────────────────────
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_content) > max_bytes:
        raise ValueError(f"File exceeds {settings.MAX_FILE_SIZE_MB}MB limit.")

    if len(file_content) == 0:
        raise ValueError("File is empty.")

    try:
        if filename.endswith(".pdf"):
            text = await _extract_from_pdf(file_content)
        elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".heic")):
            text = await _extract_from_image(file_content)
        else:
            raise ValueError("Unsupported format. Use PDF, PNG, JPG, JPEG, WEBP, or HEIC.")

        if not text or not text.strip():
            raise ValueError("No text found in file.")

        return {"text": text.strip(), "success": True}

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"File processing failed for {filename}: {str(e)}")
        raise ValueError(f"Processing error: {str(e)}")


async def _extract_from_pdf(content: bytes) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    Runs in a thread pool to avoid blocking the async event loop.
    """
    def _process_pdf(data: bytes) -> str:
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                if doc.page_count == 0:
                    raise ValueError("PDF has no pages.")

                if doc.page_count > 200:
                    raise ValueError("PDF too large (>200 pages).")

                text_blocks = []
                for page in doc:
                    page_text = page.get_text("text")
                    if page_text.strip():
                        text_blocks.append(page_text)

                if not text_blocks:
                    raise ValueError("No text content found in PDF.")

                return "\n\n".join(text_blocks)
        except Exception as e:
            # Re-raise strict ValueError, otherwise wrap in ValueError
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"PDF extraction failed: {str(e)}")

    # Run the synchronous CPU-bound work in a thread
    return await asyncio.to_thread(_process_pdf, content)


async def _extract_from_image(content: bytes) -> str:
    """
    Extract text from images using gemini-2.5-flash-lite Vision.
    Professional OCR with structure preservation.
    """
    try:
        image = Image.open(io.BytesIO(content))

        # Validate dimensions
        w, h = image.size
        if w < 50 or h < 50:
            raise ValueError("Image too small to contain readable text.")

        # Use Gemini 1.5 Flash for speed
        model = genai.GenerativeModel(
            "gemini-2.5-flash-lite",
            generation_config={"temperature": 0},
        )

        prompt = (
            "Extract all legible text from this image accurately. "
            "Maintain the structure where possible."
        )

        response = await asyncio.to_thread(model.generate_content, [prompt, image])
        result = response.text.strip()

        if not result or result.lower() in ["no text found", "no_text_found"]:
            raise ValueError("No readable text in image.")

        return result

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Image OCR failed: {str(e)}")