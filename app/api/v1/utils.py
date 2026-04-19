import logging
from typing import Optional

from fastapi import UploadFile, HTTPException

from app.services.file_service import extract_text_from_file
from app.core.config import settings

logger = logging.getLogger(__name__)

async def resolve_file_input(file: UploadFile) -> str:
    """
    Helper to extract text from a single required file upload.
    Raises 400 or 413 exceptions if the file is missing or if constraints fail.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = await extract_text_from_file(content, file.filename)
    if not result.get("success"):
        raise HTTPException(status_code=422, detail="Text extraction failed.")
    
    return result["text"]


async def resolve_text_input(text: Optional[str] = None, file: Optional[UploadFile] = None) -> str:
    """
    Helper to resolve text from either direct input or a file upload.
    If a file is provided, extracts text using extract_text_from_file.
    If text is provided, returns it directly.
    Raises 400 or 413 exceptions if both are missing or if constraints fail.
    """
    if file:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided.")

        content = await file.read()
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = await extract_text_from_file(content, file.filename)
        if not result.get("success"):
            raise HTTPException(status_code=422, detail="Text extraction failed.")
        
        return result["text"]

    if text and text.strip():
        return text.strip()

    raise HTTPException(status_code=400, detail="Either 'text' or 'file' must be provided.")


async def resolve_multi_pdf_input(files: list) -> str:
    """
    Validate and extract text from multiple uploaded PDF files.
    Raises appropriate HTTP exceptions on validation failure.
    Returns concatenated text from all files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    extracted_texts = []
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="عفواً، مسموح برفع ملفات PDF فقط.")

        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB.",
            )

        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file {file.filename} is empty.")

        result_dict = await extract_text_from_file(content, file.filename)
        if not result_dict.get("success"):
            raise HTTPException(status_code=422, detail=f"Text extraction failed for {file.filename}.")

        extracted_texts.append(result_dict["text"])

    return "\n\n---\n\n".join(extracted_texts)
