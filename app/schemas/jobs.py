"""
Nabda — Job Schemas
=====================
Request/response models for the async job polling system.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


class JobResponse(BaseModel):
    """Response for GET /jobs/{job_id} polling endpoint."""
    job_id: str
    type: str = Field(..., description="Job type: 'video' or 'podcast'")
    status: str = Field(..., description="pending | processing | completed | failed")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Human-readable status message")
    result: Optional[Any] = Field(None, description="Final output data when completed")
    error: Optional[str] = Field(None, description="Error details if failed")
    created_at: str = ""
    updated_at: str = ""


class JobCreatedResponse(BaseModel):
    """Response for POST endpoints that queue background jobs."""
    job_id: str
    status: str = "pending"
    message: str = "Job queued for processing"
    poll_url: str = Field(..., description="URL to poll for job status")
