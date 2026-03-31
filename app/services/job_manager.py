"""
Nabda — In-Memory Job Manager
================================
Thread-safe async job store for background video/podcast generation.
Tracks status, progress percentage, and results for polling endpoints.
"""

import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """
    In-memory job store. Suitable for single-process VPS deployment.
    For multi-process/multi-node, swap to Redis-backed store.
    """

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, job_type: str) -> str:
        """Create a new job and return its ID."""
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "type": job_type,
                "status": JobStatus.PENDING,
                "progress": 0,
                "message": "Job queued",
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        logger.info(f"[JOB] Created {job_type} job: {job_id}")
        return job_id

    async def update_job(self, job_id: str, **kwargs):
        """Update arbitrary fields on a job."""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
                self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a snapshot of job state."""
        async with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    async def complete_job(self, job_id: str, result: Any):
        """Mark job as completed with result data."""
        await self.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            message="Completed successfully",
            result=result,
        )
        logger.info(f"[JOB] ✓ Completed: {job_id}")

    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        await self.update_job(
            job_id,
            status=JobStatus.FAILED,
            message=f"Failed: {error}",
            error=error,
        )
        logger.error(f"[JOB] ✗ Failed: {job_id} — {error}")


# ── Singleton ─────────────────────────────────────────────────────────────────
job_manager = JobManager()
