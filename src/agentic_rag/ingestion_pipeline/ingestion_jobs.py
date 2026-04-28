"""Service helpers for ingestion job lifecycle persistence and queries."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from agentic_rag.storage.models import IngestionJob, LifecycleStatus

JOB_ID_PREFIX = "job_"


class IngestionJobService:
    """Create, update, and query ingestion jobs for document processing attempts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_job(self, *, document_id: str, document_version_id: str) -> IngestionJob:
        job = IngestionJob(
            id=f"{JOB_ID_PREFIX}{uuid4().hex}",
            document_id=document_id,
            document_version_id=document_version_id,
            status=LifecycleStatus.PENDING,
        )
        self._session.add(job)
        self._session.flush()
        return job

    def mark_pending(self, job: IngestionJob) -> IngestionJob:
        job.status = LifecycleStatus.PENDING
        job.error_message = None
        job.started_at = None
        job.finished_at = None
        self._session.flush()
        return job

    def mark_processing(self, job: IngestionJob) -> IngestionJob:
        job.status = LifecycleStatus.PROCESSING
        job.error_message = None
        job.started_at = datetime.now(timezone.utc)
        job.finished_at = None
        self._session.flush()
        return job

    def mark_ready(self, job: IngestionJob) -> IngestionJob:
        job.status = LifecycleStatus.READY
        if job.started_at is None:
            job.started_at = datetime.now(timezone.utc)
        job.error_message = None
        job.finished_at = datetime.now(timezone.utc)
        self._session.flush()
        return job

    def mark_failed(self, job: IngestionJob, *, error_message: str) -> IngestionJob:
        job.status = LifecycleStatus.FAILED
        if job.started_at is None:
            job.started_at = datetime.now(timezone.utc)
        job.error_message = error_message
        job.finished_at = datetime.now(timezone.utc)
        self._session.flush()
        return job

    def list_recent_jobs(
        self,
        *,
        limit: int = 50,
        document_id: str | None = None,
        document_version_id: str | None = None,
        status: LifecycleStatus | None = None,
    ) -> list[IngestionJob]:
        stmt: Select[tuple[IngestionJob]] = select(IngestionJob)
        if document_id is not None:
            stmt = stmt.where(IngestionJob.document_id == document_id)
        if document_version_id is not None:
            stmt = stmt.where(IngestionJob.document_version_id == document_version_id)
        if status is not None:
            stmt = stmt.where(IngestionJob.status == status)

        stmt = stmt.order_by(IngestionJob.created_at.desc()).limit(limit)
        return list(self._session.scalars(stmt))


__all__ = ["IngestionJobService", "JOB_ID_PREFIX"]
