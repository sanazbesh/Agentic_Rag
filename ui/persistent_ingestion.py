"""UI-facing helpers for persistent document ingestion via the existing orchestrator."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from agentic_rag.ingestion_pipeline import DocumentRegistry, IngestionOrchestrator
from agentic_rag.storage import (
    LocalDocumentStore,
    document_storage_config_from_env,
    get_postgres_engine,
    get_postgres_session_factory,
    postgres_config_from_env,
)
from agentic_rag.storage.models import Base, IngestionJob, LifecycleStatus
from ui.upload_manager import is_allowed_extension, sanitize_filename


@dataclass(slots=True, frozen=True)
class IngestionUIResult:
    """Serializable payload for Streamlit ingestion status/result rendering."""

    document_id: str | None
    document_version_id: str | None
    ingestion_job_id: str | None
    status: str
    error_message: str | None
    duplicate_document: bool
    status_from_database: bool


@dataclass(slots=True, frozen=True)
class IngestionRuntime:
    """Runtime dependencies for one ingestion request."""

    session_factory: Any
    document_store_root: Path


def build_ingestion_runtime_from_env() -> IngestionRuntime:
    """Build DB session factory and document store root using existing config patterns."""

    postgres_config = postgres_config_from_env()
    if not postgres_config.enabled:
        raise RuntimeError("DATABASE_URL is required for persistent ingestion UI.")

    engine = get_postgres_engine(postgres_config)
    Base.metadata.create_all(engine)
    session_factory = get_postgres_session_factory(engine)
    storage_config = document_storage_config_from_env()
    return IngestionRuntime(
        session_factory=session_factory,
        document_store_root=storage_config.root_path,
    )


def ingest_uploaded_document(
    uploaded_file: Any,
    *,
    runtime: IngestionRuntime,
) -> IngestionUIResult:
    """Persist one uploaded file through the existing ingestion orchestrator."""

    raw_name = getattr(uploaded_file, "name", "")
    if not raw_name:
        return IngestionUIResult(
            document_id=None,
            document_version_id=None,
            ingestion_job_id=None,
            status=LifecycleStatus.FAILED.value,
            error_message="Upload filename metadata is missing.",
            duplicate_document=False,
            status_from_database=False,
        )

    safe_name = sanitize_filename(raw_name)
    if not is_allowed_extension(safe_name):
        return IngestionUIResult(
            document_id=None,
            document_version_id=None,
            ingestion_job_id=None,
            status=LifecycleStatus.FAILED.value,
            error_message="Unsupported file type. Allowed: pdf, md, txt.",
            duplicate_document=False,
            status_from_database=False,
        )

    try:
        content = uploaded_file.getvalue()
    except Exception as exc:
        return IngestionUIResult(
            document_id=None,
            document_version_id=None,
            ingestion_job_id=None,
            status=LifecycleStatus.FAILED.value,
            error_message=f"Failed to read uploaded file: {type(exc).__name__}.",
            duplicate_document=False,
            status_from_database=False,
        )

    if not content:
        return IngestionUIResult(
            document_id=None,
            document_version_id=None,
            ingestion_job_id=None,
            status=LifecycleStatus.FAILED.value,
            error_message="Uploaded file is empty.",
            duplicate_document=False,
            status_from_database=False,
        )

    suffix = Path(safe_name).suffix.lower()
    temp_path = _write_temp_upload(content=content, suffix=suffix)

    try:
        with runtime.session_factory() as session:
            result = _run_ingestion(
                session=session,
                temp_path=temp_path,
                source_name=safe_name,
                source_type=suffix.lstrip(".") or "file",
                document_store_root=runtime.document_store_root,
            )

            job = session.get(IngestionJob, result.job_id)
            status = (job.status if job is not None else result.status).value
            error_message = job.error_message if job is not None else result.error_message
            return IngestionUIResult(
                document_id=result.document_id,
                document_version_id=result.document_version_id,
                ingestion_job_id=result.job_id,
                status=status,
                error_message=error_message,
                duplicate_document=not result.created_version,
                status_from_database=job is not None,
            )
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _write_temp_upload(*, content: bytes, suffix: str) -> Path:
    fd, path = tempfile.mkstemp(prefix="agentic_rag_upload_", suffix=suffix)
    temp_path = Path(path)
    with os.fdopen(fd, "wb") as file_obj:
        file_obj.write(content)
    return temp_path


def _run_ingestion(
    *,
    session: Session,
    temp_path: Path,
    source_name: str,
    source_type: str,
    document_store_root: Path,
):
    registry = DocumentRegistry(session)
    document_store = LocalDocumentStore(document_store_root)
    orchestrator = IngestionOrchestrator(
        session=session,
        registry=registry,
        document_store=document_store,
    )
    return orchestrator.ingest_file(
        file_path=temp_path,
        source_name=source_name,
        source_type=source_type,
    )
