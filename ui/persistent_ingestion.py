"""UI-facing helpers for persistent document ingestion via the existing orchestrator."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from types import GenericAlias
from typing import Any

from ui.upload_manager import is_allowed_extension, sanitize_filename


FAILED_STATUS = "FAILED"
MISSING_DATABASE_URL_MESSAGE = (
    "Persistent ingestion requires DATABASE_URL. Set DATABASE_URL or run the Docker Compose stack."
)


class PersistentIngestionSetupError(RuntimeError):
    """Expected setup/configuration error for persistent ingestion UI."""


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

    from agentic_rag.storage import (
        document_storage_config_from_env,
        get_postgres_engine,
        get_postgres_session_factory,
        postgres_config_from_env,
    )
    from agentic_rag.storage.models import Base

    postgres_config = postgres_config_from_env()
    if not postgres_config.enabled:
        raise PersistentIngestionSetupError(MISSING_DATABASE_URL_MESSAGE)

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
        return IngestionUIResult(None, None, None, FAILED_STATUS, "Upload filename metadata is missing.", False, False)

    safe_name = sanitize_filename(raw_name)
    if not is_allowed_extension(safe_name):
        return IngestionUIResult(None, None, None, FAILED_STATUS, "Unsupported file type. Allowed: pdf, md, txt.", False, False)

    try:
        content = uploaded_file.getvalue()
    except Exception as exc:
        return IngestionUIResult(None, None, None, FAILED_STATUS, f"Failed to read uploaded file: {type(exc).__name__}.", False, False)

    if isinstance(content, (type, GenericAlias)) or not isinstance(content, (bytes, bytearray, memoryview)):
        return IngestionUIResult(
            None,
            None,
            None,
            FAILED_STATUS,
            "Uploaded file payload is invalid. Expected bytes content.",
            False,
            False,
        )

    if isinstance(content, (bytearray, memoryview)):
        content = bytes(content)

    if not content:
        return IngestionUIResult(None, None, None, FAILED_STATUS, "Uploaded file is empty.", False, False)

    suffix = Path(safe_name).suffix.lower()
    temp_path = _write_temp_upload(content=content, suffix=suffix)

    try:
        from agentic_rag.storage.models import IngestionJob

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
    session: Any,
    temp_path: Path,
    source_name: str,
    source_type: str,
    document_store_root: Path,
):
    from agentic_rag.ingestion_pipeline import DocumentRegistry, IngestionOrchestrator
    from agentic_rag.storage import LocalDocumentStore

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
