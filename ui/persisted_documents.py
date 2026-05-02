"""Lazy-loaded helpers for listing persisted documents in Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import GenericAlias
from typing import Any


@dataclass(slots=True, frozen=True)
class PersistedDocumentRow:
    document_id: str
    source_name: str
    source_type: str
    status: str
    current_version_id: str | None
    updated_at: datetime | None


class PersistedDocumentServiceError(RuntimeError):
    """Raised when persisted document listing cannot be completed."""


def _raise_if_type_object(value: Any, *, label: str) -> None:
    if isinstance(value, (type, GenericAlias)):
        raise PersistedDocumentServiceError(
            f"Invalid {label}: expected iterable/instances, got type object {value!r}."
        )


def list_persisted_documents() -> list[PersistedDocumentRow]:
    """List persisted documents from Postgres using lazy optional imports."""

    try:
        from sqlalchemy import select

        from agentic_rag.storage import get_postgres_engine, get_postgres_session_factory, postgres_config_from_env
        from agentic_rag.storage.models import Document
    except ModuleNotFoundError as exc:
        raise PersistedDocumentServiceError(f"Missing dependency: {exc.name or 'optional dependency'}") from exc

    postgres_config = postgres_config_from_env()
    if not postgres_config.enabled:
        raise PersistedDocumentServiceError("Persistent document database is not configured (DATABASE_URL missing).")

    try:
        engine = get_postgres_engine(postgres_config)
        session_factory = get_postgres_session_factory(engine)
        _raise_if_type_object(session_factory, label="session factory")
        with session_factory() as session:
            rows = session.execute(select(Document).order_by(Document.updated_at.desc())).scalars().all()
            _raise_if_type_object(rows, label="document query result")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        raise PersistedDocumentServiceError(f"Database unavailable: {type(exc).__name__}: {exc}") from exc

    return [
        PersistedDocumentRow(
            document_id=row.id,
            source_name=row.source_name,
            source_type=row.source_type,
            status=row.status.value if hasattr(row.status, "value") else str(row.status),
            current_version_id=row.current_version_id,
            updated_at=row.updated_at,
        )
        for row in rows
    ]


def ready_persisted_documents(rows: list[PersistedDocumentRow]) -> list[PersistedDocumentRow]:
    _raise_if_type_object(rows, label="persisted document rows")
    return [row for row in rows if row.status == "READY"]


def to_document_descriptor(row: PersistedDocumentRow) -> dict[str, Any]:
    return {
        "id": row.document_id,
        "name": row.source_name,
        "type": row.source_type,
        "source": "persisted",
        "document_id": row.document_id,
        "document_version_id": row.current_version_id,
        "status": row.status,
    }
