from __future__ import annotations

from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
sessionmaker = pytest.importorskip("sqlalchemy.orm").sessionmaker

from agentic_rag.storage.models import Base, LifecycleStatus
from ui.persistent_ingestion import IngestionRuntime, ingest_uploaded_document


class DummyUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def _runtime(tmp_path: Path) -> IngestionRuntime:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return IngestionRuntime(
        session_factory=SessionFactory,
        document_store_root=tmp_path / "documents",
    )


def test_ingest_uploaded_document_rejects_unsupported_type(tmp_path: Path) -> None:
    result = ingest_uploaded_document(
        DummyUploadedFile("archive.zip", b"123"),
        runtime=_runtime(tmp_path),
    )

    assert result.status == LifecycleStatus.FAILED.value
    assert result.error_message == "Unsupported file type. Allowed: pdf, md, txt."
    assert result.document_id is None
    assert result.ingestion_job_id is None




def test_ingest_uploaded_document_rejects_type_payload(tmp_path: Path) -> None:
    class InvalidPayloadUpload:
        name = "bad.md"

        @staticmethod
        def getvalue():
            return type

    result = ingest_uploaded_document(InvalidPayloadUpload(), runtime=_runtime(tmp_path))

    assert result.status == LifecycleStatus.FAILED.value
    assert result.error_message == "Uploaded file payload is invalid. Expected bytes content."
    assert result.document_id is None
    assert result.ingestion_job_id is None
def test_ingest_uploaded_document_marks_duplicate_versions(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = ingest_uploaded_document(DummyUploadedFile("policy.md", b"# Policy\n\nBody"), runtime=runtime)
    second = ingest_uploaded_document(DummyUploadedFile("policy.md", b"# Policy\n\nBody"), runtime=runtime)

    assert first.status == LifecycleStatus.PROCESSING.value
    assert second.status == LifecycleStatus.PROCESSING.value
    assert first.document_id == second.document_id
    assert first.document_version_id == second.document_version_id
    assert second.duplicate_document is True
    assert second.status_from_database is True
