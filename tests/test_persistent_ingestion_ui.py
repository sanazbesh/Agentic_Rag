from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
sessionmaker = pytest.importorskip("sqlalchemy.orm").sessionmaker

from agentic_rag.storage.models import Base, LifecycleStatus
from ui.persistent_ingestion import IngestionRuntime, ingest_uploaded_document
from ui.status_utils import normalize_status


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


def test_normalize_status_handles_enum_string_and_none() -> None:
    assert normalize_status(LifecycleStatus.PROCESSING) == "PROCESSING"
    assert normalize_status("READY") == "READY"
    assert normalize_status(None) is None


def test_ingest_uploaded_document_accepts_string_job_status(monkeypatch, tmp_path: Path) -> None:
    from ui import persistent_ingestion as module

    class DummySession:
        def get(self, _model, _job_id):
            return SimpleNamespace(status="PROCESSING", error_message=None)

    class DummySessionFactory:
        def __call__(self):
            return self

        def __enter__(self):
            return DummySession()

        def __exit__(self, exc_type, exc, tb):
            return False

    runtime = IngestionRuntime(
        session_factory=DummySessionFactory(),
        document_store_root=tmp_path / "documents",
    )

    monkeypatch.setattr(
        module,
        "_run_ingestion",
        lambda **_: SimpleNamespace(
            document_id="doc-1",
            document_version_id="ver-1",
            job_id="job-1",
            status=LifecycleStatus.PROCESSING,
            error_message=None,
            created_version=True,
        ),
    )

    result = ingest_uploaded_document(DummyUploadedFile("policy.md", b"# Policy"), runtime=runtime)

    assert result.status == "PROCESSING"
    assert result.ingestion_job_id == "job-1"
