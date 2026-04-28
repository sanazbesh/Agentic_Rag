from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.chunking.interfaces import Chunker
from agentic_rag.chunking.models import ChunkingResult
from agentic_rag.ingestion_pipeline.document_registry import DocumentRegistry
from agentic_rag.ingestion_pipeline.orchestrator import IngestionOrchestrator
from agentic_rag.storage.document_store import LocalDocumentStore
from agentic_rag.storage.models import Base, Document, DocumentVersion, IngestionJob, LifecycleStatus
from agentic_rag.types import Document as IngestedDocument


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(bind=engine) as db_session:
        yield db_session


class FailingChunker(Chunker):
    def chunk(self, document: IngestedDocument) -> ChunkingResult:
        raise RuntimeError("chunking exploded")


class RecoveringChunker(Chunker):
    def __init__(self) -> None:
        self.calls = 0

    def chunk(self, document: IngestedDocument) -> ChunkingResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first attempt failed")
        return ChunkingResult(parent_chunks=[], child_chunks=[])


def test_successful_orchestration_registers_and_stores_file(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "policy.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Policy\n\nBody text.", encoding="utf-8")

    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    orchestrator = IngestionOrchestrator(
        session=session,
        registry=registry,
        document_store=store,
    )

    result = orchestrator.ingest_file(source_file, source_type="upload")

    assert result.document_id
    assert result.document_version_id
    assert result.job_id
    assert result.status == LifecycleStatus.READY
    assert store.exists(result.storage_path) is True
    assert result.parsed_documents
    assert result.chunking_result is not None

    persisted_job = session.get(IngestionJob, result.job_id)
    persisted_document = session.get(Document, result.document_id)
    persisted_version = session.get(DocumentVersion, result.document_version_id)

    assert persisted_job is not None
    assert persisted_job.status == LifecycleStatus.READY
    assert persisted_job.started_at is not None
    assert persisted_job.finished_at is not None
    assert persisted_document is not None
    assert persisted_document.status == LifecycleStatus.READY
    assert persisted_version is not None
    assert persisted_version.status == LifecycleStatus.READY


def test_duplicate_content_reuses_existing_document_version(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "dup.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Same\n\nexact text", encoding="utf-8")

    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    orchestrator = IngestionOrchestrator(
        session=session,
        registry=registry,
        document_store=store,
    )

    first = orchestrator.ingest_file(source_file, source_type="upload")
    second = orchestrator.ingest_file(source_file, source_type="upload")

    all_versions = session.query(DocumentVersion).all()

    assert first.document_id == second.document_id
    assert first.document_version_id == second.document_version_id
    assert second.created_version is False
    assert len(all_versions) == 1


def test_failure_marks_job_document_and_version_failed(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "bad.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Will fail", encoding="utf-8")

    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    orchestrator = IngestionOrchestrator(
        session=session,
        registry=registry,
        document_store=store,
        chunker=FailingChunker(),
    )

    result = orchestrator.ingest_file(source_file, source_type="upload")

    assert result.status == LifecycleStatus.FAILED
    assert result.error_message == "chunking exploded"

    persisted_job = session.get(IngestionJob, result.job_id)
    persisted_document = session.get(Document, result.document_id)
    persisted_version = session.get(DocumentVersion, result.document_version_id)

    assert persisted_job is not None
    assert persisted_job.status == LifecycleStatus.FAILED
    assert persisted_job.started_at is not None
    assert persisted_job.finished_at is not None
    assert persisted_job.error_message == "chunking exploded"
    assert persisted_document is not None
    assert persisted_document.status == LifecycleStatus.FAILED
    assert persisted_version is not None
    assert persisted_version.status == LifecycleStatus.FAILED


def test_retry_only_allowed_for_failed_jobs(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "ok.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Success", encoding="utf-8")

    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    orchestrator = IngestionOrchestrator(session=session, registry=registry, document_store=store)
    result = orchestrator.ingest_file(source_file, source_type="upload")

    with pytest.raises(ValueError, match="Retry is only allowed for FAILED jobs"):
        orchestrator.retry_failed_job(result.job_id)


def test_retry_failed_job_reuses_storage_path_and_resets_error(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "retry.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Retry me", encoding="utf-8")

    chunker = RecoveringChunker()
    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    orchestrator = IngestionOrchestrator(session=session, registry=registry, document_store=store, chunker=chunker)

    failed = orchestrator.ingest_file(source_file, source_type="upload")
    assert failed.status == LifecycleStatus.FAILED

    retry = orchestrator.retry_failed_job(failed.job_id)
    assert retry.status == LifecycleStatus.READY
    assert retry.storage_path == failed.storage_path

    version = session.get(DocumentVersion, failed.document_version_id)
    assert version is not None
    assert version.storage_path == failed.storage_path
    assert version.status == LifecycleStatus.READY

    job = session.get(IngestionJob, failed.job_id)
    assert job is not None
    assert job.status == LifecycleStatus.READY
    assert job.error_message is None


def test_retry_failed_document_version_records_new_error(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "retry-fail.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Still failing", encoding="utf-8")

    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    orchestrator = IngestionOrchestrator(
        session=session,
        registry=registry,
        document_store=store,
        chunker=FailingChunker(),
    )

    failed = orchestrator.ingest_file(source_file, source_type="upload")
    retried = orchestrator.retry_failed_document_version(failed.document_version_id)

    assert retried.status == LifecycleStatus.FAILED
    assert retried.error_message == "chunking exploded"
    job = session.get(IngestionJob, failed.job_id)
    assert job is not None
    assert job.error_message == "chunking exploded"
