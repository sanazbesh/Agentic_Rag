from __future__ import annotations

from datetime import datetime

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.ingestion_pipeline.document_registry import DocumentRegistry
from agentic_rag.ingestion_pipeline.ingestion_jobs import IngestionJobService
from agentic_rag.storage.models import Base, LifecycleStatus


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(bind=engine) as db_session:
        yield db_session


def _seed_document(session: Session) -> tuple[str, str]:
    registry = DocumentRegistry(session)
    registration = registry.register_document(
        source_name="seed.md",
        source_type="upload",
        content_bytes=b"# Seed",
        status=LifecycleStatus.PENDING,
    )
    return registration.document.id, registration.version.id


def test_job_creation_defaults_to_pending(session: Session) -> None:
    document_id, version_id = _seed_document(session)
    service = IngestionJobService(session)

    job = service.create_job(document_id=document_id, document_version_id=version_id)

    assert job.id.startswith("job_")
    assert job.status == LifecycleStatus.PENDING
    assert job.started_at is None
    assert job.finished_at is None


def test_status_transitions_set_expected_timestamps(session: Session) -> None:
    document_id, version_id = _seed_document(session)
    service = IngestionJobService(session)
    job = service.create_job(document_id=document_id, document_version_id=version_id)

    service.mark_processing(job)
    processing_started_at = job.started_at

    assert job.status == LifecycleStatus.PROCESSING
    assert isinstance(processing_started_at, datetime)
    assert job.finished_at is None

    service.mark_ready(job)

    assert job.status == LifecycleStatus.READY
    assert job.started_at == processing_started_at
    assert isinstance(job.finished_at, datetime)
    assert job.finished_at >= job.started_at


def test_failed_status_persists_error_message_and_finished_at(session: Session) -> None:
    document_id, version_id = _seed_document(session)
    service = IngestionJobService(session)
    job = service.create_job(document_id=document_id, document_version_id=version_id)

    service.mark_failed(job, error_message="boom")

    assert job.status == LifecycleStatus.FAILED
    assert job.error_message == "boom"
    assert isinstance(job.started_at, datetime)
    assert isinstance(job.finished_at, datetime)
    assert job.finished_at >= job.started_at


def test_list_recent_jobs_filters_by_document_version_and_status(session: Session) -> None:
    first_document_id, first_version_id = _seed_document(session)
    second_document_id, second_version_id = _seed_document(session)
    service = IngestionJobService(session)

    job_a = service.create_job(document_id=first_document_id, document_version_id=first_version_id)
    service.mark_ready(job_a)

    job_b = service.create_job(document_id=first_document_id, document_version_id=first_version_id)
    service.mark_failed(job_b, error_message="bad")

    job_c = service.create_job(document_id=second_document_id, document_version_id=second_version_id)
    service.mark_processing(job_c)

    by_document = service.list_recent_jobs(document_id=first_document_id)
    by_version = service.list_recent_jobs(document_version_id=first_version_id)
    by_status = service.list_recent_jobs(status=LifecycleStatus.FAILED)

    assert {job.id for job in by_document} == {job_a.id, job_b.id}
    assert {job.id for job in by_version} == {job_a.id, job_b.id}
    assert [job.id for job in by_status] == [job_b.id]


def test_getters_return_jobs_by_id_and_latest_for_document_version(session: Session) -> None:
    document_id, version_id = _seed_document(session)
    service = IngestionJobService(session)

    first = service.create_job(document_id=document_id, document_version_id=version_id)
    service.mark_failed(first, error_message="first")
    second = service.create_job(document_id=document_id, document_version_id=version_id)

    by_id = service.get_job(second.id)
    latest = service.get_latest_job_for_document_version(version_id)

    assert by_id is not None
    assert by_id.id == second.id
    assert latest is not None
    assert latest.id == second.id
