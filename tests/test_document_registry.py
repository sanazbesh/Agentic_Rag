from __future__ import annotations

import hashlib

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.ingestion_pipeline.document_registry import (
    DocumentRegistry,
    compute_sha256_from_bytes,
    compute_sha256_from_file_path,
)
from agentic_rag.storage.models import Base, DocumentVersion, LifecycleStatus


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(bind=engine) as db_session:
        yield db_session


def test_compute_sha256_from_bytes_is_stable() -> None:
    payload = b"legal-rag-doc-content"

    first = compute_sha256_from_bytes(payload)
    second = compute_sha256_from_bytes(payload)

    assert first == second
    assert first == hashlib.sha256(payload).hexdigest()


def test_compute_sha256_from_file_path_is_stable(tmp_path) -> None:
    payload = b"registered bytes from path"
    file_path = tmp_path / "agreement.md"
    file_path.write_bytes(payload)

    first = compute_sha256_from_file_path(file_path)
    second = compute_sha256_from_file_path(str(file_path))

    assert first == second
    assert first == hashlib.sha256(payload).hexdigest()


def test_first_registration_creates_document_and_version(session: Session) -> None:
    registry = DocumentRegistry(session)

    result = registry.register_document(
        source_name="nda.pdf",
        source_type="pdf",
        content_bytes=b"first-version",
    )

    assert result.created_document is True
    assert result.created_version is True
    assert result.document.id
    assert result.version.id
    assert result.version.document_id == result.document.id
    assert result.document.current_version_id == result.version.id


def test_duplicate_content_does_not_create_duplicate_version(session: Session) -> None:
    registry = DocumentRegistry(session)

    first = registry.register_document(
        source_name="memo.pdf",
        source_type="pdf",
        content_bytes=b"same-content",
    )
    second = registry.register_document(
        source_name="memo.pdf",
        source_type="pdf",
        content_bytes=b"same-content",
    )

    versions = session.query(DocumentVersion).filter(DocumentVersion.document_id == first.document.id).all()

    assert second.created_document is False
    assert second.created_version is False
    assert second.version.id == first.version.id
    assert len(versions) == 1


def test_changed_content_creates_new_version_and_updates_current_pointer(session: Session) -> None:
    registry = DocumentRegistry(session)

    first = registry.register_document(
        source_name="msa.pdf",
        source_type="pdf",
        content_bytes=b"v1",
    )
    second = registry.register_document(
        source_name="msa.pdf",
        source_type="pdf",
        content_bytes=b"v2",
    )

    assert second.created_document is False
    assert second.created_version is True
    assert second.version.id != first.version.id
    assert second.document.current_version_id == second.version.id


def test_status_update_works_for_document_and_version(session: Session) -> None:
    registry = DocumentRegistry(session)
    registration = registry.register_document(
        source_name="status-test.pdf",
        source_type="pdf",
        content_bytes=b"status-content",
    )

    updated_document = registry.update_document_status(
        registration.document.id,
        LifecycleStatus.PROCESSING,
    )
    updated_version = registry.update_version_status(
        registration.version.id,
        LifecycleStatus.READY,
    )

    assert updated_document.status == LifecycleStatus.PROCESSING
    assert updated_version.status == LifecycleStatus.READY


def test_lookup_by_content_hash_works(session: Session) -> None:
    registry = DocumentRegistry(session)
    content_bytes = b"lookup-content"

    registration = registry.register_document(
        source_name="lookup.pdf",
        source_type="pdf",
        content_bytes=content_bytes,
    )
    content_hash = compute_sha256_from_bytes(content_bytes)

    by_hash = registry.find_version_by_content_hash(content_hash)
    by_document_hash = registry.find_version_by_content_hash(
        content_hash,
        document_id=registration.document.id,
    )

    assert by_hash is not None
    assert by_document_hash is not None
    assert by_hash.id == registration.version.id
    assert by_document_hash.id == registration.version.id


def test_getters_load_document_and_version_by_id(session: Session) -> None:
    registry = DocumentRegistry(session)
    registration = registry.register_document(
        source_name="getter.pdf",
        source_type="pdf",
        content_bytes=b"getter-content",
    )

    loaded_document = registry.get_document_by_id(registration.document.id)
    loaded_version = registry.get_version_by_id(registration.version.id)

    assert loaded_document is not None
    assert loaded_version is not None
    assert loaded_document.id == registration.document.id
    assert loaded_version.id == registration.version.id
