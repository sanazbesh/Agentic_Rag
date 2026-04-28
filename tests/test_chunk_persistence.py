from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.chunking.models import ChildChunk, ChunkingResult, ParentChunk
from agentic_rag.ingestion_pipeline.chunk_persistence import (
    CHILD_CHUNK_TYPE,
    PARENT_CHUNK_TYPE,
    ChunkPersistenceService,
)
from agentic_rag.storage.models import Base, Chunk, Document, DocumentVersion, LifecycleStatus


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(bind=engine) as db_session:
        yield db_session


def _create_document_and_version(session: Session) -> tuple[Document, DocumentVersion]:
    document = Document(
        id="doc_test_1",
        source_name="agreement.md",
        source_type="markdown",
        status=LifecycleStatus.PENDING,
    )
    version = DocumentVersion(
        id="ver_test_1",
        document_id=document.id,
        content_hash="hash_test_1",
        status=LifecycleStatus.PENDING,
    )
    document.current_version_id = version.id
    session.add(document)
    session.add(version)
    session.flush()
    return document, version


def _sample_chunking_result(document_id: str) -> ChunkingResult:
    parent = ParentChunk(
        parent_chunk_id="parent_1",
        document_id=document_id,
        source="/tmp/agreement.md",
        source_name="agreement.md",
        text="# Section 1\nParent legal text.",
        heading_path=("Section 1",),
        heading_text="Section 1",
        parent_order=1,
        parent_token_count=20,
        original_heading_context=("Section 1",),
        part_number=1,
        total_parts=1,
    )
    child = ChildChunk(
        child_chunk_id="child_1",
        parent_chunk_id=parent.parent_chunk_id,
        document_id=document_id,
        source="/tmp/agreement.md",
        source_name="agreement.md",
        text="Parent legal text.",
        child_order=1,
        token_count=8,
        heading_path=("Section 1",),
    )
    return ChunkingResult(parent_chunks=[parent], child_chunks=[child])


def test_parent_and_child_chunks_are_saved_and_linked(session: Session) -> None:
    document, version = _create_document_and_version(session)
    service = ChunkPersistenceService(session=session)

    persisted = service.persist_chunks(
        document_id=document.id,
        document_version_id=version.id,
        chunking_result=_sample_chunking_result(document.id),
    )

    assert len(persisted) == 2

    parent = session.get(Chunk, "parent_1")
    child = session.get(Chunk, "child_1")

    assert parent is not None
    assert child is not None
    assert parent.chunk_type == PARENT_CHUNK_TYPE
    assert child.chunk_type == CHILD_CHUNK_TYPE
    assert child.parent_chunk_id == parent.id


def test_chunks_are_linked_to_document_and_version(session: Session) -> None:
    document, version = _create_document_and_version(session)
    service = ChunkPersistenceService(session=session)
    service.persist_chunks(
        document_id=document.id,
        document_version_id=version.id,
        chunking_result=_sample_chunking_result(document.id),
    )

    parent = service.get_chunk_by_id("parent_1")
    child = service.get_chunk_by_id("child_1")

    assert parent is not None
    assert child is not None
    assert parent.document_id == document.id
    assert child.document_id == document.id
    assert parent.document_version_id == version.id
    assert child.document_version_id == version.id


def test_lookup_by_chunk_id_and_document_version_id(session: Session) -> None:
    document, version = _create_document_and_version(session)
    service = ChunkPersistenceService(session=session)
    service.persist_chunks(
        document_id=document.id,
        document_version_id=version.id,
        chunking_result=_sample_chunking_result(document.id),
    )

    by_id = service.get_chunk_by_id("child_1")
    by_version = service.get_chunks_by_document_version_id(version.id)

    assert by_id is not None
    assert by_id.id == "child_1"
    assert {item.id for item in by_version} == {"parent_1", "child_1"}


def test_lookup_children_for_parent_chunk(session: Session) -> None:
    document, version = _create_document_and_version(session)
    service = ChunkPersistenceService(session=session)
    service.persist_chunks(
        document_id=document.id,
        document_version_id=version.id,
        chunking_result=_sample_chunking_result(document.id),
    )

    children = service.get_children_for_parent_chunk("parent_1")

    assert [item.id for item in children] == ["child_1"]
    assert children[0].parent_chunk_id == "parent_1"


def test_repeated_persistence_is_idempotent_for_document_version(session: Session) -> None:
    document, version = _create_document_and_version(session)
    service = ChunkPersistenceService(session=session)
    chunking_result = _sample_chunking_result(document.id)

    service.persist_chunks(
        document_id=document.id,
        document_version_id=version.id,
        chunking_result=chunking_result,
    )
    service.persist_chunks(
        document_id=document.id,
        document_version_id=version.id,
        chunking_result=chunking_result,
    )

    all_chunks = service.get_chunks_by_document_version_id(version.id)

    assert len(all_chunks) == 2
    assert sorted(item.id for item in all_chunks) == ["child_1", "parent_1"]
