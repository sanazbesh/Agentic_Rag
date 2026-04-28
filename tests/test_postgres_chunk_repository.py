from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.retrieval.postgres_chunk_repository import PostgresChunkRepository
from agentic_rag.storage.models import Base, Chunk, Document, DocumentVersion, LifecycleStatus


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(bind=engine) as db_session:
        yield db_session


def _seed_chunks(session: Session) -> None:
    doc = Document(
        id="doc-1",
        source_name="employment.md",
        source_type="markdown",
        status=LifecycleStatus.READY,
    )
    version = DocumentVersion(
        id="ver-1",
        document_id="doc-1",
        content_hash="hash-1",
        status=LifecycleStatus.READY,
    )
    session.add_all([doc, version])
    session.flush()

    session.add_all(
        [
            Chunk(
                id="parent-1",
                document_id="doc-1",
                document_version_id="ver-1",
                parent_chunk_id=None,
                chunk_type="parent",
                text="Parent clause text.",
                heading="Section 1",
                section_path="Section 1",
                metadata_json={
                    "source": "/tmp/employment.md",
                    "source_name": "employment.md",
                    "heading_path": ["Section 1"],
                    "parent_order": 1,
                    "part_number": 1,
                    "total_parts": 1,
                    "custom_parent_key": "parent-value",
                },
            ),
            Chunk(
                id="child-1",
                document_id="doc-1",
                document_version_id="ver-1",
                parent_chunk_id="parent-1",
                chunk_type="child",
                text="Child evidence one.",
                heading="Section 1",
                section_path="Section 1",
                qdrant_point_id="point-1",
                metadata_json={"child_order": 1, "token_count": 10},
            ),
            Chunk(
                id="child-2",
                document_id="doc-1",
                document_version_id="ver-1",
                parent_chunk_id="parent-1",
                chunk_type="child",
                text="Child evidence two.",
                heading="Section 1",
                section_path="Section 1",
                qdrant_point_id="point-2",
                metadata_json={"child_order": 2, "token_count": 11},
            ),
        ]
    )
    session.flush()


def test_fetch_chunk_by_id(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    chunk = repository.get_chunk_by_id("child-1")

    assert chunk is not None
    assert chunk.chunk_id == "child-1"
    assert chunk.parent_chunk_id == "parent-1"
    assert chunk.qdrant_point_id == "point-1"


def test_fetch_multiple_chunks_by_ids(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    chunks = repository.get_chunks_by_ids(["child-2", "missing", "child-1", "child-2"])

    assert [chunk.chunk_id for chunk in chunks] == ["child-2", "child-1"]


def test_fetch_parent_chunk(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    parent = repository.get_parent_chunk("parent-1")

    assert parent is not None
    assert parent.parent_chunk_id == "parent-1"
    assert parent.document_id == "doc-1"
    assert parent.heading_path == ("Section 1",)
    assert parent.metadata["custom_parent_key"] == "parent-value"


def test_fetch_children_for_parent(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    children = repository.get_children_for_parent("parent-1")

    assert [chunk.chunk_id for chunk in children] == ["child-1", "child-2"]
    assert all(chunk.parent_chunk_id == "parent-1" for chunk in children)


def test_fetch_chunks_by_document_version_id(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    chunks = repository.get_chunks_for_document_version("ver-1")

    assert [chunk.chunk_id for chunk in chunks] == ["child-1", "child-2", "parent-1"]


def test_mapping_from_orm_chunk_to_parent_domain_object(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    parent = repository.get_by_ids(["parent-1"])[0]

    assert parent.parent_chunk_id == "parent-1"
    assert parent.heading_text == "Section 1"
    assert parent.source_name == "employment.md"


def test_missing_chunk_returns_none_or_empty(session: Session) -> None:
    _seed_chunks(session)
    repository = PostgresChunkRepository(session=session)

    assert repository.get_chunk_by_id("missing") is None
    assert repository.get_parent_chunk("missing") is None
    assert repository.resolve_qdrant_point_id("missing") is None
    assert repository.get_chunks_by_ids(["missing"]) == []
    assert repository.get_children_for_parent("missing") == []
