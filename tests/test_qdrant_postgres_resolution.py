from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.retrieval.parent_child import ChildChunkSearcher, ParentChunkStore
from agentic_rag.retrieval.postgres_chunk_repository import PostgresChunkRepository
from agentic_rag.retrieval.qdrant_postgres_resolver import (
    PostgresResolvedQdrantChildRepository,
    QdrantResultResolver,
)
from agentic_rag.storage.models import Base, Chunk, Document, DocumentVersion, LifecycleStatus


class FakeQdrantBackend:
    def __init__(self, hits: list[dict[str, object]]) -> None:
        self._hits = hits

    def search(self, query: str, *, filters=None, limit: int = 10):  # type: ignore[override]
        del query, filters
        return self._hits[: max(0, limit)]


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(bind=engine) as db_session:
        yield db_session


def _seed(session: Session) -> None:
    doc = Document(
        id="doc-1",
        source_name="contract.md",
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
                text="Parent context text.",
                heading="Section A",
                section_path="Section A",
                metadata_json={"source": "s3://bucket/contract.md", "source_name": "contract.md"},
            ),
            Chunk(
                id="child-1",
                document_id="doc-1",
                document_version_id="ver-1",
                parent_chunk_id="parent-1",
                chunk_type="child",
                text="Child context text.",
                heading="Section A",
                section_path="Section A",
                qdrant_point_id="point-1",
                metadata_json={"jurisdiction": "NY", "token_count": 12},
            ),
        ]
    )
    session.flush()


def _build_searcher(session: Session, hits: list[dict[str, object]]) -> ChildChunkSearcher:
    chunk_repo = PostgresChunkRepository(session=session)
    resolver = QdrantResultResolver(chunk_repository=chunk_repo)
    child_repo = PostgresResolvedQdrantChildRepository(
        qdrant_backend=FakeQdrantBackend(hits),
        resolver=resolver,
    )
    return ChildChunkSearcher(repository=child_repo, default_limit=10)


def test_qdrant_result_resolves_to_postgres_chunk(session: Session) -> None:
    _seed(session)
    searcher = _build_searcher(
        session,
        [
            {
                "id": "point-1",
                "score": 0.88,
                "payload": {"chunk_id": "child-1", "rank_reason": "semantic"},
            }
        ],
    )

    results = searcher.search_child_chunks("breach")

    assert len(results) == 1
    hit = results[0]
    assert hit.child_chunk_id == "child-1"
    assert hit.parent_chunk_id == "parent-1"
    assert hit.document_id == "doc-1"
    assert hit.text == "Child context text."


def test_missing_chunk_resolution_is_skipped_safely(session: Session) -> None:
    _seed(session)
    searcher = _build_searcher(
        session,
        [{"id": "point-missing", "score": 0.42, "payload": {"chunk_id": "child-missing"}}],
    )

    results = searcher.search_child_chunks("breach")

    assert results == []


def test_parent_expansion_with_persisted_chunks(session: Session) -> None:
    _seed(session)
    searcher = _build_searcher(
        session,
        [{"id": "point-1", "score": 0.77, "payload": {"chunk_id": "child-1"}}],
    )
    child_hits = searcher.search_child_chunks("breach")

    parent_store = ParentChunkStore(repository=PostgresChunkRepository(session=session))
    parents = parent_store.retrieve_parent_chunks([hit.parent_chunk_id for hit in child_hits])

    assert [parent.parent_chunk_id for parent in parents] == ["parent-1"]
    assert parents[0].text == "Parent context text."


def test_scores_and_metadata_are_preserved_from_qdrant_hit(session: Session) -> None:
    _seed(session)
    searcher = _build_searcher(
        session,
        [
            {
                "id": "point-1",
                "score": 0.915,
                "payload": {"chunk_id": "child-1", "source_rank": 3, "document_type": "agreement"},
            }
        ],
    )

    results = searcher.search_child_chunks("breach")

    assert len(results) == 1
    hit = results[0]
    assert hit.score == pytest.approx(0.915)
    assert hit.payload["source_rank"] == 3
    assert hit.payload["document_type"] == "agreement"
    assert hit.payload["jurisdiction"] == "NY"


def test_output_shape_matches_existing_child_search_result_contract(session: Session) -> None:
    _seed(session)
    searcher = _build_searcher(
        session,
        [{"id": "point-1", "score": 0.51, "payload": {"qdrant_point_id": "point-1"}}],
    )

    results = searcher.search_child_chunks("breach", filters={"selected_document_ids": ["doc-1"]})

    assert len(results) == 1
    hit = results[0]
    assert isinstance(hit.child_chunk_id, str)
    assert isinstance(hit.parent_chunk_id, str)
    assert isinstance(hit.document_id, str)
    assert isinstance(hit.text, str)
    assert isinstance(hit.score, float)
    assert isinstance(hit.payload, dict)


def test_document_version_scope_filter_is_supported(session: Session) -> None:
    _seed(session)
    searcher = _build_searcher(
        session,
        [{"id": "point-1", "score": 0.51, "payload": {"chunk_id": "child-1"}}],
    )

    scoped = searcher.search_child_chunks(
        "breach",
        filters={"selected_document_version_ids": ["ver-1"]},
    )
    excluded = searcher.search_child_chunks(
        "breach",
        filters={"selected_document_version_ids": ["ver-2"]},
    )

    assert len(scoped) == 1
    assert excluded == []
