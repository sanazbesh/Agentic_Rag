from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.ingestion_pipeline.document_deletion import DocumentDeletionError, DocumentDeletionService
from agentic_rag.storage.document_store import LocalDocumentStore
from agentic_rag.storage.models import Base, Chunk, Document, DocumentVersion, IngestionJob, LifecycleStatus


class FakeQdrantClient:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.fail_delete = fail_delete
        self.deleted_calls: list[tuple[str, dict[str, list[str]]]] = []

    def delete(self, collection_name: str, *, points_selector: dict[str, list[str]]) -> None:
        if self.fail_delete:
            raise RuntimeError("qdrant unavailable")
        self.deleted_calls.append((collection_name, points_selector))


class FakeQdrantStore:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.client = FakeQdrantClient(fail_delete=fail_delete)
        self.collection_name = "legal_child_chunks_dense"

    def delete_points(self, point_ids: tuple[str, ...]) -> None:
        self.client.delete(self.collection_name, points_selector={"points": list(point_ids)})


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(bind=engine) as db_session:
        yield db_session


def _seed_graph(session: Session, store: LocalDocumentStore) -> str:
    doc = Document(id="doc_1", source_name="policy.md", source_type="upload", status=LifecycleStatus.READY)
    version = DocumentVersion(
        id="ver_1",
        document_id="doc_1",
        content_hash="abc",
        storage_path=store.save_bytes(content=b"v1", document_id="doc_1", document_version_id="ver_1", source_name="policy.md"),
        status=LifecycleStatus.READY,
    )
    job = IngestionJob(id="job_1", document_id="doc_1", document_version_id="ver_1", status=LifecycleStatus.READY)
    parent = Chunk(
        id="chunk_parent",
        document_id="doc_1",
        document_version_id="ver_1",
        parent_chunk_id=None,
        chunk_type="parent",
        text="parent",
        qdrant_point_id=None,
    )
    child = Chunk(
        id="chunk_child",
        document_id="doc_1",
        document_version_id="ver_1",
        parent_chunk_id="chunk_parent",
        chunk_type="child",
        text="child",
        qdrant_point_id="point-123",
    )
    session.add_all([doc, version, job, parent, child])
    session.flush()
    return doc.id


def test_delete_document_cleans_related_records_vectors_and_files(session: Session, tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")
    document_id = _seed_graph(session, store)
    qdrant_store = FakeQdrantStore()

    service = DocumentDeletionService(session=session, document_store=store, qdrant_store=qdrant_store)
    result = service.delete_document(document_id=document_id)

    assert result.deleted is True
    assert session.get(Document, document_id) is None
    assert session.query(DocumentVersion).count() == 0
    assert session.query(Chunk).count() == 0
    assert session.query(IngestionJob).count() == 0
    assert qdrant_store.client.deleted_calls == [
        ("legal_child_chunks_dense", {"points": ["point-123"]}),
    ]
    assert (store.root_path / "doc_1").exists() is False


def test_delete_document_missing_is_safe_noop(session: Session, tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")
    service = DocumentDeletionService(session=session, document_store=store, qdrant_store=FakeQdrantStore())

    result = service.delete_document(document_id="doc_missing")

    assert result.deleted is False


def test_delete_document_reports_partial_cleanup_failure(session: Session, tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")
    document_id = _seed_graph(session, store)
    service = DocumentDeletionService(session=session, document_store=store, qdrant_store=FakeQdrantStore(fail_delete=True))

    with pytest.raises(DocumentDeletionError, match="qdrant_cleanup_failed"):
        service.delete_document(document_id=document_id)

    assert session.get(Document, document_id) is not None
