from __future__ import annotations

from collections.abc import Sequence

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.indexing.dense_child_chunks import DenseEmbeddingConfig, DenseEmbeddingService, QdrantChildChunkStore, stable_qdrant_point_id
from agentic_rag.ingestion_pipeline.chunk_persistence import ChunkPersistenceService
from agentic_rag.ingestion_pipeline.document_registry import DocumentRegistry
from agentic_rag.ingestion_pipeline.orchestrator import IngestionOrchestrator
from agentic_rag.ingestion_pipeline.vector_indexing import ChildChunkVectorIndexingService
from agentic_rag.storage.document_store import LocalDocumentStore
from agentic_rag.storage.models import Base, Chunk, DocumentVersion, IngestionJob, LifecycleStatus


class RecordingEmbeddingBackend:
    def __init__(self, *, dimension: int = 3) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        return [[float(len(text)), 1.0, 2.0] for text in texts]


class RecordingQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, *, vectors_config: dict[str, object]) -> None:
        self.collections[collection_name] = {"config": dict(vectors_config), "points": {}}

    def get_collection(self, collection_name: str) -> dict[str, object]:
        config = self.collections[collection_name]["config"]
        assert isinstance(config, dict)
        return {"size": config["size"], "distance": config["distance"]}

    def upsert(self, collection_name: str, *, points: Sequence[dict[str, object]]) -> None:
        bucket = self.collections[collection_name]["points"]
        assert isinstance(bucket, dict)
        for point in points:
            bucket[str(point["id"])] = dict(point)

    def retrieve(self, collection_name: str, *, ids: Sequence[str]) -> list[dict[str, object]]:
        bucket = self.collections[collection_name]["points"]
        assert isinstance(bucket, dict)
        return [dict(bucket[item]) for item in ids if item in bucket]


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(bind=engine) as db_session:
        yield db_session


def _build_orchestrator(session: Session, tmp_path, *, failing_qdrant: bool = False) -> IngestionOrchestrator:
    store = LocalDocumentStore(tmp_path / "documents")
    registry = DocumentRegistry(session)
    chunk_service = ChunkPersistenceService(session=session)

    backend = RecordingEmbeddingBackend()
    embedding = DenseEmbeddingService(config=DenseEmbeddingConfig(model_name="test-model", batch_size=8), backend=backend)

    qdrant_client = RecordingQdrantClient()
    if failing_qdrant:
        def _explode(collection_name: str, *, points: Sequence[dict[str, object]]) -> None:
            raise RuntimeError("qdrant exploded")

        qdrant_client.upsert = _explode  # type: ignore[method-assign]

    vector_store = QdrantChildChunkStore(client=qdrant_client, collection_name="legal_chunks")
    vector_service = ChildChunkVectorIndexingService(session=session, embedding_service=embedding, store=vector_store)

    return IngestionOrchestrator(
        session=session,
        registry=registry,
        document_store=store,
        chunk_persistence_service=chunk_service,
        vector_indexing_service=vector_service,
    )


def test_reindex_loads_persisted_storage_path(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "policy.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# Header\n\nBody text.", encoding="utf-8")

    orchestrator = _build_orchestrator(session, tmp_path)
    first = orchestrator.ingest_file(source_file, source_type="upload")

    source_file.write_text("# Header\n\nCHANGED SOURCE SHOULD NOT BE USED.", encoding="utf-8")
    result = orchestrator.reindex_document_version(first.document_version_id)

    assert result.status == LifecycleStatus.READY
    chunks = session.query(Chunk).filter(Chunk.document_version_id == first.document_version_id).all()
    assert all("CHANGED SOURCE SHOULD NOT BE USED." not in chunk.text for chunk in chunks)


def test_reindex_refreshes_chunks_without_duplicates_and_point_ids_are_deterministic(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "msa.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# A\n\nOne.\n\n## B\n\nTwo.", encoding="utf-8")

    orchestrator = _build_orchestrator(session, tmp_path)
    first = orchestrator.ingest_file(source_file, source_type="upload")

    before = sorted(session.query(Chunk).filter(Chunk.document_version_id == first.document_version_id).all(), key=lambda x: x.id)
    before_child_ids = [c.id for c in before if c.chunk_type == "child"]

    second = orchestrator.reindex_document(first.document_id)
    after = sorted(session.query(Chunk).filter(Chunk.document_version_id == first.document_version_id).all(), key=lambda x: x.id)

    assert second.status == LifecycleStatus.READY
    assert len(after) == len(before)
    assert [item.id for item in after] == [item.id for item in before]

    for child in [c for c in after if c.chunk_type == "child"]:
        assert child.id in before_child_ids
        assert child.qdrant_point_id == stable_qdrant_point_id(child.id)


def test_failed_reindex_records_failed_job_status(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "failure.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# F\n\nBody", encoding="utf-8")

    ok_orchestrator = _build_orchestrator(session, tmp_path)
    first = ok_orchestrator.ingest_file(source_file, source_type="upload")

    bad_orchestrator = _build_orchestrator(session, tmp_path, failing_qdrant=True)
    result = bad_orchestrator.reindex_document_version(first.document_version_id)

    assert result.status == LifecycleStatus.FAILED
    job = session.get(IngestionJob, result.job_id)
    assert job is not None
    assert job.status == LifecycleStatus.FAILED
    assert job.error_message is not None


def test_successful_reindex_sets_ready_status_and_metadata(session: Session, tmp_path) -> None:
    source_file = tmp_path / "uploads" / "ready.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("# R\n\nBody", encoding="utf-8")

    orchestrator = _build_orchestrator(session, tmp_path)
    first = orchestrator.ingest_file(source_file, source_type="upload")

    result = orchestrator.reindex_document_version(first.document_version_id)

    assert result.status == LifecycleStatus.READY
    version = session.get(DocumentVersion, first.document_version_id)
    assert version is not None
    assert version.status == LifecycleStatus.READY
    assert version.parser_version == "MarkdownDocumentIngestor"
    assert version.chunker_version == "MarkdownParentChildChunker"
    assert version.embedding_model == "test-model"
