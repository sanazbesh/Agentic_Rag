from __future__ import annotations

from collections.abc import Sequence

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.chunking.interfaces import Chunker
from agentic_rag.chunking.models import ChildChunk, ChunkingResult, ParentChunk
from agentic_rag.indexing.dense_child_chunks import DenseEmbeddingConfig, DenseEmbeddingService, QdrantChildChunkStore
from agentic_rag.ingestion.interfaces import DocumentIngestor
from agentic_rag.ingestion_pipeline.chunk_persistence import ChunkPersistenceService
from agentic_rag.ingestion_pipeline.document_registry import DocumentRegistry
from agentic_rag.ingestion_pipeline.orchestrator import IngestionOrchestrator
from agentic_rag.ingestion_pipeline.vector_indexing import ChildChunkVectorIndexingService
from agentic_rag.storage.document_store import LocalDocumentStore
from agentic_rag.storage.models import Base, Chunk, Document
from agentic_rag.storage.models import LifecycleStatus
from agentic_rag.types import Document as IngestedDocument


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(bind=engine) as db_session:
        yield db_session


class EmptyTextIngestor(DocumentIngestor):
    def ingest(self, records):
        return [IngestedDocument(id="empty", text="   ", metadata={"source": "x", "source_name": "x"})]


class NoChunksChunker(Chunker):
    def chunk(self, document: IngestedDocument) -> ChunkingResult:
        return ChunkingResult(parent_chunks=[], child_chunks=[])


class EmptyChildTextChunker(Chunker):
    def chunk(self, document: IngestedDocument) -> ChunkingResult:
        return ChunkingResult(
            parent_chunks=[ParentChunk(parent_chunk_id="p1", document_id="doc", source="s", source_name="n", text="parent")],
            child_chunks=[ChildChunk(child_chunk_id="c1", parent_chunk_id="p1", document_id="doc", source="s", source_name="n", text=" ", child_order=0, token_count=0)],
        )


class GoodChunker(Chunker):
    def chunk(self, document: IngestedDocument) -> ChunkingResult:
        return ChunkingResult(
            parent_chunks=[ParentChunk(parent_chunk_id="p1", document_id="doc", source="s", source_name="n", text="parent")],
            child_chunks=[ChildChunk(child_chunk_id="c1", parent_chunk_id="p1", document_id="doc", source="s", source_name="n", text="child text", child_order=0, token_count=2)],
        )


class RecordingEmbeddingBackend:
    @property
    def dimension(self) -> int:
        return 3

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


class NoopQdrantClient:
    def collection_exists(self, collection_name: str) -> bool:
        return True

    def create_collection(self, collection_name: str, *, vectors_config: dict[str, object]) -> None:
        return None

    def get_collection(self, collection_name: str) -> dict[str, object]:
        return {"size": 3, "distance": "Cosine"}

    def upsert(self, collection_name: str, *, points: Sequence[dict[str, object]]) -> None:
        return None


def _build(
    session: Session,
    tmp_path,
    *,
    chunker: Chunker,
    markdown_ingestor: DocumentIngestor | None = None,
    with_vector: bool = False,
    disable_chunk_persistence: bool = False,
) -> IngestionOrchestrator:
    vector_service = None
    chunk_service = None if disable_chunk_persistence else ChunkPersistenceService(session=session)
    if with_vector:
        embedding = DenseEmbeddingService(config=DenseEmbeddingConfig(model_name="test", batch_size=8), backend=RecordingEmbeddingBackend())
        store = QdrantChildChunkStore(client=NoopQdrantClient(), collection_name="test")
        vector_service = ChildChunkVectorIndexingService(session=session, embedding_service=embedding, store=store)

    return IngestionOrchestrator(
        session=session,
        registry=DocumentRegistry(session),
        document_store=LocalDocumentStore(tmp_path / "docs"),
        markdown_ingestor=markdown_ingestor,
        chunker=chunker,
        chunk_persistence_service=chunk_service,
        vector_indexing_service=vector_service,
    )


def test_empty_parsed_text_fails_validation(session: Session, tmp_path) -> None:
    source = tmp_path / "a.md"
    source.write_text("x", encoding="utf-8")
    result = _build(session, tmp_path, chunker=GoodChunker(), markdown_ingestor=EmptyTextIngestor()).ingest_file(source)
    assert result.status == LifecycleStatus.FAILED
    assert "parsed text is empty" in (result.error_message or "")


def test_no_chunks_fails_validation(session: Session, tmp_path) -> None:
    source = tmp_path / "b.md"
    source.write_text("# H", encoding="utf-8")
    result = _build(session, tmp_path, chunker=NoChunksChunker()).ingest_file(source)
    assert result.status == LifecycleStatus.FAILED
    assert "no parent chunks" in (result.error_message or "")


def test_child_chunk_without_text_fails_validation(session: Session, tmp_path) -> None:
    source = tmp_path / "c.md"
    source.write_text("# H", encoding="utf-8")
    result = _build(session, tmp_path, chunker=EmptyChildTextChunker()).ingest_file(source)
    assert result.status == LifecycleStatus.FAILED
    assert "empty text" in (result.error_message or "")


def test_missing_qdrant_point_id_fails_validation_after_indexing(session: Session, tmp_path) -> None:
    source = tmp_path / "d.md"
    source.write_text("# H\n\nBody", encoding="utf-8")
    result = _build(session, tmp_path, chunker=GoodChunker(), with_vector=True).ingest_file(source)
    assert result.status == LifecycleStatus.FAILED
    assert "missing qdrant_point_id" in (result.error_message or "")


def test_successful_ingestion_passes_validation(session: Session, tmp_path) -> None:
    source = tmp_path / "e.md"
    source.write_text("# H\n\nBody", encoding="utf-8")
    result = _build(session, tmp_path, chunker=GoodChunker(), with_vector=True).ingest_file(source)
    assert result.status == LifecycleStatus.READY
    assert result.parent_chunk_count == 1
    assert result.child_chunk_count == 1
    assert result.indexed_vector_count == 1
    assert result.parent_chunks_persisted == 1
    assert result.child_chunks_persisted == 1


def test_missing_persisted_chunks_fails_validation(session: Session, tmp_path) -> None:
    source = tmp_path / "g.md"
    source.write_text("# H\n\nBody", encoding="utf-8")
    result = _build(session, tmp_path, chunker=GoodChunker(), disable_chunk_persistence=True).ingest_file(source)
    assert result.status == LifecycleStatus.FAILED
    assert "no persisted chunks" in (result.error_message or "")


def test_txt_ingestion_persists_parent_and_child_chunks(session: Session, tmp_path) -> None:
    source = tmp_path / "small.txt"
    source.write_text("Heading\n\nBody text for chunking.", encoding="utf-8")
    result = _build(session, tmp_path, chunker=GoodChunker(), with_vector=True).ingest_file(source)
    assert result.status == LifecycleStatus.READY

    persisted = session.query(Chunk).filter(Chunk.document_version_id == result.document_version_id).all()
    parents = [chunk for chunk in persisted if chunk.chunk_type == "parent"]
    children = [chunk for chunk in persisted if chunk.chunk_type == "child"]
    assert len(parents) == 1
    assert len(children) == 1



def test_validation_uses_provided_persisted_chunks_when_available(session: Session, tmp_path) -> None:
    source = tmp_path / "h.md"
    source.write_text("# H\n\nBody", encoding="utf-8")
    orchestrator = _build(session, tmp_path, chunker=GoodChunker(), with_vector=True)
    result = orchestrator.ingest_file(source)
    assert result.status == LifecycleStatus.READY
    assert result.parent_chunks_persisted == 1
    assert result.child_chunks_persisted == 1

def test_failed_validation_prevents_ready_promotion(session: Session, tmp_path) -> None:
    source = tmp_path / "f.md"
    source.write_text("# V1\n\nBody", encoding="utf-8")
    ok = _build(session, tmp_path, chunker=GoodChunker()).ingest_file(source)
    assert ok.status == LifecycleStatus.READY

    source.write_text("# V2\n\nBroken", encoding="utf-8")
    failed = _build(session, tmp_path, chunker=NoChunksChunker()).ingest_file(source)
    assert failed.status == LifecycleStatus.FAILED

    persisted_document = session.get(Document, ok.document_id)
    assert persisted_document is not None
    assert persisted_document.current_version_id == ok.document_version_id
    assert persisted_document.status == LifecycleStatus.READY
