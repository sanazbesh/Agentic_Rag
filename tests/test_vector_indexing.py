from __future__ import annotations

from collections.abc import Sequence

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
Session = pytest.importorskip("sqlalchemy.orm").Session

from agentic_rag.indexing.dense_child_chunks import DenseEmbeddingConfig, DenseEmbeddingService, QdrantChildChunkStore, stable_qdrant_point_id
from agentic_rag.ingestion_pipeline.chunk_persistence import CHILD_CHUNK_TYPE, PARENT_CHUNK_TYPE
from agentic_rag.ingestion_pipeline.vector_indexing import ChildChunkVectorIndexingService
from agentic_rag.storage.models import Base, Chunk, Document, DocumentVersion, LifecycleStatus


class RecordingEmbeddingBackend:
    def __init__(self, *, dimension: int = 3) -> None:
        self._dimension = dimension
        self.text_calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        self.text_calls.append(list(texts))
        vectors: list[list[float]] = []
        for text in texts:
            base = float(len(text))
            vectors.append([base, base + 1.0, base + 2.0])
        return vectors


class RecordingQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.upsert_calls: list[list[dict[str, object]]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, *, vectors_config: dict[str, object]) -> None:
        self.collections[collection_name] = {"config": dict(vectors_config), "points": {}}

    def get_collection(self, collection_name: str) -> dict[str, object]:
        config = self.collections[collection_name]["config"]
        assert isinstance(config, dict)
        return {"size": config["size"], "distance": config["distance"]}

    def upsert(self, collection_name: str, *, points: Sequence[dict[str, object]]) -> None:
        point_list = [dict(point) for point in points]
        self.upsert_calls.append(point_list)
        bucket = self.collections[collection_name]["points"]
        assert isinstance(bucket, dict)
        for point in point_list:
            bucket[str(point["id"])] = point

    def retrieve(self, collection_name: str, *, ids: Sequence[str]) -> list[dict[str, object]]:
        bucket = self.collections[collection_name]["points"]
        assert isinstance(bucket, dict)
        found: list[dict[str, object]] = []
        for pid in ids:
            point = bucket.get(pid)
            if point is not None:
                found.append(dict(point))
        return found


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(bind=engine) as db_session:
        yield db_session


def _setup_rows(session: Session) -> tuple[Document, DocumentVersion]:
    document = Document(
        id="doc_1",
        source_name="master-service-agreement.md",
        source_type="markdown",
        status=LifecycleStatus.PENDING,
    )
    version = DocumentVersion(
        id="ver_1",
        document_id=document.id,
        content_hash="hash_1",
        status=LifecycleStatus.PENDING,
    )
    document.current_version_id = version.id
    session.add_all([document, version])
    session.flush()

    session.add(
        Chunk(
            id="parent_1",
            document_id=document.id,
            document_version_id=version.id,
            parent_chunk_id=None,
            chunk_type=PARENT_CHUNK_TYPE,
            text="# Parent",
            heading="Parent",
            section_path="Parent",
            metadata_json={"source_name": "master-service-agreement.md"},
        )
    )
    session.add_all(
        [
            Chunk(
                id="child_1",
                document_id=document.id,
                document_version_id=version.id,
                parent_chunk_id="parent_1",
                chunk_type=CHILD_CHUNK_TYPE,
                text="First child text",
                heading="Definitions",
                section_path="Main > Definitions",
                metadata_json={"source_name": "master-service-agreement.md"},
            ),
            Chunk(
                id="child_2",
                document_id=document.id,
                document_version_id=version.id,
                parent_chunk_id="parent_1",
                chunk_type=CHILD_CHUNK_TYPE,
                text="Second child text",
                heading="Term",
                section_path="Main > Term",
                metadata_json={"source_name": "master-service-agreement.md"},
            ),
        ]
    )
    session.flush()
    return document, version


def _build_service(session: Session) -> tuple[ChildChunkVectorIndexingService, RecordingEmbeddingBackend, RecordingQdrantClient]:
    backend = RecordingEmbeddingBackend()
    embedding_service = DenseEmbeddingService(config=DenseEmbeddingConfig(batch_size=8), backend=backend)
    qdrant_client = RecordingQdrantClient()
    store = QdrantChildChunkStore(client=qdrant_client, collection_name="legal_chunks")
    return ChildChunkVectorIndexingService(session=session, embedding_service=embedding_service, store=store), backend, qdrant_client


def test_selects_only_child_chunks_for_document_version(session: Session) -> None:
    _, version = _setup_rows(session)
    service, _, _ = _build_service(session)

    child_chunks = service.get_child_chunks_for_document_version(document_version_id=version.id)

    assert [chunk.id for chunk in child_chunks] == ["child_1", "child_2"]


def test_generates_deterministic_point_ids(session: Session) -> None:
    _, version = _setup_rows(session)
    service, _, _ = _build_service(session)

    result = service.index_document_version(document_version_id=version.id)

    assert result.upserted_child_chunks == 2
    assert session.get(Chunk, "child_1").qdrant_point_id == stable_qdrant_point_id("child_1")
    assert session.get(Chunk, "child_2").qdrant_point_id == stable_qdrant_point_id("child_2")


def test_embedding_backend_called_with_child_text(session: Session) -> None:
    _, version = _setup_rows(session)
    service, backend, _ = _build_service(session)

    service.index_document_version(document_version_id=version.id)

    assert backend.text_calls == [["First child text", "Second child text"]]


def test_qdrant_upsert_receives_expected_vectors_and_payload(session: Session) -> None:
    _, version = _setup_rows(session)
    service, _, client = _build_service(session)

    service.index_document_version(document_version_id=version.id)

    assert len(client.upsert_calls) == 1
    points = client.upsert_calls[0]
    assert [point["id"] for point in points] == [stable_qdrant_point_id("child_1"), stable_qdrant_point_id("child_2")]
    assert [point["vector"] for point in points] == [[16.0, 17.0, 18.0], [17.0, 18.0, 19.0]]

    first_payload = points[0]["payload"]
    assert first_payload == {
        "document_id": "doc_1",
        "document_version_id": "ver_1",
        "chunk_id": "child_1",
        "parent_chunk_id": "parent_1",
        "source_name": "master-service-agreement.md",
        "heading": "Definitions",
        "section_path": ["Main", "Definitions"],
    }


def test_chunk_qdrant_point_id_updated_in_postgres(session: Session) -> None:
    _, version = _setup_rows(session)
    service, _, _ = _build_service(session)

    service.index_document_version(document_version_id=version.id)

    child_1 = session.get(Chunk, "child_1")
    child_2 = session.get(Chunk, "child_2")
    assert child_1.qdrant_point_id == stable_qdrant_point_id("child_1")
    assert child_2.qdrant_point_id == stable_qdrant_point_id("child_2")


def test_repeated_indexing_is_idempotent_and_reuses_existing_ids(session: Session) -> None:
    _, version = _setup_rows(session)
    service, _, client = _build_service(session)

    first = service.index_document_version(document_version_id=version.id)
    first_child_id = session.get(Chunk, "child_1").qdrant_point_id
    second = service.index_document_version(document_version_id=version.id)

    assert first.upserted_child_chunks == 2
    assert second.upserted_child_chunks == 2
    assert session.get(Chunk, "child_1").qdrant_point_id == first_child_id
    assert session.get(Chunk, "child_1").qdrant_point_id == stable_qdrant_point_id("child_1")
    assert len(client.collections["legal_chunks"]["points"]) == 2
