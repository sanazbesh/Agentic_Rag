from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.chunking.models import ChildChunk
from agentic_rag.indexing import (
    ChildChunkDenseIndexer,
    DenseEmbeddingConfig,
    DenseEmbeddingService,
    QdrantChildChunkStore,
    stable_qdrant_point_id,
)


class FakeEmbeddingBackend:
    def __init__(self, *, dimension: int = 4, fail_texts: set[str] | None = None) -> None:
        self._dimension = dimension
        self.fail_texts = fail_texts or set()
        self.encode_calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        self.encode_calls += 1
        if any(text in self.fail_texts for text in texts):
            raise RuntimeError("synthetic embedding failure")
        vectors: list[list[float]] = []
        for text in texts:
            base = float((sum(ord(ch) for ch in text) % 13) + 1)
            vectors.append([base + i for i in range(self._dimension)])
        return vectors


class InMemoryQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, collection_name: str, *, vectors_config: dict[str, object]) -> None:
        self.collections[collection_name] = {
            "config": dict(vectors_config),
            "points": {},
        }

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
        found: list[dict[str, object]] = []
        for pid in ids:
            point = bucket.get(pid)
            if point is not None:
                found.append(dict(point))
        return found


def _chunk(child_id: str, text: str, *, parent_id: str = "parent-1") -> ChildChunk:
    return ChildChunk(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id="doc-1",
        source="s3://legal/doc.md",
        source_name="doc.md",
        text=text,
        child_order=1,
        token_count=42,
        heading_path=("Contracts", "Definitions"),
    )


def test_dense_embeddings_generated_and_indexed_with_metadata() -> None:
    backend = FakeEmbeddingBackend(dimension=4)
    service = DenseEmbeddingService(config=DenseEmbeddingConfig(batch_size=2), backend=backend)
    client = InMemoryQdrantClient()
    store = QdrantChildChunkStore(client=client, collection_name="child_dense")
    indexer = ChildChunkDenseIndexer(embedding_service=service, store=store, batch_size=2)

    result = indexer.index_child_chunks_dense([_chunk("child-1", "alpha"), _chunk("child-2", "beta")])

    assert result.total_chunks_received == 2
    assert result.total_chunks_embedded == 2
    assert result.total_chunks_indexed == 2
    assert result.failed_chunk_ids == ()
    assert result.embedding_model_name == "Qwen/Qwen3-Embedding-8B"

    saved = store.get_by_child_chunk_id("child-1")
    assert saved is not None
    assert saved["id"] == stable_qdrant_point_id("child-1")
    payload = saved["payload"]
    assert payload["child_chunk_id"] == "child-1"
    assert payload["parent_chunk_id"] == "parent-1"
    assert payload["document_id"] == "doc-1"
    assert payload["source"] == "s3://legal/doc.md"
    assert payload["source_name"] == "doc.md"
    assert payload["heading_path"] == ["Contracts", "Definitions"]
    assert payload["token_count"] == 42
    assert payload["text"] == "alpha"


def test_empty_input_is_safe_and_returns_zero_stats() -> None:
    service = DenseEmbeddingService(config=DenseEmbeddingConfig(), backend=FakeEmbeddingBackend())
    client = InMemoryQdrantClient()
    store = QdrantChildChunkStore(client=client)
    indexer = ChildChunkDenseIndexer(embedding_service=service, store=store)

    result = indexer.index_child_chunks_dense([])

    assert result.total_chunks_received == 0
    assert result.total_chunks_embedded == 0
    assert result.total_chunks_indexed == 0
    assert result.failed_chunk_ids == ()


def test_duplicate_child_ids_are_idempotent_and_update_existing_points() -> None:
    service = DenseEmbeddingService(config=DenseEmbeddingConfig(), backend=FakeEmbeddingBackend())
    client = InMemoryQdrantClient()
    store = QdrantChildChunkStore(client=client)
    indexer = ChildChunkDenseIndexer(embedding_service=service, store=store)

    first = _chunk("child-dup", "original text")
    second = _chunk("child-dup", "updated text")
    result = indexer.index_child_chunks_dense([first, second])

    assert result.total_chunks_received == 1
    assert result.total_chunks_indexed == 1
    saved = store.get_by_child_chunk_id("child-dup")
    assert saved is not None
    assert saved["payload"]["text"] == "updated text"


def test_existing_collection_vector_mismatch_fails_fast() -> None:
    service = DenseEmbeddingService(config=DenseEmbeddingConfig(), backend=FakeEmbeddingBackend(dimension=4))
    client = InMemoryQdrantClient()
    client.create_collection("legal_child_chunks_dense", vectors_config={"size": 8, "distance": "Cosine"})
    store = QdrantChildChunkStore(client=client)
    indexer = ChildChunkDenseIndexer(embedding_service=service, store=store)

    try:
        indexer.index_child_chunks_dense([_chunk("child-1", "hello")])
    except ValueError as exc:
        assert "mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected vector configuration mismatch error")


def test_embedding_failures_are_tracked_and_valid_chunks_still_indexed() -> None:
    backend = FakeEmbeddingBackend(fail_texts={"bad"})
    service = DenseEmbeddingService(config=DenseEmbeddingConfig(), backend=backend)
    client = InMemoryQdrantClient()
    store = QdrantChildChunkStore(client=client)
    indexer = ChildChunkDenseIndexer(embedding_service=service, store=store)

    result = indexer.index_child_chunks_dense([_chunk("ok-1", "good"), _chunk("bad-1", "bad"), _chunk("ok-2", "fine")])

    assert sorted(result.failed_chunk_ids) == ["bad-1"]
    assert result.total_chunks_received == 3
    assert result.total_chunks_embedded == 2
    assert result.total_chunks_indexed == 2
    assert store.get_by_child_chunk_id("ok-1") is not None
    assert store.get_by_child_chunk_id("ok-2") is not None
    assert store.get_by_child_chunk_id("bad-1") is None
