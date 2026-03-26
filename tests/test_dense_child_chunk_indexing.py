from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from agentic_rag.chunking.models import ChildChunk
from agentic_rag.indexing import (
    ChildChunkDenseIndexer,
    DenseEmbeddingConfig,
    DenseEmbeddingService,
    QdrantChildChunkStore,
    stable_qdrant_point_id,
)


class FakeEmbeddingBackend:
    def __init__(
        self,
        *,
        dimension: int = 4,
        fail_texts: set[str] | None = None,
        wrong_dimension_texts: set[str] | None = None,
    ) -> None:
        self._dimension = dimension
        self.fail_texts = fail_texts or set()
        self.wrong_dimension_texts = wrong_dimension_texts or set()
        self.encode_calls = 0
        self.batch_sizes: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        self.encode_calls += 1
        self.batch_sizes.append(len(texts))
        if any(text in self.fail_texts for text in texts):
            raise RuntimeError("synthetic embedding failure")
        vectors: list[list[float]] = []
        for text in texts:
            if text in self.wrong_dimension_texts:
                vectors.append([1.0])
                continue
            base = float((sum(ord(ch) for ch in text) % 13) + 1)
            vectors.append([base + i for i in range(self._dimension)])
        return vectors


class InMemoryQdrantClient:
    def __init__(self, *, fail_upsert_ids: set[str] | None = None) -> None:
        self.collections: dict[str, dict[str, object]] = {}
        self.fail_upsert_ids = fail_upsert_ids or set()

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
        for point in points:
            payload = point["payload"]
            assert isinstance(payload, dict)
            child_chunk_id = str(payload.get("child_chunk_id", ""))
            if child_chunk_id in self.fail_upsert_ids:
                raise RuntimeError("synthetic upsert failure")

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


def _chunk(
    child_id: str,
    text: str,
    *,
    parent_id: str = "parent-1",
    doc_id: str = "doc-1",
    order: int = 1,
) -> ChildChunk:
    return ChildChunk(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id=doc_id,
        source="s3://legal/doc.md",
        source_name="doc.md",
        text=text,
        child_order=order,
        token_count=42,
        heading_path=("Contracts", "Definitions"),
    )


def _build_indexer(
    *,
    backend: FakeEmbeddingBackend | None = None,
    client: InMemoryQdrantClient | None = None,
    embed_batch_size: int = 2,
    index_batch_size: int = 2,
    max_input_tokens: int | None = None,
) -> tuple[ChildChunkDenseIndexer, FakeEmbeddingBackend, InMemoryQdrantClient]:
    actual_backend = backend or FakeEmbeddingBackend(dimension=4)
    service = DenseEmbeddingService(
        config=DenseEmbeddingConfig(batch_size=embed_batch_size, max_input_tokens=max_input_tokens),
        backend=actual_backend,
    )
    actual_client = client or InMemoryQdrantClient()
    store = QdrantChildChunkStore(client=actual_client, collection_name="child_dense")
    return (ChildChunkDenseIndexer(embedding_service=service, store=store, batch_size=index_batch_size), actual_backend, actual_client)


def test_dense_indexing_happy_path() -> None:
    indexer, _, _ = _build_indexer()

    result = indexer.index_child_chunks_dense([_chunk("child-1", "alpha"), _chunk("child-2", "beta")])

    assert result.total_chunks_received == 2
    assert result.total_chunks_processed == 2
    assert result.total_chunks_embedded == 2
    assert result.total_chunks_indexed == 2
    assert result.failed_chunk_ids == []
    assert result.skipped_chunk_ids == []
    assert result.embedding_model_name == "Qwen/Qwen3-Embedding-8B"


def test_payload_schema_contains_required_fields() -> None:
    indexer, _, _ = _build_indexer()
    indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])

    saved = indexer.store.get_by_child_chunk_id("child-1")
    assert saved is not None
    assert saved["id"] == stable_qdrant_point_id("child-1")
    payload = saved["payload"]
    assert payload["child_chunk_id"] == "child-1"
    assert payload["parent_chunk_id"] == "parent-1"
    assert payload["document_id"] == "doc-1"
    assert payload["source"] == "s3://legal/doc.md"
    assert payload["source_name"] == "doc.md"
    assert payload["heading"] == "Definitions"
    assert payload["section_path"] == ["Contracts", "Definitions"]
    assert payload["child_order"] == 1
    assert payload["token_count"] == 42
    assert payload["text"] == "alpha"
    assert payload["schema_version"] == "1.0"
    assert "indexed_at" in payload


def test_collection_created_with_expected_dense_config() -> None:
    indexer, _, client = _build_indexer()
    indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])

    config = client.get_collection("child_dense")
    assert config["size"] == 4
    assert config["distance"] == "Cosine"


def test_existing_collection_vector_size_mismatch_fails_fast() -> None:
    indexer, _, client = _build_indexer()
    client.create_collection("child_dense", vectors_config={"size": 8, "distance": "Cosine"})

    with pytest.raises(ValueError, match="size mismatch"):
        indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])


def test_existing_collection_distance_mismatch_fails_fast() -> None:
    indexer, _, client = _build_indexer()
    client.create_collection("child_dense", vectors_config={"size": 4, "distance": "Dot"})

    with pytest.raises(ValueError, match="distance mismatch"):
        indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])


def test_empty_input_returns_typed_result_with_warning() -> None:
    indexer, _, _ = _build_indexer()

    result = indexer.index_child_chunks_dense([])

    assert result.total_chunks_received == 0
    assert result.total_chunks_processed == 0
    assert result.total_chunks_embedded == 0
    assert result.total_chunks_indexed == 0
    assert result.failed_chunk_ids == []
    assert result.skipped_chunk_ids == []
    assert any("No child chunks" in warning for warning in result.warnings)


def test_duplicate_child_ids_are_deduped_deterministically_last_write_wins() -> None:
    indexer, _, _ = _build_indexer()

    first = _chunk("child-dup", "original text")
    second = _chunk("child-dup", "updated text")
    result = indexer.index_child_chunks_dense([first, second])

    assert result.total_chunks_received == 1
    assert any("Duplicate child_chunk_id" in warning for warning in result.warnings)
    saved = indexer.store.get_by_child_chunk_id("child-dup")
    assert saved is not None
    assert saved["payload"]["text"] == "updated text"


def test_missing_required_fields_are_skipped() -> None:
    indexer, _, _ = _build_indexer()
    invalid = _chunk("", "alpha")
    missing_parent = _chunk("child-2", "alpha", parent_id="")
    missing_doc = _chunk("child-3", "alpha", doc_id="")

    result = indexer.index_child_chunks_dense([invalid, missing_parent, missing_doc])

    assert result.total_chunks_processed == 0
    assert sorted(result.skipped_chunk_ids) == ["", "child-2", "child-3"]
    assert result.total_chunks_indexed == 0


def test_whitespace_only_text_is_skipped() -> None:
    indexer, _, _ = _build_indexer()

    result = indexer.index_child_chunks_dense([_chunk("child-1", "   ")])

    assert result.skipped_chunk_ids == ["child-1"]
    assert result.total_chunks_embedded == 0


def test_overlong_text_is_explicitly_skipped_with_warning() -> None:
    indexer, _, _ = _build_indexer(max_input_tokens=3)

    result = indexer.index_child_chunks_dense([_chunk("child-1", "one two three four")])

    assert result.skipped_chunk_ids == ["child-1"]
    assert any("max_input_tokens" in warning for warning in result.warnings)


def test_partial_embedding_failure_does_not_block_successful_chunks() -> None:
    backend = FakeEmbeddingBackend(fail_texts={"bad"})
    indexer, _, _ = _build_indexer(backend=backend)

    result = indexer.index_child_chunks_dense([_chunk("ok-1", "good"), _chunk("bad-1", "bad"), _chunk("ok-2", "fine")])

    assert result.failed_chunk_ids == ["bad-1"]
    assert result.total_chunks_embedded == 2
    assert result.total_chunks_indexed == 2
    assert indexer.store.get_by_child_chunk_id("ok-1") is not None
    assert indexer.store.get_by_child_chunk_id("ok-2") is not None


def test_partial_upsert_failure_tracks_failed_chunk_ids() -> None:
    client = InMemoryQdrantClient(fail_upsert_ids={"bad-1"})
    indexer, _, _ = _build_indexer(client=client)

    result = indexer.index_child_chunks_dense([_chunk("ok-1", "good"), _chunk("bad-1", "fine")])

    assert result.total_chunks_embedded == 2
    assert result.total_chunks_indexed == 1
    assert result.failed_chunk_ids == ["bad-1"]


def test_idempotent_reindexing_does_not_create_duplicates() -> None:
    indexer, _, client = _build_indexer()

    chunks = [_chunk("child-1", "alpha"), _chunk("child-2", "beta")]
    indexer.index_child_chunks_dense(chunks)
    indexer.index_child_chunks_dense(chunks)

    points = client.collections["child_dense"]["points"]
    assert isinstance(points, dict)
    assert len(points) == 2


def test_upsert_updates_existing_point_payload() -> None:
    indexer, _, _ = _build_indexer()

    indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])
    indexer.index_child_chunks_dense([_chunk("child-1", "alpha updated")])

    saved = indexer.store.get_by_child_chunk_id("child-1")
    assert saved is not None
    assert saved["payload"]["text"] == "alpha updated"


def test_batching_respects_embedding_batch_size() -> None:
    backend = FakeEmbeddingBackend()
    indexer, _, _ = _build_indexer(backend=backend, embed_batch_size=2, index_batch_size=5)

    indexer.index_child_chunks_dense([_chunk(f"child-{idx}", f"text-{idx}") for idx in range(5)])

    assert backend.batch_sizes == [2, 2, 1]


def test_deterministic_point_id_and_indexing_behavior() -> None:
    indexer, _, _ = _build_indexer()

    one = indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])
    two = indexer.index_child_chunks_dense([_chunk("child-1", "alpha")])

    assert stable_qdrant_point_id("child-1") == stable_qdrant_point_id("child-1")
    assert one.total_chunks_indexed == two.total_chunks_indexed == 1
    assert one.failed_chunk_ids == two.failed_chunk_ids == []


def test_invalid_embedding_dimension_is_tracked_as_failure() -> None:
    backend = FakeEmbeddingBackend(wrong_dimension_texts={"bad"})
    indexer, _, _ = _build_indexer(backend=backend)

    result = indexer.index_child_chunks_dense([_chunk("ok-1", "good"), _chunk("bad-1", "bad")])

    assert result.total_chunks_embedded == 1
    assert result.total_chunks_indexed == 1
    assert result.failed_chunk_ids == ["bad-1"]


def test_observability_logs_do_not_leak_full_chunk_text(caplog: pytest.LogCaptureFixture) -> None:
    indexer, _, _ = _build_indexer()
    secret_text = "CONFIDENTIAL LEGAL CLAUSE CONTENT"

    with caplog.at_level(logging.INFO):
        indexer.index_child_chunks_dense([_chunk("child-1", secret_text)])

    full_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "dense indexing completed" in full_logs
    assert secret_text not in full_logs
