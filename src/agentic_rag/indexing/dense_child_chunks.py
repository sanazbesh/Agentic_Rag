"""Dense child-chunk embedding + Qdrant indexing pipeline.

Child chunks are the retrieval unit in this legal RAG architecture. This module
implements deterministic dense embedding generation and idempotent Qdrant
upserts while preserving parent linkage metadata for downstream parent fetch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from agentic_rag.chunking.models import ChildChunk


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_COLLECTION_NAME = "legal_child_chunks_dense"
DEFAULT_PAYLOAD_SCHEMA_VERSION = "1.0"

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Backend protocol for reusable embedding model instances."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]: ...


class QdrantClientLike(Protocol):
    """Subset of Qdrant client operations used by the dense indexer."""

    def collection_exists(self, collection_name: str) -> bool: ...

    def get_collection(self, collection_name: str) -> dict[str, Any]: ...

    def create_collection(self, collection_name: str, *, vectors_config: dict[str, Any]) -> None: ...

    def upsert(self, collection_name: str, *, points: Sequence[dict[str, Any]]) -> None: ...

    def retrieve(self, collection_name: str, *, ids: Sequence[str]) -> list[dict[str, Any]]: ...

    def delete(self, collection_name: str, *, points_selector: dict[str, Any]) -> None: ...


@dataclass(slots=True, frozen=True)
class DenseEmbeddingConfig:
    """Embedding configuration shared by indexing and query embedding paths."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 16
    device: str = "cpu"
    embedding_dimension: int | None = None
    max_input_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class ChildChunkQdrantPayload:
    """Strict and stable payload schema for each child chunk point."""

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    source: str | None
    source_name: str | None
    heading: str | None
    section_path: list[str]
    child_order: int | None
    token_count: int | None
    text: str
    document_type: str | None = None
    jurisdiction: str | None = None
    clause_type: str | None = None
    indexed_at: str | None = None
    schema_version: str = DEFAULT_PAYLOAD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DenseIndexingResult:
    """Structured diagnostics for dense child-chunk indexing runs."""

    collection_name: str
    embedding_model_name: str
    embedding_dimension: int
    total_chunks_received: int = 0
    total_chunks_processed: int = 0
    total_chunks_embedded: int = 0
    total_chunks_indexed: int = 0
    failed_chunk_ids: list[str] = field(default_factory=list)
    skipped_chunk_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DenseEmbeddingService:
    """Embeds child chunk text in batches with a single reusable model instance."""

    config: DenseEmbeddingConfig = field(default_factory=DenseEmbeddingConfig)
    backend: EmbeddingBackend | None = None
    token_counter: Callable[[str], int] | None = None

    def __post_init__(self) -> None:
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.backend is None:
            self.backend = _SentenceTransformerEmbeddingBackend(
                model_name=self.config.model_name,
                device=self.config.device,
                expected_dimension=self.config.embedding_dimension,
            )
        if self.config.embedding_dimension is not None and self.backend.dimension != self.config.embedding_dimension:
            raise ValueError(
                "Embedding backend dimension mismatch: "
                f"expected={self.config.embedding_dimension}, actual={self.backend.dimension}"
            )
        if self.token_counter is None:
            self.token_counter = lambda text: len(text.split())

    @property
    def dimension(self) -> int:
        return self.backend.dimension

    def is_text_within_limits(self, text: str) -> bool:
        if self.config.max_input_tokens is None:
            return True
        return self.token_counter(text) <= self.config.max_input_tokens

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.backend.encode(texts, batch_size=self.config.batch_size)
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding backend returned inconsistent vector count")

        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"Invalid embedding dimension returned by backend: expected={self.dimension}, actual={len(vector)}"
                )
        return vectors


@dataclass(slots=True)
class QdrantChildChunkStore:
    """Qdrant wrapper for dense child-chunk upsert/retrieval with schema validation."""

    client: QdrantClientLike
    collection_name: str = DEFAULT_COLLECTION_NAME
    distance: str = "Cosine"

    def ensure_collection(self, *, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be > 0")

        if not self.client.collection_exists(self.collection_name):
            try:
                self.client.create_collection(
                    self.collection_name,
                    vectors_config={"size": vector_size, "distance": self.distance},
                )
            except Exception:
                if not self.client.collection_exists(self.collection_name):
                    raise

        existing = self.client.get_collection(self.collection_name)
        existing_size = int(existing.get("size", 0))
        existing_distance = str(existing.get("distance", ""))
        if existing_size != vector_size:
            raise ValueError(
                "Existing collection vector size mismatch: "
                f"size={existing_size}, expected={vector_size}"
            )
        if existing_distance.lower() != self.distance.lower():
            raise ValueError(
                "Existing collection distance mismatch: "
                f"distance={existing_distance}, expected={self.distance}"
            )

    def upsert_points(self, points: Sequence[dict[str, Any]]) -> tuple[int, list[str]]:
        if not points:
            return (0, [])
        try:
            self.client.upsert(self.collection_name, points=points)
            return (len(points), [])
        except Exception:
            failed: list[str] = []
            indexed = 0
            for point in points:
                try:
                    self.client.upsert(self.collection_name, points=[point])
                except Exception:
                    payload = point.get("payload", {})
                    failed.append(str(payload.get("child_chunk_id", "")))
                else:
                    indexed += 1
            return (indexed, failed)

    def get_by_child_chunk_id(self, child_chunk_id: str) -> dict[str, Any] | None:
        points = self.client.retrieve(self.collection_name, ids=[stable_qdrant_point_id(child_chunk_id)])
        return points[0] if points else None

    def delete_points(self, point_ids: Sequence[str]) -> None:
        if not point_ids:
            return
        self.client.delete(
            self.collection_name,
            points_selector={"points": list(point_ids)},
        )


@dataclass(slots=True)
class ChildChunkDenseIndexer:
    """Orchestrates validation, batched embedding, and idempotent dense upsert."""

    embedding_service: DenseEmbeddingService
    store: QdrantChildChunkStore
    batch_size: int = 64

    def index_child_chunks_dense(self, child_chunks: Iterable[ChildChunk]) -> DenseIndexingResult:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        deduped_chunks, duplicate_ids = _dedupe_chunks(child_chunks)
        result = DenseIndexingResult(
            collection_name=self.store.collection_name,
            embedding_model_name=self.embedding_service.config.model_name,
            embedding_dimension=self.embedding_service.dimension,
            total_chunks_received=len(deduped_chunks),
        )

        if duplicate_ids:
            dup_msg = (
                "Duplicate child_chunk_id values detected; deterministic last-write-wins dedup applied: "
                f"{sorted(duplicate_ids)}"
            )
            result.warnings.append(dup_msg)
            logger.warning("duplicate child chunk ids", extra={"duplicate_child_chunk_ids": sorted(duplicate_ids)})

        if not deduped_chunks:
            result.warnings.append("No child chunks were provided for dense indexing")
            logger.info(
                "dense indexing completed",
                extra={"received": 0, "processed": 0, "embedded": 0, "indexed": 0},
            )
            return result

        self.store.ensure_collection(vector_size=self.embedding_service.dimension)

        for batch in _iter_batches(iter(deduped_chunks), self.batch_size):
            valid_chunks: list[ChildChunk] = []
            for chunk in batch:
                validation_error = _validate_chunk(chunk)
                if validation_error:
                    result.skipped_chunk_ids.append(chunk.child_chunk_id or "")
                    result.warnings.append(validation_error)
                    continue
                if not self.embedding_service.is_text_within_limits(chunk.text):
                    result.skipped_chunk_ids.append(chunk.child_chunk_id)
                    result.warnings.append(
                        f"Chunk {chunk.child_chunk_id} skipped due to max_input_tokens limit "
                        f"({self.embedding_service.config.max_input_tokens})"
                    )
                    continue
                valid_chunks.append(chunk)

            result.total_chunks_processed += len(valid_chunks)
            if not valid_chunks:
                continue

            for embed_batch in _iter_batches(iter(valid_chunks), self.embedding_service.config.batch_size):
                embedded_chunks, vectors = self._embed_chunk_batch(embed_batch, result)
                if not embedded_chunks:
                    continue

                points = [
                    {
                        "id": stable_qdrant_point_id(chunk.child_chunk_id),
                        "vector": vector,
                        "payload": child_chunk_payload(chunk).to_dict(),
                    }
                    for chunk, vector in zip(embedded_chunks, vectors, strict=True)
                ]
                indexed_count, upsert_failed_ids = self.store.upsert_points(points)
                result.total_chunks_indexed += indexed_count
                result.failed_chunk_ids.extend(upsert_failed_ids)

        logger.info(
            "dense indexing completed",
            extra={
                "received": result.total_chunks_received,
                "processed": result.total_chunks_processed,
                "embedded": result.total_chunks_embedded,
                "indexed": result.total_chunks_indexed,
                "failed_chunk_ids": result.failed_chunk_ids,
                "skipped_chunk_ids": result.skipped_chunk_ids,
            },
        )
        return result

    def _embed_chunk_batch(
        self, chunks: Sequence[ChildChunk], result: DenseIndexingResult
    ) -> tuple[list[ChildChunk], list[list[float]]]:
        texts = [chunk.text for chunk in chunks]
        try:
            vectors = self.embedding_service.embed_texts(texts)
            result.total_chunks_embedded += len(chunks)
            return (list(chunks), vectors)
        except Exception:
            embedded_chunks: list[ChildChunk] = []
            vectors: list[list[float]] = []
            for chunk in chunks:
                try:
                    one = self.embedding_service.embed_texts([chunk.text])
                except Exception:
                    result.failed_chunk_ids.append(chunk.child_chunk_id)
                else:
                    embedded_chunks.append(chunk)
                    vectors.extend(one)
                    result.total_chunks_embedded += 1
            return (embedded_chunks, vectors)


def child_chunk_payload(chunk: ChildChunk) -> ChildChunkQdrantPayload:
    """Build Qdrant payload with strict schema and parent-child traceability."""

    heading = chunk.heading_path[-1] if chunk.heading_path else None
    return ChildChunkQdrantPayload(
        child_chunk_id=chunk.child_chunk_id,
        parent_chunk_id=chunk.parent_chunk_id,
        document_id=chunk.document_id,
        source=chunk.source,
        source_name=chunk.source_name,
        heading=heading,
        section_path=list(chunk.heading_path),
        child_order=chunk.child_order,
        token_count=chunk.token_count,
        text=chunk.text,
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )


def stable_qdrant_point_id(child_chunk_id: str) -> str:
    """Return deterministic point IDs to support idempotent upsert updates."""

    return str(uuid5(NAMESPACE_URL, f"agentic-rag-child:{child_chunk_id}"))


def _validate_chunk(chunk: ChildChunk) -> str | None:
    if not chunk.child_chunk_id.strip():
        return "Chunk skipped due to missing child_chunk_id"
    if not chunk.parent_chunk_id.strip():
        return f"Chunk {chunk.child_chunk_id} skipped due to missing parent_chunk_id"
    if not chunk.document_id.strip():
        return f"Chunk {chunk.child_chunk_id} skipped due to missing document_id"
    if not chunk.text.strip():
        return f"Chunk {chunk.child_chunk_id} skipped due to empty or whitespace-only text"
    return None


def _dedupe_chunks(chunks: Iterable[ChildChunk]) -> tuple[list[ChildChunk], set[str]]:
    by_id: dict[str, ChildChunk] = {}
    duplicates: set[str] = set()
    for chunk in chunks:
        if chunk.child_chunk_id in by_id:
            duplicates.add(chunk.child_chunk_id)
        by_id[chunk.child_chunk_id] = chunk
    ordered_ids = sorted(by_id.keys())
    return ([by_id[chunk_id] for chunk_id in ordered_ids], duplicates)


def _iter_batches(items: Iterable[ChildChunk], size: int) -> Iterator[list[ChildChunk]]:
    if size <= 0:
        raise ValueError("size must be greater than zero")
    batch: list[ChildChunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_COLLECTION_NAME",
    "DenseEmbeddingConfig",
    "DenseEmbeddingService",
    "DenseIndexingResult",
    "ChildChunkQdrantPayload",
    "QdrantChildChunkStore",
    "ChildChunkDenseIndexer",
    "child_chunk_payload",
    "stable_qdrant_point_id",
]
