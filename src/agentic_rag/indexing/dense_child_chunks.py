"""Dense child-chunk embedding + Qdrant indexing pipeline.

Child chunks are the retrieval unit in this legal RAG architecture, so they are
embedded for semantic search and stored with parent linkage metadata. Preserving
``parent_chunk_id`` in payload is required so retrieval can expand to larger
parent context blocks after dense search returns matching child chunks.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid5

from agentic_rag.chunking.models import ChildChunk


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_COLLECTION_NAME = "legal_child_chunks_dense"


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


@dataclass(slots=True, frozen=True)
class DenseEmbeddingConfig:
    """Embedding configuration shared by indexing and query embedding paths."""

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 16
    device: str = "cpu"
    embedding_dimension: int | None = None


@dataclass(slots=True, frozen=True)
class DenseIndexingResult:
    """Structured diagnostics for dense child-chunk indexing runs."""

    total_chunks_received: int
    total_chunks_embedded: int
    total_chunks_indexed: int
    failed_chunk_ids: tuple[str, ...]
    collection_name: str
    embedding_model_name: str


@dataclass(slots=True)
class DenseEmbeddingService:
    """Embeds child chunk text in batches with a single reusable model instance."""

    config: DenseEmbeddingConfig = field(default_factory=DenseEmbeddingConfig)
    backend: EmbeddingBackend | None = None

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

    @property
    def dimension(self) -> int:
        return self.backend.dimension

    def embed_batch(self, chunks: Sequence[ChildChunk]) -> tuple[list[list[float]], list[str]]:
        """Return embeddings and failed IDs for one batch without stopping on partial failures."""

        if not chunks:
            return ([], [])

        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.child_chunk_id for chunk in chunks]
        try:
            vectors = self.backend.encode(texts, batch_size=self.config.batch_size)
            return (vectors, [])
        except Exception:
            vectors: list[list[float]] = []
            failed: list[str] = []
            for chunk in chunks:
                try:
                    encoded = self.backend.encode([chunk.text], batch_size=1)
                except Exception:
                    failed.append(chunk.child_chunk_id)
                    continue
                vectors.extend(encoded)
            successful_ids = [cid for cid in chunk_ids if cid not in set(failed)]
            if len(vectors) != len(successful_ids):
                raise RuntimeError("Embedding backend returned inconsistent vector count")
            return (vectors, failed)


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
            self.client.create_collection(
                self.collection_name,
                vectors_config={"size": vector_size, "distance": self.distance},
            )
            return

        existing = self.client.get_collection(self.collection_name)
        existing_size = int(existing.get("size", 0))
        existing_distance = str(existing.get("distance", ""))
        if existing_size != vector_size or existing_distance.lower() != self.distance.lower():
            raise ValueError(
                "Existing collection vector config mismatch: "
                f"size={existing_size}, distance={existing_distance}; "
                f"expected size={vector_size}, distance={self.distance}"
            )

    def upsert_chunks(self, chunks: Sequence[ChildChunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        if not chunks:
            return 0

        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            points.append(
                {
                    "id": stable_qdrant_point_id(chunk.child_chunk_id),
                    "vector": list(vector),
                    "payload": child_chunk_payload(chunk),
                }
            )
        self.client.upsert(self.collection_name, points=points)
        return len(points)

    def get_by_child_chunk_id(self, child_chunk_id: str) -> dict[str, Any] | None:
        points = self.client.retrieve(self.collection_name, ids=[stable_qdrant_point_id(child_chunk_id)])
        return points[0] if points else None


@dataclass(slots=True)
class ChildChunkDenseIndexer:
    """Orchestrates batch embedding and idempotent dense upsert for child chunks."""

    embedding_service: DenseEmbeddingService
    store: QdrantChildChunkStore
    batch_size: int = 64

    def index_child_chunks_dense(self, child_chunks: Iterable[ChildChunk]) -> DenseIndexingResult:
        """Embed child chunks and upsert dense points, preserving parent link metadata."""

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        deduped_iter = _iter_deduped_child_chunks(child_chunks)
        total_received = 0
        total_embedded = 0
        total_indexed = 0
        failed_ids: list[str] = []

        self.store.ensure_collection(vector_size=self.embedding_service.dimension)

        for batch in _iter_batches(deduped_iter, self.batch_size):
            total_received += len(batch)
            vectors, batch_failures = self.embedding_service.embed_batch(batch)
            failed_ids.extend(batch_failures)

            if not batch:
                continue

            if not batch_failures:
                embedded_chunks = batch
            else:
                failed_set = set(batch_failures)
                embedded_chunks = [chunk for chunk in batch if chunk.child_chunk_id not in failed_set]

            total_embedded += len(embedded_chunks)
            if not embedded_chunks:
                continue

            offset = 0
            for chunk_group in _iter_batches(iter(embedded_chunks), self.embedding_service.config.batch_size):
                group_vectors = vectors[offset : offset + len(chunk_group)]
                offset += len(chunk_group)
                indexed = self.store.upsert_chunks(chunk_group, group_vectors)
                total_indexed += indexed

        return DenseIndexingResult(
            total_chunks_received=total_received,
            total_chunks_embedded=total_embedded,
            total_chunks_indexed=total_indexed,
            failed_chunk_ids=tuple(failed_ids),
            collection_name=self.store.collection_name,
            embedding_model_name=self.embedding_service.config.model_name,
        )


class _SentenceTransformerEmbeddingBackend:
    """Sentence-transformers backend for dense embeddings.

    This lazily loads once per service instance and keeps configuration
    identical between indexing and query-time embedding.
    """

    def __init__(self, *, model_name: str, device: str, expected_dimension: int | None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - exercised only when dependency is absent
            raise RuntimeError(
                "sentence-transformers is required for default dense embedding backend"
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        if expected_dimension is not None and expected_dimension != self._dimension:
            raise ValueError(
                f"Configured embedding_dimension={expected_dimension} does not match model dimension={self._dimension}"
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], *, batch_size: int) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [vector.tolist() for vector in vectors]


def child_chunk_payload(chunk: ChildChunk) -> dict[str, Any]:
    """Build Qdrant payload with retrieval metadata and parent linkage for expansion."""

    return {
        "schema_version": 1,
        "child_chunk_id": chunk.child_chunk_id,
        "parent_chunk_id": chunk.parent_chunk_id,
        "document_id": chunk.document_id,
        "source": chunk.source,
        "source_name": chunk.source_name,
        "heading_path": list(chunk.heading_path),
        "child_order": chunk.child_order,
        "token_count": chunk.token_count,
        "text": chunk.text,
    }


def stable_qdrant_point_id(child_chunk_id: str) -> str:
    """Return deterministic point IDs to support idempotent upsert updates."""

    return str(uuid5(NAMESPACE_URL, f"agentic-rag-child:{child_chunk_id}"))


def _iter_deduped_child_chunks(chunks: Iterable[ChildChunk]) -> Iterator[ChildChunk]:
    by_id: dict[str, ChildChunk] = {}
    for chunk in chunks:
        by_id[chunk.child_chunk_id] = chunk
    for chunk_id in sorted(by_id):
        yield by_id[chunk_id]


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
    "QdrantChildChunkStore",
    "ChildChunkDenseIndexer",
    "child_chunk_payload",
    "stable_qdrant_point_id",
]
