"""Persistent child chunk vector indexing service for ingestion ticket 3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_rag.indexing.dense_child_chunks import (
    DenseEmbeddingService,
    QdrantChildChunkStore,
    stable_qdrant_point_id,
)
from agentic_rag.ingestion_pipeline.chunk_persistence import CHILD_CHUNK_TYPE
from agentic_rag.storage.models import Chunk


@dataclass(slots=True, frozen=True)
class VectorIndexingResult:
    """Diagnostics for one document version child-chunk indexing pass."""

    document_version_id: str
    selected_child_chunks: int = 0
    upserted_child_chunks: int = 0
    failed_chunk_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class ChildChunkVectorIndexingService:
    """Embed persisted child chunks, upsert to Qdrant, and persist point ids."""

    session: Session
    embedding_service: DenseEmbeddingService
    store: QdrantChildChunkStore

    def index_document_version(self, *, document_version_id: str) -> VectorIndexingResult:
        child_chunks = self.get_child_chunks_for_document_version(document_version_id=document_version_id)
        if not child_chunks:
            return VectorIndexingResult(document_version_id=document_version_id)

        self.store.ensure_collection(vector_size=self.embedding_service.dimension)

        texts = [chunk.text for chunk in child_chunks]
        vectors = self.embedding_service.embed_texts(texts)
        points = [self._point_for_chunk(chunk=chunk, vector=vector) for chunk, vector in zip(child_chunks, vectors, strict=True)]

        upserted_count, failed_chunk_ids = self.store.upsert_points(points)
        failed_set = set(failed_chunk_ids)

        for chunk in child_chunks:
            if chunk.id in failed_set:
                continue
            point_id = self._point_id_for_chunk(chunk)
            if chunk.qdrant_point_id != point_id:
                chunk.qdrant_point_id = point_id

        self.session.flush()
        return VectorIndexingResult(
            document_version_id=document_version_id,
            selected_child_chunks=len(child_chunks),
            upserted_child_chunks=upserted_count,
            failed_chunk_ids=tuple(failed_chunk_ids),
        )

    def get_child_chunks_for_document_version(self, *, document_version_id: str) -> list[Chunk]:
        query = (
            select(Chunk)
            .where(Chunk.document_version_id == document_version_id)
            .where(Chunk.chunk_type == CHILD_CHUNK_TYPE)
            .order_by(Chunk.id.asc())
        )
        return list(self.session.execute(query).scalars().all())

    def _point_for_chunk(self, *, chunk: Chunk, vector: list[float]) -> dict[str, Any]:
        return {
            "id": self._point_id_for_chunk(chunk),
            "vector": vector,
            "payload": self._payload_for_chunk(chunk),
        }

    def _point_id_for_chunk(self, chunk: Chunk) -> str:
        deterministic_id = stable_qdrant_point_id(chunk.id)
        if chunk.qdrant_point_id == deterministic_id:
            return chunk.qdrant_point_id
        return deterministic_id

    def _payload_for_chunk(self, chunk: Chunk) -> dict[str, Any]:
        metadata = chunk.metadata_json or {}
        heading = chunk.heading or None
        if heading is None and chunk.section_path:
            heading = chunk.section_path.split(" > ")[-1]

        section_path = []
        if chunk.section_path:
            section_path = [item.strip() for item in chunk.section_path.split(">") if item.strip()]

        return {
            "document_id": chunk.document_id,
            "document_version_id": chunk.document_version_id,
            "chunk_id": chunk.id,
            "parent_chunk_id": chunk.parent_chunk_id,
            "source_name": metadata.get("source_name"),
            "heading": heading,
            "section_path": section_path,
        }


__all__ = ["VectorIndexingResult", "ChildChunkVectorIndexingService"]
