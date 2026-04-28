"""Postgres-backed repository for persisted parent/child chunk reads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_rag.retrieval.parent_child import ParentChunkRepository, ParentChunkResult
from agentic_rag.storage.models import Chunk


@dataclass(slots=True, frozen=True)
class PersistedChunk:
    """Retrieval-friendly representation of a persisted chunk row."""

    chunk_id: str
    document_id: str
    document_version_id: str
    chunk_type: str
    text: str
    parent_chunk_id: str | None
    heading: str
    section_path: str
    qdrant_point_id: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class PostgresChunkRepository(ParentChunkRepository):
    """Read persisted parent/child chunk rows through a typed repository API."""

    session: Session

    def get_chunk_by_id(self, chunk_id: str) -> PersistedChunk | None:
        row = self.session.get(Chunk, chunk_id)
        if row is None:
            return None
        return _to_persisted_chunk(row)

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[PersistedChunk]:
        ordered_ids = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id]
        if not ordered_ids:
            return []
        rows = self.session.execute(select(Chunk).where(Chunk.id.in_(ordered_ids))).scalars().all()
        by_id = {row.id: row for row in rows}
        return [_to_persisted_chunk(by_id[chunk_id]) for chunk_id in ordered_ids if chunk_id in by_id]

    def get_parent_chunk(self, parent_chunk_id: str) -> ParentChunkResult | None:
        row = self.session.get(Chunk, parent_chunk_id)
        if row is None or row.chunk_type != "parent":
            return None
        return _to_parent_chunk_result(row)

    def get_parent_chunks(self, parent_chunk_ids: Sequence[str]) -> list[ParentChunkResult]:
        rows = self.get_chunks_by_ids(parent_chunk_ids)
        return [_persisted_to_parent_chunk_result(row) for row in rows if row.chunk_type == "parent"]

    def get_children_for_parent(self, parent_chunk_id: str) -> list[PersistedChunk]:
        rows = (
            self.session.execute(
                select(Chunk)
                .where(Chunk.parent_chunk_id == parent_chunk_id)
                .order_by(Chunk.id.asc())
            )
            .scalars()
            .all()
        )
        return [_to_persisted_chunk(row) for row in rows]

    def get_chunks_for_document_version(self, document_version_id: str) -> list[PersistedChunk]:
        rows = (
            self.session.execute(
                select(Chunk)
                .where(Chunk.document_version_id == document_version_id)
                .order_by(Chunk.chunk_type.asc(), Chunk.id.asc())
            )
            .scalars()
            .all()
        )
        return [_to_persisted_chunk(row) for row in rows]

    def resolve_qdrant_point_id(self, point_id: str) -> PersistedChunk | None:
        row = (
            self.session.execute(select(Chunk).where(Chunk.qdrant_point_id == point_id))
            .scalars()
            .first()
        )
        if row is None:
            return None
        return _to_persisted_chunk(row)

    def get_by_ids(self, parent_ids: Sequence[str]) -> list[ParentChunkResult]:
        """ParentChunkRepository interface for retrieval parent expansion."""

        return self.get_parent_chunks(parent_ids)


def _to_persisted_chunk(row: Chunk) -> PersistedChunk:
    return PersistedChunk(
        chunk_id=row.id,
        document_id=row.document_id,
        document_version_id=row.document_version_id,
        chunk_type=row.chunk_type,
        text=row.text,
        parent_chunk_id=row.parent_chunk_id,
        heading=row.heading or "",
        section_path=row.section_path or "",
        qdrant_point_id=row.qdrant_point_id or "",
        metadata=dict(row.metadata_json or {}),
    )


def _to_parent_chunk_result(row: Chunk) -> ParentChunkResult:
    return _persisted_to_parent_chunk_result(_to_persisted_chunk(row))


def _persisted_to_parent_chunk_result(chunk: PersistedChunk) -> ParentChunkResult:
    heading_path_raw = chunk.metadata.get("heading_path", ())
    heading_path = (
        tuple(str(item) for item in heading_path_raw)
        if isinstance(heading_path_raw, Sequence) and not isinstance(heading_path_raw, (str, bytes))
        else ()
    )
    return ParentChunkResult(
        parent_chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        source=str(chunk.metadata.get("source", "")),
        source_name=str(chunk.metadata.get("source_name", "")),
        heading_path=heading_path,
        heading_text=str(chunk.heading or chunk.metadata.get("heading_text", "")),
        parent_order=int(chunk.metadata.get("parent_order", 0) or 0),
        part_number=int(chunk.metadata.get("part_number", 1) or 1),
        total_parts=int(chunk.metadata.get("total_parts", 1) or 1),
        metadata={
            key: value
            for key, value in chunk.metadata.items()
            if key
            not in {
                "source",
                "source_name",
                "heading_path",
                "heading_text",
                "parent_order",
                "part_number",
                "total_parts",
            }
        },
    )


__all__ = ["PersistedChunk", "PostgresChunkRepository"]
