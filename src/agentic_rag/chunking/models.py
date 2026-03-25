"""Typed parent-child chunk records and storage-ready views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParentChunk:
    """A structure-aware markdown context chunk used for downstream generation."""

    parent_chunk_id: str
    document_id: str
    source: str
    source_name: str
    text: str
    heading_path: tuple[str, ...] = ()
    heading_text: str = ""
    parent_order: int = 0
    parent_token_count: int = 0
    original_heading_context: tuple[str, ...] = ()
    part_number: int = 1
    total_parts: int = 1

    def to_record(self) -> dict[str, Any]:
        """Return JSON/SQLite friendly structure for parent lookup by id."""

        return {
            "parent_chunk_id": self.parent_chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "source_name": self.source_name,
            "text": self.text,
            "heading_path": list(self.heading_path),
            "heading_text": self.heading_text,
            "parent_order": self.parent_order,
            "parent_token_count": self.parent_token_count,
            "original_heading_context": list(self.original_heading_context),
            "part_number": self.part_number,
            "total_parts": self.total_parts,
        }


@dataclass(slots=True)
class ChildChunk:
    """A retrieval-focused child chunk linked to exactly one parent chunk."""

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    source: str
    source_name: str
    text: str
    child_order: int
    token_count: int
    heading_path: tuple[str, ...] = ()

    def to_qdrant_record(self) -> dict[str, Any]:
        """Return a Qdrant-ready record with id, text, and payload metadata."""

        payload: dict[str, Any] = {
            "parent_chunk_id": self.parent_chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "source_name": self.source_name,
            "child_order": self.child_order,
            "token_count": self.token_count,
            "heading_path": list(self.heading_path),
            "text": self.text,
        }
        return {"id": self.child_chunk_id, "text": self.text, "payload": payload}


@dataclass(slots=True)
class ChunkingResult:
    """Chunking output with parent records and Qdrant-ready child records."""

    parent_chunks: list[ParentChunk] = field(default_factory=list)
    child_chunks: list[ChildChunk] = field(default_factory=list)

    def parent_lookup(self) -> dict[str, dict[str, Any]]:
        """Return parent chunks keyed by parent_chunk_id for quick lookup."""

        return {parent.parent_chunk_id: parent.to_record() for parent in self.parent_chunks}

    def child_qdrant_records(self) -> list[dict[str, Any]]:
        """Return child chunks in an upsert-friendly format for vector indexes."""

        return [child.to_qdrant_record() for child in self.child_chunks]
