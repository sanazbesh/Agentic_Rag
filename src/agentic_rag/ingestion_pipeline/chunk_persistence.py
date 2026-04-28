"""Persistence service for parent/child chunks tied to a document version."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from agentic_rag.chunking.models import ChildChunk, ChunkingResult, ParentChunk
from agentic_rag.storage.models import Chunk


PARENT_CHUNK_TYPE = "parent"
CHILD_CHUNK_TYPE = "child"


@dataclass(slots=True)
class ChunkPersistenceService:
    """Persist and query chunk rows in Postgres-compatible SQLAlchemy sessions.

    Idempotency behavior:
    - ``persist_chunks`` performs a replace for the target ``document_version_id``.
    - Existing rows for that version are deleted first, then parent/child chunks are inserted.
    - This guarantees re-persisting the same chunk set does not create duplicates.
    """

    session: Session

    def persist_chunks(
        self,
        *,
        document_id: str,
        document_version_id: str,
        chunking_result: ChunkingResult,
    ) -> list[Chunk]:
        """Replace and persist parent + child chunk rows for one document version."""

        self._replace_chunks_for_version(document_version_id=document_version_id)

        persisted_parents = self.persist_parent_chunks(
            document_id=document_id,
            document_version_id=document_version_id,
            parent_chunks=chunking_result.parent_chunks,
        )
        persisted_children = self.persist_child_chunks(
            document_id=document_id,
            document_version_id=document_version_id,
            child_chunks=chunking_result.child_chunks,
        )
        self.session.flush()
        return [*persisted_parents, *persisted_children]

    def persist_parent_chunks(
        self,
        *,
        document_id: str,
        document_version_id: str,
        parent_chunks: list[ParentChunk],
    ) -> list[Chunk]:
        """Persist parent chunks for one document version."""

        rows: list[Chunk] = []
        for parent in parent_chunks:
            section_path = self._section_path_from_heading_path(parent.heading_path)
            row = Chunk(
                id=parent.parent_chunk_id,
                document_id=document_id,
                document_version_id=document_version_id,
                parent_chunk_id=None,
                chunk_type=PARENT_CHUNK_TYPE,
                text=parent.text,
                heading=parent.heading_text or None,
                section_path=section_path,
                metadata_json={
                    "source": parent.source,
                    "source_name": parent.source_name,
                    "heading_path": list(parent.heading_path),
                    "parent_order": parent.parent_order,
                    "parent_token_count": parent.parent_token_count,
                    "original_heading_context": list(parent.original_heading_context),
                    "part_number": parent.part_number,
                    "total_parts": parent.total_parts,
                },
            )
            rows.append(row)
            self.session.add(row)
        return rows

    def persist_child_chunks(
        self,
        *,
        document_id: str,
        document_version_id: str,
        child_chunks: list[ChildChunk],
    ) -> list[Chunk]:
        """Persist child chunks linked to existing parent chunks."""

        rows: list[Chunk] = []
        for child in child_chunks:
            section_path = self._section_path_from_heading_path(child.heading_path)
            row = Chunk(
                id=child.child_chunk_id,
                document_id=document_id,
                document_version_id=document_version_id,
                parent_chunk_id=child.parent_chunk_id,
                chunk_type=CHILD_CHUNK_TYPE,
                text=child.text,
                heading=child.heading_path[-1] if child.heading_path else None,
                section_path=section_path,
                metadata_json={
                    "source": child.source,
                    "source_name": child.source_name,
                    "heading_path": list(child.heading_path),
                    "child_order": child.child_order,
                    "token_count": child.token_count,
                },
            )
            rows.append(row)
            self.session.add(row)
        return rows

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Return one chunk row by primary key."""

        return self.session.get(Chunk, chunk_id)

    def get_chunks_by_document_version_id(self, document_version_id: str) -> list[Chunk]:
        """Return chunks for a version ordered by type and id for deterministic reads."""

        query = (
            select(Chunk)
            .where(Chunk.document_version_id == document_version_id)
            .order_by(Chunk.chunk_type.asc(), Chunk.id.asc())
        )
        return list(self.session.execute(query).scalars().all())

    def get_children_for_parent_chunk(self, parent_chunk_id: str) -> list[Chunk]:
        """Return child chunk rows linked to one parent chunk."""

        query = (
            select(Chunk)
            .where(Chunk.parent_chunk_id == parent_chunk_id)
            .order_by(Chunk.id.asc())
        )
        return list(self.session.execute(query).scalars().all())

    def _replace_chunks_for_version(self, *, document_version_id: str) -> None:
        self.session.execute(delete(Chunk).where(Chunk.document_version_id == document_version_id))
        self.session.flush()

    @staticmethod
    def _section_path_from_heading_path(heading_path: tuple[str, ...]) -> str | None:
        if not heading_path:
            return None
        return " > ".join(heading_path)


__all__ = [
    "CHILD_CHUNK_TYPE",
    "PARENT_CHUNK_TYPE",
    "ChunkPersistenceService",
]
