"""Validation checks that gate READY promotion for ingestion/re-index flows."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_rag.chunking.models import ChunkingResult
from agentic_rag.ingestion_pipeline.chunk_persistence import CHILD_CHUNK_TYPE, PARENT_CHUNK_TYPE
from agentic_rag.ingestion_pipeline.vector_indexing import VectorIndexingResult
from agentic_rag.storage.models import Chunk, DocumentVersion
from agentic_rag.types import Document


@dataclass(slots=True, frozen=True)
class ValidationResult:
    is_valid: bool
    error_message: str | None = None


class IngestionValidationService:
    """Run deterministic quality gates prior to READY status transitions."""
    def __init__(self, session: Session) -> None:
        self._session = session

    def validate(
        self,
        *,
        document_version: DocumentVersion,
        parsed_documents: list[Document],
        chunking_result: ChunkingResult | None,
        persisted_chunks: list[Chunk] | None,
        indexing_result: VectorIndexingResult | None,
    ) -> ValidationResult:
        parsed_text = "\n".join(document.text for document in parsed_documents).strip()
        if not parsed_text:
            return ValidationResult(is_valid=False, error_message="Validation failed: parsed text is empty")

        if not document_version.storage_path:
            return ValidationResult(is_valid=False, error_message="Validation failed: document version is missing storage_path")

        if chunking_result is None:
            return ValidationResult(is_valid=False, error_message="Validation failed: no chunking result was produced")

        if not chunking_result.parent_chunks:
            return ValidationResult(is_valid=False, error_message="Validation failed: no parent chunks were produced")

        if not chunking_result.child_chunks:
            return ValidationResult(is_valid=False, error_message="Validation failed: no child chunks were produced")

        invalid_child = next((chunk for chunk in chunking_result.child_chunks if not chunk.text.strip()), None)
        if invalid_child is not None:
            return ValidationResult(is_valid=False, error_message=f"Validation failed: child chunk {invalid_child.child_chunk_id} has empty text")

        persisted = list(persisted_chunks) if persisted_chunks is not None else self._persisted_chunks_for_version(document_version.id)
        if not persisted:
            return ValidationResult(is_valid=False, error_message="Validation failed: no persisted chunks found")

        persisted_children = [chunk for chunk in persisted if chunk.chunk_type == CHILD_CHUNK_TYPE]
        persisted_parents = [chunk for chunk in persisted if chunk.chunk_type == PARENT_CHUNK_TYPE]
        if not persisted_parents:
            return ValidationResult(is_valid=False, error_message="Validation failed: no persisted parent chunks found")
        if not persisted_children:
            return ValidationResult(is_valid=False, error_message="Validation failed: no persisted child chunks found")

        empty_child = next((chunk for chunk in persisted_children if not chunk.text.strip()), None)
        if empty_child is not None:
            return ValidationResult(is_valid=False, error_message=f"Validation failed: persisted child chunk {empty_child.id} has empty text")

        missing_point = next((chunk for chunk in persisted_children if not chunk.qdrant_point_id), None)
        if missing_point is not None:
            return ValidationResult(is_valid=False, error_message=f"Validation failed: child chunk {missing_point.id} is missing qdrant_point_id")

        if indexing_result is None:
            return ValidationResult(is_valid=False, error_message="Validation failed: vector indexing did not run")

        expected_indexed = len(persisted_children)
        if indexing_result.upserted_child_chunks != expected_indexed:
            return ValidationResult(
                is_valid=False,
                error_message=(
                    "Validation failed: vector upsert count mismatch "
                    f"(upserted={indexing_result.upserted_child_chunks}, expected={expected_indexed})"
                ),
            )

        return ValidationResult(is_valid=True)

    def _persisted_chunks_for_version(self, document_version_id: str) -> list[Chunk]:
        query = (
            select(Chunk)
            .where(Chunk.document_version_id == document_version_id)
            .order_by(Chunk.chunk_type.asc(), Chunk.id.asc())
        )
        return list(self._session.execute(query).scalars().all())


__all__ = ["IngestionValidationService", "ValidationResult"]
