"""Safe deletion service for persisted documents and related ingestion artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_rag.indexing.dense_child_chunks import QdrantChildChunkStore
from agentic_rag.storage.document_store import LocalDocumentStore
from agentic_rag.storage.models import Chunk, Document, DocumentVersion


class DocumentDeletionError(RuntimeError):
    """Raised when document deletion cleanup cannot complete safely."""


@dataclass(slots=True, frozen=True)
class DocumentDeletionResult:
    document_id: str
    deleted: bool
    deleted_qdrant_points: tuple[str, ...] = ()
    deleted_storage_paths: tuple[str, ...] = ()


@dataclass(slots=True)
class DocumentDeletionService:
    session: Session
    document_store: LocalDocumentStore
    qdrant_store: QdrantChildChunkStore | None = None

    def delete_document(self, *, document_id: str) -> DocumentDeletionResult:
        document = self.session.get(Document, document_id)
        if document is None:
            return DocumentDeletionResult(document_id=document_id, deleted=False)

        storage_paths, qdrant_point_ids = self._collect_cleanup_targets(document_id=document_id)

        if self.qdrant_store is not None and qdrant_point_ids:
            try:
                self.qdrant_store.delete_points(qdrant_point_ids)
            except Exception as exc:
                raise DocumentDeletionError(
                    f"qdrant_cleanup_failed for document_id={document_id}: {type(exc).__name__}: {exc}"
                ) from exc

        try:
            self._delete_storage_paths(storage_paths)
        except Exception as exc:
            raise DocumentDeletionError(
                f"storage_cleanup_failed for document_id={document_id}: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            self.session.delete(document)
            self.session.flush()
        except Exception as exc:
            raise DocumentDeletionError(
                f"database_cleanup_failed for document_id={document_id}: {type(exc).__name__}: {exc}"
            ) from exc

        return DocumentDeletionResult(
            document_id=document_id,
            deleted=True,
            deleted_qdrant_points=qdrant_point_ids,
            deleted_storage_paths=storage_paths,
        )

    def _collect_cleanup_targets(self, *, document_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        version_paths = self.session.execute(
            select(DocumentVersion.storage_path).where(DocumentVersion.document_id == document_id)
        ).scalars()
        storage_paths = tuple(sorted({path for path in version_paths if path}))

        qdrant_ids = self.session.execute(
            select(Chunk.qdrant_point_id).where(Chunk.document_id == document_id)
        ).scalars()
        qdrant_point_ids = tuple(sorted({point_id for point_id in qdrant_ids if point_id}))
        return (storage_paths, qdrant_point_ids)

    def _delete_storage_paths(self, storage_paths: tuple[str, ...]) -> None:
        for storage_path in storage_paths:
            if not self.document_store.exists(storage_path):
                continue
            resolved = self.document_store.root_path / Path(storage_path)
            resolved.unlink()

        for storage_path in storage_paths:
            top_level = storage_path.split("/", 1)[0]
            directory = self.document_store.root_path / top_level
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)


__all__ = ["DocumentDeletionService", "DocumentDeletionResult", "DocumentDeletionError"]
