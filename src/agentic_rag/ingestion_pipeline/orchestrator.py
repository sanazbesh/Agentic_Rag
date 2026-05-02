"""Persistent ingestion orchestrator for registry, file storage, and parsing/chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from agentic_rag.chunking.interfaces import Chunker
from agentic_rag.chunking.markdown import MarkdownParentChildChunker
from agentic_rag.chunking.models import ChunkingResult
from agentic_rag.ingestion.document_ingestors import MarkdownDocumentIngestor, PDFDocumentIngestor
from agentic_rag.ingestion.interfaces import DocumentIngestor
from agentic_rag.ingestion_pipeline.document_registry import DocumentRegistry
from agentic_rag.ingestion_pipeline.chunk_persistence import ChunkPersistenceService
from agentic_rag.storage.document_store import LocalDocumentStore
from agentic_rag.ingestion_pipeline.ingestion_jobs import IngestionJobService
from agentic_rag.ingestion_pipeline.vector_indexing import ChildChunkVectorIndexingService, VectorIndexingResult
from agentic_rag.ingestion_pipeline.validation import IngestionValidationService
from agentic_rag.storage.models import DocumentVersion, IngestionJob, LifecycleStatus
from agentic_rag.types import Document


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class IngestionResult:
    """Structured orchestration result from a single file ingestion request."""

    document_id: str
    document_version_id: str
    job_id: str
    status: LifecycleStatus
    created_document: bool
    created_version: bool
    storage_path: str
    parsed_documents: list[Document] = field(default_factory=list)
    chunking_result: ChunkingResult | None = None
    parent_chunk_count: int = 0
    child_chunk_count: int = 0
    indexed_vector_count: int = 0
    parsed_text_length: int = 0
    parent_chunks_created: int = 0
    child_chunks_created: int = 0
    chunk_persistence_called: bool = False
    parent_chunks_persisted: int = 0
    child_chunks_persisted: int = 0
    vectors_indexed: int = 0
    error_message: str | None = None


class IngestionOrchestrator:
    """Coordinate persistent registration, storage, and in-memory parse/chunk steps."""

    def __init__(
        self,
        *,
        session: Session,
        registry: DocumentRegistry,
        document_store: LocalDocumentStore,
        markdown_ingestor: DocumentIngestor | None = None,
        pdf_ingestor: DocumentIngestor | None = None,
        chunker: Chunker | None = None,
        chunk_persistence_service: ChunkPersistenceService | None = None,
        vector_indexing_service: ChildChunkVectorIndexingService | None = None,
        validation_service: IngestionValidationService | None = None,
    ) -> None:
        self._session = session
        self._registry = registry
        self._document_store = document_store
        self._markdown_ingestor = markdown_ingestor or MarkdownDocumentIngestor()
        self._pdf_ingestor = pdf_ingestor or PDFDocumentIngestor()
        self._chunker = chunker or MarkdownParentChildChunker()
        self._job_service = IngestionJobService(session)
        self._chunk_persistence_service = chunk_persistence_service or ChunkPersistenceService(session=session)
        self._vector_indexing_service = vector_indexing_service
        self._validation_service = validation_service or IngestionValidationService(session)

    def reindex_document(self, document_id: str) -> IngestionResult:
        version = self._registry.get_current_ready_version(document_id)
        if version is None:
            raise ValueError(f"No READY document version found for document: {document_id}")
        return self.reindex_document_version(version.id)

    def reindex_document_version(self, document_version_id: str) -> IngestionResult:
        version = self._session.get(DocumentVersion, document_version_id)
        if version is None:
            raise ValueError(f"Document version not found: {document_version_id}")
        if version.storage_path is None:
            raise ValueError(f"Document version {document_version_id} has no storage_path to reindex")

        document = version.document
        source_path = Path(version.storage_path)
        content_bytes = source_path.read_bytes()

        job = self._job_service.create_job(document_id=document.id, document_version_id=version.id)
        self._job_service.mark_pending(job)
        self._registry.update_document_status(document.id, LifecycleStatus.PROCESSING)
        self._registry.update_version_status(version.id, LifecycleStatus.PROCESSING)
        self._job_service.mark_processing(job)

        parsed_text_length = 0
        parent_chunks_created = 0
        child_chunks_created = 0
        chunk_persistence_called = False
        parent_chunks_persisted = 0
        child_chunks_persisted = 0
        vectors_indexed = 0

        try:
            parsed_documents = self._parse_content(
                content_bytes=content_bytes,
                source_name=document.source_name,
                source_type=document.source_type,
                storage_path=version.storage_path,
            )
            chunking_result = self._chunk_documents(parsed_documents)
            if chunking_result is None:
                raise RuntimeError("No parsed documents were available for chunking")
            parsed_text_length = len("\n".join(document.text for document in parsed_documents).strip())
            parent_chunks_created = len(chunking_result.parent_chunks)
            child_chunks_created = len(chunking_result.child_chunks)
            logger.info("ingestion_chunk_debug parsed_text_length=%s parent_chunks_created=%s child_chunks_created=%s", parsed_text_length, parent_chunks_created, child_chunks_created)

            persisted_chunks = None
            chunk_persistence_called = True
            logger.info("ingestion_chunk_debug calling_persistence=True document_id=%s document_version_id=%s", document.id, version.id)
            persisted_chunks = self._chunk_persistence_service.persist_chunks(
                document_id=document.id,
                document_version_id=version.id,
                chunking_result=chunking_result,
            )
            parent_chunks_persisted = len([chunk for chunk in persisted_chunks if chunk.chunk_type == "parent"])
            child_chunks_persisted = len([chunk for chunk in persisted_chunks if chunk.chunk_type == "child"])
            logger.info("ingestion_chunk_debug chunks_inserted=%s", len(persisted_chunks))
            indexing_result: VectorIndexingResult | None = None
            if self._vector_indexing_service is not None:
                indexing_result = self._vector_indexing_service.index_document_version(document_version_id=version.id)
                vectors_indexed = indexing_result.upserted_child_chunks

            validation_result = self._validation_service.validate(
                document_version=version,
                parsed_documents=parsed_documents,
                chunking_result=chunking_result,
                persisted_chunks=persisted_chunks,
                indexing_result=indexing_result,
            )
            if not validation_result.is_valid:
                raise RuntimeError(validation_result.error_message or "Validation failed")

            version.parser_version = self._ingestor_version_for_source(document.source_name)
            version.chunker_version = self._version_string(self._chunker)
            if self._vector_indexing_service is not None:
                version.embedding_model = self._embedding_model_name()

            self._registry.update_version_status(version.id, LifecycleStatus.READY)
            self._registry.promote_ready_version(document.id, version.id)
            self._registry.update_document_status(document.id, LifecycleStatus.READY)
            self._job_service.mark_ready(job)
            self._session.commit()
            return IngestionResult(
                document_id=document.id,
                document_version_id=version.id,
                job_id=job.id,
                status=LifecycleStatus.READY,
                created_document=False,
                created_version=False,
                storage_path=version.storage_path,
                parsed_documents=parsed_documents,
                chunking_result=chunking_result,
                parent_chunk_count=len(chunking_result.parent_chunks),
                child_chunk_count=len(chunking_result.child_chunks),
                indexed_vector_count=(indexing_result.upserted_child_chunks if indexing_result is not None else 0),
                parsed_text_length=parsed_text_length,
                parent_chunks_created=parent_chunks_created,
                child_chunks_created=child_chunks_created,
                chunk_persistence_called=chunk_persistence_called,
                parent_chunks_persisted=parent_chunks_persisted,
                child_chunks_persisted=child_chunks_persisted,
                vectors_indexed=vectors_indexed,
            )
        except Exception as exc:
            self._registry.update_version_status(version.id, LifecycleStatus.FAILED)
            current_ready = self._registry.get_current_ready_version(document.id)
            next_document_status = LifecycleStatus.READY if current_ready is not None else LifecycleStatus.FAILED
            self._registry.update_document_status(document.id, next_document_status)
            self._job_service.mark_failed(job, error_message=str(exc))
            self._session.commit()
            return IngestionResult(
                document_id=document.id,
                document_version_id=version.id,
                job_id=job.id,
                status=LifecycleStatus.FAILED,
                created_document=False,
                created_version=False,
                storage_path=version.storage_path,
                parsed_text_length=parsed_text_length,
                parent_chunks_created=parent_chunks_created,
                child_chunks_created=child_chunks_created,
                chunk_persistence_called=chunk_persistence_called,
                parent_chunks_persisted=parent_chunks_persisted,
                child_chunks_persisted=child_chunks_persisted,
                vectors_indexed=vectors_indexed,
                error_message=str(exc),
            )

    def ingest_file(
        self,
        file_path: str | Path,
        source_name: str | None = None,
        source_type: str | None = None,
    ) -> IngestionResult:
        """Persist and process one local file path through implemented ingestion stages."""

        path = Path(file_path)
        resolved_source_name = source_name or path.name
        resolved_source_type = source_type or self._resolve_source_type(path)
        content_bytes = path.read_bytes()

        registration = self._registry.register_document(
            source_name=resolved_source_name,
            source_type=resolved_source_type,
            content_bytes=content_bytes,
            status=LifecycleStatus.PENDING,
        )

        storage_path = self._document_store.save_file(
            source_path=path,
            document_id=registration.document.id,
            document_version_id=registration.version.id,
        )
        registration.version.storage_path = storage_path

        job = self._job_service.create_job(
            document_id=registration.document.id,
            document_version_id=registration.version.id,
        )
        self._job_service.mark_pending(job)

        self._registry.update_document_status(registration.document.id, LifecycleStatus.PROCESSING)
        self._registry.update_version_status(registration.version.id, LifecycleStatus.PROCESSING)
        self._job_service.mark_processing(job)

        parsed_text_length = 0
        parent_chunks_created = 0
        child_chunks_created = 0
        chunk_persistence_called = False
        parent_chunks_persisted = 0
        child_chunks_persisted = 0
        vectors_indexed = 0

        try:
            parsed_documents = self._parse_content(
                content_bytes=content_bytes,
                source_name=resolved_source_name,
                source_type=resolved_source_type,
                storage_path=storage_path,
            )
            chunking_result = self._chunk_documents(parsed_documents)
            if chunking_result is None:
                raise RuntimeError("No parsed documents were available for chunking")
            parsed_text_length = len("\n".join(document.text for document in parsed_documents).strip())
            parent_chunks_created = len(chunking_result.parent_chunks)
            child_chunks_created = len(chunking_result.child_chunks)
            logger.info("ingestion_chunk_debug parsed_text_length=%s parent_chunks_created=%s child_chunks_created=%s", parsed_text_length, parent_chunks_created, child_chunks_created)
            persisted_chunks = None
            chunk_persistence_called = True
            logger.info("ingestion_chunk_debug calling_persistence=True document_id=%s document_version_id=%s", registration.document.id, registration.version.id)
            persisted_chunks = self._chunk_persistence_service.persist_chunks(
                document_id=registration.document.id,
                document_version_id=registration.version.id,
                chunking_result=chunking_result,
            )
            parent_chunks_persisted = len([chunk for chunk in persisted_chunks if chunk.chunk_type == "parent"])
            child_chunks_persisted = len([chunk for chunk in persisted_chunks if chunk.chunk_type == "child"])
            logger.info("ingestion_chunk_debug chunks_inserted=%s", len(persisted_chunks))
            indexing_result: VectorIndexingResult | None = None
            if self._vector_indexing_service is not None:
                indexing_result = self._vector_indexing_service.index_document_version(
                    document_version_id=registration.version.id,
                )
                vectors_indexed = indexing_result.upserted_child_chunks

            validation_result = self._validation_service.validate(
                document_version=registration.version,
                parsed_documents=parsed_documents,
                chunking_result=chunking_result,
                persisted_chunks=persisted_chunks,
                indexing_result=indexing_result,
            )
            if not validation_result.is_valid:
                raise RuntimeError(validation_result.error_message or "Validation failed")

            self._registry.update_version_status(registration.version.id, LifecycleStatus.READY)
            self._registry.promote_ready_version(registration.document.id, registration.version.id)
            self._registry.update_document_status(registration.document.id, LifecycleStatus.READY)
            self._job_service.mark_ready(job)
            self._session.commit()
            return IngestionResult(
                document_id=registration.document.id,
                document_version_id=registration.version.id,
                job_id=job.id,
                status=LifecycleStatus.READY,
                created_document=registration.created_document,
                created_version=registration.created_version,
                storage_path=storage_path,
                parsed_documents=parsed_documents,
                chunking_result=chunking_result,
                parent_chunk_count=len(chunking_result.parent_chunks),
                child_chunk_count=len(chunking_result.child_chunks),
                indexed_vector_count=(indexing_result.upserted_child_chunks if indexing_result is not None else 0),
                parsed_text_length=parsed_text_length,
                parent_chunks_created=parent_chunks_created,
                child_chunks_created=child_chunks_created,
                chunk_persistence_called=chunk_persistence_called,
                parent_chunks_persisted=parent_chunks_persisted,
                child_chunks_persisted=child_chunks_persisted,
                vectors_indexed=vectors_indexed,
            )
        except Exception as exc:
            self._registry.update_version_status(registration.version.id, LifecycleStatus.FAILED)
            current_ready = self._registry.get_current_ready_version(registration.document.id)
            next_document_status = LifecycleStatus.READY if current_ready is not None else LifecycleStatus.FAILED
            self._registry.update_document_status(registration.document.id, next_document_status)
            self._job_service.mark_failed(job, error_message=str(exc))
            self._session.commit()
            return IngestionResult(
                document_id=registration.document.id,
                document_version_id=registration.version.id,
                job_id=job.id,
                status=LifecycleStatus.FAILED,
                created_document=registration.created_document,
                created_version=registration.created_version,
                storage_path=storage_path,
                parsed_text_length=parsed_text_length,
                parent_chunks_created=parent_chunks_created,
                child_chunks_created=child_chunks_created,
                chunk_persistence_called=chunk_persistence_called,
                parent_chunks_persisted=parent_chunks_persisted,
                child_chunks_persisted=child_chunks_persisted,
                vectors_indexed=vectors_indexed,
                error_message=str(exc),
            )

    def retry_failed_job(self, job_id: str) -> IngestionResult:
        job = self._job_service.get_job(job_id)
        if job is None:
            raise ValueError(f"Ingestion job not found: {job_id}")
        return self._retry_failed_job(job=job)

    def retry_failed_document_version(self, document_version_id: str) -> IngestionResult:
        job = self._job_service.get_latest_job_for_document_version(document_version_id)
        if job is None:
            raise ValueError(f"No ingestion job found for document version: {document_version_id}")
        return self._retry_failed_job(job=job)

    def _retry_failed_job(self, *, job: IngestionJob) -> IngestionResult:
        if job.status != LifecycleStatus.FAILED:
            raise ValueError(f"Retry is only allowed for FAILED jobs. job_id={job.id} status={job.status.value}")

        document = job.document
        version = job.document_version
        storage_path = version.storage_path
        if storage_path is None:
            raise ValueError(f"Document version {version.id} has no storage_path to retry")

        source_path = Path(storage_path)
        content_bytes = source_path.read_bytes()
        source_name = document.source_name
        source_type = document.source_type

        self._job_service.mark_pending(job)
        self._registry.update_document_status(document.id, LifecycleStatus.PENDING)
        self._registry.update_version_status(version.id, LifecycleStatus.PENDING)

        self._registry.update_document_status(document.id, LifecycleStatus.PROCESSING)
        self._registry.update_version_status(version.id, LifecycleStatus.PROCESSING)
        self._job_service.mark_processing(job)

        parsed_text_length = 0
        parent_chunks_created = 0
        child_chunks_created = 0
        chunk_persistence_called = False
        parent_chunks_persisted = 0
        child_chunks_persisted = 0
        vectors_indexed = 0

        try:
            parsed_documents = self._parse_content(
                content_bytes=content_bytes,
                source_name=source_name,
                source_type=source_type,
                storage_path=storage_path,
            )
            chunking_result = self._chunk_documents(parsed_documents)
            if chunking_result is None:
                raise RuntimeError("No parsed documents were available for chunking")
            parsed_text_length = len("\n".join(document.text for document in parsed_documents).strip())
            parent_chunks_created = len(chunking_result.parent_chunks)
            child_chunks_created = len(chunking_result.child_chunks)
            logger.info("ingestion_chunk_debug parsed_text_length=%s parent_chunks_created=%s child_chunks_created=%s", parsed_text_length, parent_chunks_created, child_chunks_created)
            persisted_chunks = None
            chunk_persistence_called = True
            logger.info("ingestion_chunk_debug calling_persistence=True document_id=%s document_version_id=%s", document.id, version.id)
            persisted_chunks = self._chunk_persistence_service.persist_chunks(
                document_id=document.id,
                document_version_id=version.id,
                chunking_result=chunking_result,
            )
            parent_chunks_persisted = len([chunk for chunk in persisted_chunks if chunk.chunk_type == "parent"])
            child_chunks_persisted = len([chunk for chunk in persisted_chunks if chunk.chunk_type == "child"])
            logger.info("ingestion_chunk_debug chunks_inserted=%s", len(persisted_chunks))
            indexing_result: VectorIndexingResult | None = None
            if self._vector_indexing_service is not None:
                indexing_result = self._vector_indexing_service.index_document_version(document_version_id=version.id)
                vectors_indexed = indexing_result.upserted_child_chunks

            validation_result = self._validation_service.validate(
                document_version=version,
                parsed_documents=parsed_documents,
                chunking_result=chunking_result,
                persisted_chunks=persisted_chunks,
                indexing_result=indexing_result,
            )
            if not validation_result.is_valid:
                raise RuntimeError(validation_result.error_message or "Validation failed")
            self._registry.update_version_status(version.id, LifecycleStatus.READY)
            self._registry.promote_ready_version(document.id, version.id)
            self._registry.update_document_status(document.id, LifecycleStatus.READY)
            self._job_service.mark_ready(job)
            self._session.commit()
            return IngestionResult(
                document_id=document.id,
                document_version_id=version.id,
                job_id=job.id,
                status=LifecycleStatus.READY,
                created_document=False,
                created_version=False,
                storage_path=storage_path,
                parsed_documents=parsed_documents,
                chunking_result=chunking_result,
                parent_chunk_count=len(chunking_result.parent_chunks),
                child_chunk_count=len(chunking_result.child_chunks),
                indexed_vector_count=(indexing_result.upserted_child_chunks if indexing_result is not None else 0),
                parsed_text_length=parsed_text_length,
                parent_chunks_created=parent_chunks_created,
                child_chunks_created=child_chunks_created,
                chunk_persistence_called=chunk_persistence_called,
                parent_chunks_persisted=parent_chunks_persisted,
                child_chunks_persisted=child_chunks_persisted,
                vectors_indexed=vectors_indexed,
            )
        except Exception as exc:
            self._registry.update_version_status(version.id, LifecycleStatus.FAILED)
            current_ready = self._registry.get_current_ready_version(document.id)
            next_document_status = LifecycleStatus.READY if current_ready is not None else LifecycleStatus.FAILED
            self._registry.update_document_status(document.id, next_document_status)
            self._job_service.mark_failed(job, error_message=str(exc))
            self._session.commit()
            return IngestionResult(
                document_id=document.id,
                document_version_id=version.id,
                job_id=job.id,
                status=LifecycleStatus.FAILED,
                created_document=False,
                created_version=False,
                storage_path=storage_path,
                parsed_text_length=parsed_text_length,
                parent_chunks_created=parent_chunks_created,
                child_chunks_created=child_chunks_created,
                chunk_persistence_called=chunk_persistence_called,
                parent_chunks_persisted=parent_chunks_persisted,
                child_chunks_persisted=child_chunks_persisted,
                vectors_indexed=vectors_indexed,
                error_message=str(exc),
            )

    def _parse_content(
        self,
        *,
        content_bytes: bytes,
        source_name: str,
        source_type: str,
        storage_path: str,
    ) -> list[Document]:
        ingestor = self._resolve_ingestor(source_name)
        record: dict[str, object] = {
            "source": storage_path,
            "source_name": source_name,
            "source_type": source_type,
        }
        if self._is_pdf_source(source_name):
            record["content"] = content_bytes
        else:
            record["text"] = content_bytes.decode("utf-8")
        return ingestor.ingest([record])

    def _chunk_documents(self, parsed_documents: list[Document]) -> ChunkingResult | None:
        if not parsed_documents:
            return None
        return self._chunker.chunk(parsed_documents[0])

    def _resolve_ingestor(self, source_name: str) -> DocumentIngestor:
        if self._is_pdf_source(source_name):
            return self._pdf_ingestor
        return self._markdown_ingestor

    @staticmethod
    def _is_pdf_source(source_name: str) -> bool:
        return Path(source_name).suffix.lower() == ".pdf"

    def _resolve_source_type(self, path: Path) -> str:
        if self._is_pdf_source(path.name):
            return "pdf"
        suffix = path.suffix.lower().lstrip(".")
        return suffix or "file"

    def _version_string(self, obj: object) -> str:
        return obj.__class__.__name__

    def _ingestor_version_for_source(self, source_name: str) -> str:
        return self._version_string(self._resolve_ingestor(source_name))

    def _embedding_model_name(self) -> str:
        service = self._vector_indexing_service
        if service is None:
            return ""
        return service.embedding_service.config.model_name


__all__ = ["IngestionOrchestrator", "IngestionResult"]
