"""Persistent document/version registry for ingestion lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from agentic_rag.storage.models import Document, DocumentVersion, LifecycleStatus


DOC_ID_PREFIX = "doc_"
VERSION_ID_PREFIX = "ver_"


def compute_sha256_from_bytes(content: bytes) -> str:
    """Return a stable SHA-256 hash digest for in-memory content bytes."""

    return hashlib.sha256(content).hexdigest()


def compute_sha256_from_file_path(file_path: str | Path) -> str:
    """Return a stable SHA-256 hash digest for file content."""

    path = Path(file_path)
    return compute_sha256_from_bytes(path.read_bytes())


@dataclass(slots=True, frozen=True)
class DocumentRegistrationResult:
    """Result object describing a registration operation outcome."""

    document: Document
    version: DocumentVersion
    created_document: bool
    created_version: bool


class DocumentRegistry:
    """Service for idempotent registration and lookup of document versions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def register_document(
        self,
        *,
        source_name: str,
        source_type: str,
        content_hash: str | None = None,
        content_bytes: bytes | None = None,
        storage_path: str | None = None,
        parser_version: str | None = None,
        chunker_version: str | None = None,
        embedding_model: str | None = None,
        status: LifecycleStatus = LifecycleStatus.PENDING,
    ) -> DocumentRegistrationResult:
        """Register a document and version idempotently by content hash."""

        resolved_content_hash = self._resolve_content_hash(
            content_hash=content_hash,
            content_bytes=content_bytes,
        )

        document = self._get_document_by_source(source_name=source_name, source_type=source_type)
        created_document = False
        if document is None:
            document = Document(
                id=self._new_document_id(),
                source_name=source_name,
                source_type=source_type,
                status=status,
            )
            self._session.add(document)
            self._session.flush()
            created_document = True

        existing_version = self._find_version_for_document_hash(
            document_id=document.id,
            content_hash=resolved_content_hash,
        )
        if existing_version is not None:
            return DocumentRegistrationResult(
                document=document,
                version=existing_version,
                created_document=created_document,
                created_version=False,
            )

        new_version = DocumentVersion(
            id=self._new_version_id(),
            document_id=document.id,
            content_hash=resolved_content_hash,
            storage_path=storage_path,
            parser_version=parser_version,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            status=status,
        )
        self._session.add(new_version)
        self._session.flush()

        return DocumentRegistrationResult(
            document=document,
            version=new_version,
            created_document=created_document,
            created_version=True,
        )

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Load a document by primary key."""

        return self._session.get(Document, document_id)

    def get_version_by_id(self, version_id: str) -> DocumentVersion | None:
        """Load a document version by primary key."""

        return self._session.get(DocumentVersion, version_id)

    def find_version_by_content_hash(
        self,
        content_hash: str,
        *,
        document_id: str | None = None,
    ) -> DocumentVersion | None:
        """Find the most recent version matching a content hash."""

        query = select(DocumentVersion).where(DocumentVersion.content_hash == content_hash)
        if document_id is not None:
            query = query.where(DocumentVersion.document_id == document_id)

        query = query.order_by(DocumentVersion.created_at.desc(), DocumentVersion.id.desc())
        return self._session.execute(query).scalar_one_or_none()

    def update_document_status(self, document_id: str, status: LifecycleStatus) -> Document:
        """Update and persist a document lifecycle status."""

        document = self.get_document_by_id(document_id)
        if document is None:
            raise ValueError(f"document_not_found: {document_id}")

        document.status = status
        self._session.flush()
        return document

    def update_version_status(self, version_id: str, status: LifecycleStatus) -> DocumentVersion:
        """Update and persist a document version lifecycle status."""

        version = self.get_version_by_id(version_id)
        if version is None:
            raise ValueError(f"document_version_not_found: {version_id}")

        version.status = status
        self._session.flush()
        return version

    def list_versions_for_document(self, document_id: str) -> list[DocumentVersion]:
        """List document versions newest-first by create timestamp."""

        query = (
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.created_at.desc(), DocumentVersion.id.desc())
        )
        return list(self._session.execute(query).scalars().all())

    def get_current_ready_version(self, document_id: str) -> DocumentVersion | None:
        """Return current version when it exists and is READY."""

        document = self.get_document_by_id(document_id)
        if document is None or document.current_version_id is None:
            return None
        version = self.get_version_by_id(document.current_version_id)
        if version is None or version.status != LifecycleStatus.READY:
            return None
        return version

    def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        """Return latest version regardless of lifecycle status."""

        versions = self.list_versions_for_document(document_id)
        return versions[0] if versions else None

    def promote_ready_version(self, document_id: str, version_id: str) -> DocumentVersion:
        """Promote a READY version to document.current_version_id."""

        document = self.get_document_by_id(document_id)
        if document is None:
            raise ValueError(f"document_not_found: {document_id}")

        version = self.get_version_by_id(version_id)
        if version is None or version.document_id != document_id:
            raise ValueError(f"document_version_not_found_for_document: {version_id}")
        if version.status != LifecycleStatus.READY:
            raise ValueError(f"document_version_not_ready: {version_id}")

        document.current_version_id = version.id
        self._session.flush()
        return version

    def _get_document_by_source(self, *, source_name: str, source_type: str) -> Document | None:
        query = (
            select(Document)
            .where(Document.source_name == source_name, Document.source_type == source_type)
            .order_by(Document.created_at.asc(), Document.id.asc())
        )
        return self._session.execute(query).scalar_one_or_none()

    def _find_version_for_document_hash(self, *, document_id: str, content_hash: str) -> DocumentVersion | None:
        query = (
            select(DocumentVersion)
            .where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.content_hash == content_hash,
            )
            .order_by(DocumentVersion.created_at.asc(), DocumentVersion.id.asc())
        )
        return self._session.execute(query).scalar_one_or_none()

    @staticmethod
    def _new_document_id() -> str:
        return f"{DOC_ID_PREFIX}{uuid4().hex}"

    @staticmethod
    def _new_version_id() -> str:
        return f"{VERSION_ID_PREFIX}{uuid4().hex}"

    @staticmethod
    def _resolve_content_hash(*, content_hash: str | None, content_bytes: bytes | None) -> str:
        if content_hash:
            return content_hash
        if content_bytes is not None:
            return compute_sha256_from_bytes(content_bytes)
        raise ValueError("content_hash_or_content_bytes_required")
