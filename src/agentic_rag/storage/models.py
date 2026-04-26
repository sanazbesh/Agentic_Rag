"""Canonical SQLAlchemy ORM models for persistent RAG schema."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class LifecycleStatus(str, Enum):
    """Canonical lifecycle statuses for documents, versions, and jobs."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    SKIPPED_DUPLICATE = "SKIPPED_DUPLICATE"


class Base(DeclarativeBase):
    """Base declarative class for Agentic RAG storage models."""


_LIFECYCLE_ENUM = Enum(
    LifecycleStatus,
    name="lifecycle_status",
    native_enum=False,
    validate_strings=True,
)


class Document(Base):
    """Canonical source document tracked through ingestion lifecycle."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(String(128), nullable=False)
    current_version_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[LifecycleStatus] = mapped_column(_LIFECYCLE_ENUM, nullable=False, default=LifecycleStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    versions: Mapped[list[DocumentVersion]] = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="DocumentVersion.document_id",
    )
    current_version: Mapped[DocumentVersion | None] = relationship(
        "DocumentVersion",
        foreign_keys=[current_version_id],
        uselist=False,
        post_update=True,
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Chunk.document_id",
    )
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        "IngestionJob",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="IngestionJob.document_id",
    )

    __table_args__ = (Index("ix_documents_status", "status"),)


class DocumentVersion(Base):
    """Immutable revision of source document content and processing metadata."""

    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunker_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[LifecycleStatus] = mapped_column(_LIFECYCLE_ENUM, nullable=False, default=LifecycleStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="versions",
        foreign_keys=[document_id],
    )
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="document_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="Chunk.document_version_id",
    )
    ingestion_jobs: Mapped[list[IngestionJob]] = relationship(
        "IngestionJob",
        back_populates="document_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        foreign_keys="IngestionJob.document_version_id",
    )

    __table_args__ = (
        Index("ix_document_versions_content_hash", "content_hash"),
        Index("ix_document_versions_document_id", "document_id"),
        Index("ix_document_versions_status", "status"),
    )


class Chunk(Base):
    """Persisted parent/child chunk rows connected to document versions."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(128),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    chunk_type: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    section_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    qdrant_point_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="chunks",
        foreign_keys=[document_id],
    )
    document_version: Mapped[DocumentVersion] = relationship(
        "DocumentVersion",
        back_populates="chunks",
        foreign_keys=[document_version_id],
    )
    parent_chunk: Mapped[Chunk | None] = relationship(
        "Chunk",
        remote_side=[id],
        back_populates="child_chunks",
        foreign_keys=[parent_chunk_id],
    )
    child_chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="parent_chunk",
        cascade="all",
    )

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_document_version_id", "document_version_id"),
        Index("ix_chunks_qdrant_point_id", "qdrant_point_id"),
    )


class IngestionJob(Base):
    """Ingestion processing attempt for a document version."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[LifecycleStatus] = mapped_column(_LIFECYCLE_ENUM, nullable=False, default=LifecycleStatus.PENDING)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="ingestion_jobs",
        foreign_keys=[document_id],
    )
    document_version: Mapped[DocumentVersion] = relationship(
        "DocumentVersion",
        back_populates="ingestion_jobs",
        foreign_keys=[document_version_id],
    )

    __table_args__ = (
        Index("ix_ingestion_jobs_document_id", "document_id"),
        Index("ix_ingestion_jobs_document_version_id", "document_version_id"),
        Index("ix_ingestion_jobs_status", "status"),
    )


__all__ = [
    "Base",
    "LifecycleStatus",
    "Document",
    "DocumentVersion",
    "Chunk",
    "IngestionJob",
]
