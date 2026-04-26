from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
configure_mappers = pytest.importorskip("sqlalchemy.orm").configure_mappers

from agentic_rag.storage.models import (
    Base,
    Chunk,
    Document,
    DocumentVersion,
    IngestionJob,
    LifecycleStatus,
)


def test_models_import_and_status_values() -> None:
    assert Document.__tablename__ == "documents"
    assert DocumentVersion.__tablename__ == "document_versions"
    assert Chunk.__tablename__ == "chunks"
    assert IngestionJob.__tablename__ == "ingestion_jobs"

    assert LifecycleStatus.PENDING.value == "PENDING"
    assert LifecycleStatus.PROCESSING.value == "PROCESSING"
    assert LifecycleStatus.READY.value == "READY"
    assert LifecycleStatus.FAILED.value == "FAILED"
    assert LifecycleStatus.SKIPPED_DUPLICATE.value == "SKIPPED_DUPLICATE"


def test_expected_tables_exist_in_metadata() -> None:
    table_names = set(Base.metadata.tables)

    assert {"documents", "document_versions", "chunks", "ingestion_jobs"}.issubset(table_names)


def test_required_columns_exist() -> None:
    documents = Base.metadata.tables["documents"].columns
    assert {
        "id",
        "source_name",
        "source_type",
        "current_version_id",
        "status",
        "created_at",
        "updated_at",
    }.issubset(set(documents.keys()))

    document_versions = Base.metadata.tables["document_versions"].columns
    assert {
        "id",
        "document_id",
        "content_hash",
        "storage_path",
        "parser_version",
        "chunker_version",
        "embedding_model",
        "status",
        "created_at",
        "updated_at",
    }.issubset(set(document_versions.keys()))

    chunks = Base.metadata.tables["chunks"].columns
    assert {
        "id",
        "document_id",
        "document_version_id",
        "parent_chunk_id",
        "chunk_type",
        "text",
        "heading",
        "section_path",
        "qdrant_point_id",
        "metadata_json",
        "created_at",
        "updated_at",
    }.issubset(set(chunks.keys()))

    ingestion_jobs = Base.metadata.tables["ingestion_jobs"].columns
    assert {
        "id",
        "document_id",
        "document_version_id",
        "status",
        "error_message",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    }.issubset(set(ingestion_jobs.keys()))


def test_relationships_are_configured() -> None:
    configure_mappers()

    assert "versions" in Document.__mapper__.relationships
    assert "chunks" in Document.__mapper__.relationships
    assert "ingestion_jobs" in Document.__mapper__.relationships

    assert "chunks" in DocumentVersion.__mapper__.relationships
    assert "ingestion_jobs" in DocumentVersion.__mapper__.relationships

    assert "parent_chunk" in Chunk.__mapper__.relationships
    assert "child_chunks" in Chunk.__mapper__.relationships


def test_schema_can_be_created_in_test_database() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)

    Base.metadata.create_all(engine)

    with engine.connect() as connection:
        table_rows = connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    created_tables = {row[0] for row in table_rows}
    assert {"documents", "document_versions", "chunks", "ingestion_jobs"}.issubset(created_tables)
