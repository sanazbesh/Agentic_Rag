from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.storage.document_store import LocalDocumentStore, document_storage_config_from_env


def test_document_storage_config_from_env_uses_document_storage_path() -> None:
    config = document_storage_config_from_env({"DOCUMENT_STORAGE_PATH": "./tmp/doc-store"})

    assert config.root_path == Path("./tmp/doc-store")


def test_save_file_from_path_persists_file(tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")
    source = tmp_path / "uploads" / "contract.PDF"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"contract-content")

    storage_path = store.save_file(
        source_path=source,
        document_id="doc_123",
        document_version_id="ver_001",
    )

    assert storage_path == "doc_123/ver_001.pdf"
    assert store.exists(storage_path) is True


def test_save_bytes_persists_and_reads_back(tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")

    storage_path = store.save_bytes(
        content=b"memo-content",
        document_id="doc_789",
        document_version_id="ver_010",
        source_name="memo.md",
    )

    assert storage_path == "doc_789/ver_010.md"
    assert store.read_bytes(storage_path) == b"memo-content"


def test_storage_path_is_deterministic_and_includes_ids(tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")

    first = store.build_storage_path(
        document_id="doc_abc",
        document_version_id="ver_xyz",
        source_name="report.txt",
    )
    second = store.build_storage_path(
        document_id="doc_abc",
        document_version_id="ver_xyz",
        source_name="report.txt",
    )

    assert first == "doc_abc/ver_xyz.txt"
    assert second == first


def test_unsafe_filename_extension_is_safely_normalized(tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")

    storage_path = store.save_bytes(
        content=b"payload",
        document_id="doc_safe",
        document_version_id="ver_safe",
        source_name="../../secret.exe;rm -rf",
    )

    assert storage_path == "doc_safe/ver_safe"
    assert store.read_bytes(storage_path) == b"payload"


def test_exists_false_and_clear_error_for_missing_or_unsafe_path(tmp_path) -> None:
    store = LocalDocumentStore(tmp_path / "documents")

    assert store.exists("doc_missing/ver_missing.pdf") is False
    assert store.exists("../outside.txt") is False

    with pytest.raises(ValueError, match="invalid_storage_path"):
        store.read_bytes("../outside.txt")
