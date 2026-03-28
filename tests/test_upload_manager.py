from __future__ import annotations

from pathlib import Path

from ui.upload_manager import sanitize_filename, save_uploaded_files


class DummyUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_sanitize_filename_blocks_path_traversal() -> None:
    assert sanitize_filename("../../secret contract.pdf") == "secret_contract.pdf"


def test_save_uploaded_files_handles_duplicate_names(tmp_path: Path) -> None:
    files = [
        DummyUploadedFile("contract.txt", b"first copy"),
        DummyUploadedFile("contract.txt", b"second copy"),
    ]

    saved_docs, errors = save_uploaded_files(files, upload_dir=tmp_path)

    assert not errors
    assert len(saved_docs) == 2
    assert saved_docs[0]["path"] != saved_docs[1]["path"]
    assert Path(saved_docs[0]["path"]).name == "contract.txt"
    assert Path(saved_docs[1]["path"]).name == "contract_001.txt"


def test_save_uploaded_files_metadata_shape(tmp_path: Path) -> None:
    saved_docs, errors = save_uploaded_files([DummyUploadedFile("nda.md", b"# NDA")], upload_dir=tmp_path)

    assert not errors
    assert len(saved_docs) == 1

    doc = saved_docs[0]
    assert doc["id"].startswith("uploaded:")
    assert doc["name"] == "nda.md"
    assert doc["type"] == "md"
    assert doc["source"] == "uploaded"
    assert doc["size_bytes"] == 5
