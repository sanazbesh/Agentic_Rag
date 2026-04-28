"""Local file-system storage for raw uploaded documents."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


_EXTENSION_PATTERN = re.compile(r"^[a-z0-9]{1,16}$")


@dataclass(slots=True, frozen=True)
class DocumentStorageConfig:
    """Environment-backed configuration for local document storage."""

    root_path: Path



def document_storage_config_from_env(env: dict[str, str] | None = None) -> DocumentStorageConfig:
    """Load local document storage configuration from environment variables."""

    values = env if env is not None else dict(os.environ)
    configured_path = str(values.get("DOCUMENT_STORAGE_PATH", "./.agentic_rag/documents")).strip()
    return DocumentStorageConfig(root_path=Path(configured_path))


class LocalDocumentStore:
    """Store and retrieve raw source files in deterministic local paths."""

    def __init__(self, root_path: str | Path) -> None:
        self._root_path = Path(root_path)
        self._root_path.mkdir(parents=True, exist_ok=True)

    @property
    def root_path(self) -> Path:
        """Configured root storage directory."""

        return self._root_path

    def build_storage_path(
        self,
        *,
        document_id: str,
        document_version_id: str,
        source_name: str | None = None,
        file_extension: str | None = None,
    ) -> str:
        """Build a deterministic relative storage path for one document version."""

        suffix = self._resolve_extension(source_name=source_name, file_extension=file_extension)
        return f"{document_id}/{document_version_id}{suffix}"

    def save_file(
        self,
        *,
        source_path: str | Path,
        document_id: str,
        document_version_id: str,
    ) -> str:
        """Copy a source file into permanent storage and return the relative storage path."""

        source = Path(source_path)
        content = source.read_bytes()
        storage_path = self.build_storage_path(
            document_id=document_id,
            document_version_id=document_version_id,
            source_name=source.name,
        )
        return self._save_bytes_to_storage_path(content=content, storage_path=storage_path)

    def save_bytes(
        self,
        *,
        content: bytes,
        document_id: str,
        document_version_id: str,
        source_name: str | None = None,
        file_extension: str | None = None,
    ) -> str:
        """Persist in-memory bytes and return the relative storage path."""

        storage_path = self.build_storage_path(
            document_id=document_id,
            document_version_id=document_version_id,
            source_name=source_name,
            file_extension=file_extension,
        )
        return self._save_bytes_to_storage_path(content=content, storage_path=storage_path)

    def read_bytes(self, storage_path: str) -> bytes:
        """Read stored file bytes by relative storage path."""

        path = self._resolve_storage_path(storage_path)
        return path.read_bytes()

    def exists(self, storage_path: str) -> bool:
        """Return whether a file exists for the given storage path."""

        try:
            path = self._resolve_storage_path(storage_path)
        except ValueError:
            return False
        return path.exists()

    def _save_bytes_to_storage_path(self, *, content: bytes, storage_path: str) -> str:
        destination = self._resolve_storage_path(storage_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return storage_path

    def _resolve_storage_path(self, storage_path: str) -> Path:
        candidate = Path(storage_path)
        if candidate.is_absolute():
            raise ValueError(f"invalid_storage_path: {storage_path}")

        destination = (self._root_path / candidate).resolve()
        root = self._root_path.resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"invalid_storage_path: {storage_path}")
        return destination

    @staticmethod
    def _resolve_extension(*, source_name: str | None, file_extension: str | None) -> str:
        if file_extension:
            return LocalDocumentStore._normalize_extension(file_extension)
        if source_name:
            return LocalDocumentStore._normalize_extension(Path(source_name).suffix)
        return ""

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        normalized = extension.strip().lower()
        if not normalized:
            return ""
        if not normalized.startswith("."):
            normalized = f".{normalized}"

        suffix_value = normalized[1:]
        if not _EXTENSION_PATTERN.fullmatch(suffix_value):
            return ""
        return f".{suffix_value}"
