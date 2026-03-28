"""Helpers for local-first Streamlit document uploads.

Uploaded files are stored under ``data/uploads`` (repo-local) so local test runs
can persist documents across Streamlit reruns without external services.

This module intentionally avoids ingestion/indexing. Future backend integration can
hook into uploaded metadata via ``register_uploaded_documents``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

UPLOAD_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {"pdf", "md", "txt"}


class UploadError(RuntimeError):
    """Raised when uploaded file handling fails unexpectedly."""


def ensure_upload_dir(upload_dir: Path = UPLOAD_DIR) -> Path:
    """Create and return the app-controlled upload directory."""

    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def sanitize_filename(raw_name: str) -> str:
    """Return a filesystem-safe filename.

    - Removes any directory components to block path traversal.
    - Replaces non ``[A-Za-z0-9._-]`` characters with underscores.
    - Prevents hidden/empty names by falling back to ``uploaded_file``.
    """

    base_name = Path(raw_name).name
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_name).strip("._")
    return safe_name or "uploaded_file"


def is_allowed_extension(filename: str) -> bool:
    """Return True when file extension is explicitly allowed."""

    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix in ALLOWED_EXTENSIONS


def _resolve_collision_path(upload_dir: Path, filename: str) -> Path:
    """Return a unique path in ``upload_dir`` by suffixing ``_NNN`` when needed."""

    candidate = upload_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = upload_dir / f"{stem}_{counter:03d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _make_document_id(path: Path) -> str:
    """Build a stable UI/backend document ID for an uploaded file."""

    return f"uploaded:{path.name}"


def _build_document_metadata(path: Path, original_name: str, size_bytes: int) -> dict[str, Any]:
    """Construct stable metadata for uploaded document registry/selection."""

    return {
        "id": _make_document_id(path),
        "name": path.name,
        "path": str(path),
        "type": path.suffix.lower().lstrip("."),
        "source": "uploaded",
        "size_bytes": size_bytes,
        "original_name": original_name,
    }


def _is_path_inside(base_dir: Path, candidate: Path) -> bool:
    """Guardrail to ensure writes stay within app-controlled upload directory."""

    base_resolved = base_dir.resolve()
    candidate_resolved = candidate.resolve()
    return base_resolved == candidate_resolved or base_resolved in candidate_resolved.parents


def save_uploaded_files(uploaded_files: list[Any] | None, upload_dir: Path = UPLOAD_DIR) -> tuple[list[dict[str, Any]], list[str]]:
    """Persist uploaded Streamlit files locally and return metadata + errors.

    Args:
        uploaded_files: Items returned from ``st.file_uploader(..., accept_multiple_files=True)``.
        upload_dir: Repo-local storage location (defaults to ``data/uploads``).

    Returns:
        Tuple of ``(saved_documents, errors)``.
    """

    if not uploaded_files:
        return [], []

    target_dir = ensure_upload_dir(upload_dir)
    saved_documents: list[dict[str, Any]] = []
    errors: list[str] = []

    for uploaded in uploaded_files:
        raw_name = getattr(uploaded, "name", "")
        if not raw_name:
            errors.append("Skipped one upload because filename metadata was missing.")
            continue

        safe_name = sanitize_filename(raw_name)
        if not is_allowed_extension(safe_name):
            errors.append(f"Unsupported file type for '{raw_name}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
            continue

        try:
            content = uploaded.getvalue()
        except Exception as exc:
            errors.append(f"Could not read '{raw_name}': {type(exc).__name__}.")
            continue

        if not content:
            errors.append(f"Skipped empty file '{raw_name}'.")
            continue

        destination = _resolve_collision_path(target_dir, safe_name)
        if not _is_path_inside(target_dir, destination):
            errors.append(f"Blocked unsafe path while saving '{raw_name}'.")
            continue

        try:
            destination.write_bytes(content)
        except Exception as exc:
            errors.append(f"Failed to save '{raw_name}': {type(exc).__name__}: {exc}")
            continue

        saved_documents.append(
            _build_document_metadata(path=destination, original_name=raw_name, size_bytes=len(content))
        )

    return saved_documents, errors


def remove_uploaded_document(document: dict[str, Any], upload_dir: Path = UPLOAD_DIR) -> None:
    """Remove an uploaded file from disk when it lives under upload directory."""

    path_raw = document.get("path")
    if not path_raw:
        return

    target_dir = ensure_upload_dir(upload_dir)
    doc_path = Path(path_raw)

    if not _is_path_inside(target_dir, doc_path):
        return

    if doc_path.exists():
        doc_path.unlink()


def register_uploaded_documents(uploaded_documents: list[dict[str, Any]]) -> None:
    """Future ingestion/indexing hook.

    This function is intentionally a no-op for the local-first UI. A future
    integration can invoke ingestion/indexing services here using the uploaded
    metadata list, without changing UI upload flow.
    """

    _ = uploaded_documents
