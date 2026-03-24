"""Concrete ingestion utilities for PDF and Markdown sources.

The implementation focuses on local-file ingestion and keeps third-party
requirements optional.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
import re

from agentic_rag.ingestion.interfaces import DataConnector, DocumentIngestor
from agentic_rag.types import Document


@dataclass(slots=True)
class LocalFileConnector(DataConnector):
    """Collect local Markdown and PDF files and emit normalized records."""

    roots: Sequence[str | Path]
    include_extensions: tuple[str, ...] = (".md", ".markdown", ".pdf")

    def fetch(self) -> Iterable[dict]:
        """Yield file records with path metadata and source bytes."""
        for root in self.roots:
            root_path = Path(root).expanduser().resolve()
            yield from self._walk_root(root_path)

    def _walk_root(self, root_path: Path) -> Iterator[dict]:
        if root_path.is_file():
            if self._is_supported(root_path):
                yield self._build_record(root_path)
            return

        for path in sorted(root_path.rglob("*")):
            if path.is_file() and self._is_supported(path):
                yield self._build_record(path)

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.include_extensions

    def _build_record(self, path: Path) -> dict:
        return {
            "id": f"file::{path}",
            "path": str(path),
            "name": path.name,
            "extension": path.suffix.lower(),
            "content": path.read_bytes(),
        }


@dataclass(slots=True)
class PDFDocumentIngestor(DocumentIngestor):
    """Convert PDF file records to :class:`Document` instances."""

    def ingest(self, records: Iterable[dict]) -> list[Document]:
        documents: list[Document] = []
        for record in records:
            if record.get("extension") != ".pdf":
                continue
            text, page_count = _extract_pdf_text(record["content"])
            documents.append(
                Document(
                    id=_stable_document_id(record, text),
                    text=text,
                    metadata={
                        "source": record.get("path"),
                        "source_name": record.get("name"),
                        "source_type": "pdf",
                        "page_count": page_count,
                    },
                )
            )
        return documents


@dataclass(slots=True)
class MarkdownDocumentIngestor(DocumentIngestor):
    """Convert Markdown records to :class:`Document` with section visibility.

    Inspired by Chunky's Markdown visibility tooling, this ingestor exposes
    a heading outline and per-heading line numbers in metadata so downstream
    systems can inspect structure before chunking.
    """

    keep_markdown: bool = True

    def ingest(self, records: Iterable[dict]) -> list[Document]:
        documents: list[Document] = []
        for record in records:
            extension = record.get("extension")
            if extension not in {".md", ".markdown"}:
                continue
            markdown_text = record["content"].decode("utf-8", errors="replace")
            outline = _markdown_outline(markdown_text)
            plain_text = _strip_markdown(markdown_text)
            documents.append(
                Document(
                    id=_stable_document_id(record, markdown_text),
                    text=markdown_text if self.keep_markdown else plain_text,
                    metadata={
                        "source": record.get("path"),
                        "source_name": record.get("name"),
                        "source_type": "markdown",
                        "outline": outline,
                        "plain_text": plain_text,
                    },
                )
            )
        return documents


def _stable_document_id(record: dict, text: str) -> str:
    basis = f"{record.get('path','')}::{text[:500]}"
    return f"doc::{sha1(basis.encode('utf-8')).hexdigest()}"


def _extract_pdf_text(content: bytes) -> tuple[str, int]:
    """Extract text from PDF bytes.

    Raises:
        ImportError: if ``pypdf`` is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on runtime extras.
        raise ImportError(
            "PDF ingestion requires `pypdf`. Install with: pip install pypdf"
        ) from exc

    from io import BytesIO

    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    merged = "\n\n".join(page.strip() for page in pages if page.strip())
    return merged, len(reader.pages)


def _markdown_outline(markdown_text: str) -> list[dict]:
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)\s*$")
    outline: list[dict] = []
    for line_number, line in enumerate(markdown_text.splitlines(), start=1):
        match = heading_pattern.match(line)
        if not match:
            continue
        hashes, title = match.groups()
        outline.append(
            {
                "level": len(hashes),
                "title": title.strip(),
                "line_number": line_number,
            }
        )
    return outline


def _strip_markdown(markdown_text: str) -> str:
    """Lightweight Markdown-to-text fallback without external dependencies."""
    text = re.sub(r"```[\s\S]*?```", "", markdown_text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
