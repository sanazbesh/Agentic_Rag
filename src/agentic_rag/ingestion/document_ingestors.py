"""Concrete ingestors for common source formats."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from typing import Any

from agentic_rag.ingestion.converters import (
    PDFToMarkdownConverter,
    PyMuPDF4LLMConverter,
    count_pdf_pages,
)
from agentic_rag.ingestion.interfaces import DocumentIngestor
from agentic_rag.types import Document


class MarkdownDocumentIngestor(DocumentIngestor):
    """Ingest markdown records without changing content."""

    def ingest(self, records: Iterable[dict]) -> list[Document]:
        documents: list[Document] = []
        for record in records:
            text = str(record.get("text") or record.get("content") or "")
            metadata = dict(record.get("metadata", {}))

            source = str(record.get("source", metadata.get("source", "unknown")))
            source_name = str(record.get("source_name", metadata.get("source_name", source)))
            source_type = str(
                record.get("source_type", metadata.get("source_type", "markdown"))
            )

            metadata.setdefault("source", source)
            metadata.setdefault("source_name", source_name)
            metadata.setdefault("source_type", source_type)
            metadata.setdefault("original_format", "markdown")
            metadata.setdefault("converted_format", "markdown")

            doc_id = str(record.get("id") or _stable_document_id(source=source, text=text))
            documents.append(Document(id=doc_id, text=text, metadata=metadata))

        return documents


class PDFDocumentIngestor(DocumentIngestor):
    """Convert PDF bytes to Markdown while preserving modular ingestion boundaries."""

    def __init__(
        self,
        converter: PDFToMarkdownConverter | None = None,
        page_count_resolver: Callable[[bytes], int] = count_pdf_pages,
    ) -> None:
        self._converter = converter or PyMuPDF4LLMConverter()
        self._page_count_resolver = page_count_resolver

    def ingest(self, records: Iterable[dict]) -> list[Document]:
        documents: list[Document] = []

        for record in records:
            pdf_bytes = _coerce_pdf_bytes(record)
            markdown_text = self._converter.convert(pdf_bytes)
            page_count = int(record.get("page_count") or self._page_count_resolver(pdf_bytes))
            plain_text_fallback = _markdown_to_plain_text(markdown_text)

            metadata = dict(record.get("metadata", {}))
            source = str(record.get("source", metadata.get("source", "unknown")))
            source_name = str(record.get("source_name", metadata.get("source_name", source)))
            source_type = str(record.get("source_type", metadata.get("source_type", "pdf")))

            metadata.update(
                {
                    "source": source,
                    "source_name": source_name,
                    "source_type": source_type,
                    "page_count": page_count,
                    "original_format": "pdf",
                    "converted_format": "markdown",
                    "plain_text_fallback": plain_text_fallback,
                }
            )

            doc_id = str(
                record.get("id")
                or _stable_document_id(source=source, text=markdown_text, suffix=f"pages:{page_count}")
            )

            documents.append(Document(id=doc_id, text=markdown_text, metadata=metadata))

        return documents


def _coerce_pdf_bytes(record: dict[str, Any]) -> bytes:
    payload = record.get("content", record.get("bytes", b""))
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    raise TypeError("PDF records must include raw bytes in `content` or `bytes`.")


def _stable_document_id(source: str, text: str, suffix: str = "") -> str:
    hasher = hashlib.sha256()
    hasher.update(source.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(text.encode("utf-8"))
    if suffix:
        hasher.update(b"\x00")
        hasher.update(suffix.encode("utf-8"))
    return hasher.hexdigest()


def _markdown_to_plain_text(markdown_text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", markdown_text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
