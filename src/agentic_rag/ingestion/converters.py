"""PDF conversion abstractions and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PDFToMarkdownConverter(Protocol):
    """Converts PDF bytes into structure-preserving Markdown text."""

    def convert(self, pdf_bytes: bytes) -> str:
        """Return Markdown content for the provided PDF bytes."""


@dataclass(slots=True)
class PyMuPDF4LLMConverter:
    """Default PDF → Markdown converter based on PyMuPDF + pymupdf4llm."""

    def convert(self, pdf_bytes: bytes) -> str:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "PyMuPDF (fitz) is required for PDF conversion."
            ) from exc

        try:
            import pymupdf4llm
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "pymupdf4llm is required for Markdown-first PDF conversion."
            ) from exc

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            markdown = pymupdf4llm.to_markdown(doc)

        return markdown.strip()


def count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count pages from in-memory PDF bytes."""

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyMuPDF (fitz) is required for page counting.") from exc

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count
