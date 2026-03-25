from __future__ import annotations

from agentic_rag.ingestion import MarkdownDocumentIngestor, PDFDocumentIngestor


class FakePDFConverter:
    def __init__(self, markdown: str) -> None:
        self._markdown = markdown
        self.calls: list[bytes] = []

    def convert(self, pdf_bytes: bytes) -> str:
        self.calls.append(pdf_bytes)
        return self._markdown


def test_pdf_is_converted_to_markdown_in_memory() -> None:
    converter = FakePDFConverter("# Title\n\n- bullet")
    ingestor = PDFDocumentIngestor(converter=converter, page_count_resolver=lambda _: 4)

    docs = ingestor.ingest(
        [
            {
                "content": b"%PDF fake bytes",
                "source": "s3://bucket/my.pdf",
                "source_name": "my.pdf",
                "source_type": "file",
            }
        ]
    )

    assert len(docs) == 1
    assert docs[0].text == "# Title\n\n- bullet"
    assert converter.calls == [b"%PDF fake bytes"]


def test_pdf_metadata_and_formats_are_preserved() -> None:
    converter = FakePDFConverter("## Heading\n\nParagraph")
    ingestor = PDFDocumentIngestor(converter=converter, page_count_resolver=lambda _: 7)

    doc = ingestor.ingest(
        [{"content": b"fake", "source": "a", "source_name": "a.pdf", "source_type": "upload"}]
    )[0]

    assert doc.metadata["source"] == "a"
    assert doc.metadata["source_name"] == "a.pdf"
    assert doc.metadata["source_type"] == "upload"
    assert doc.metadata["page_count"] == 7
    assert doc.metadata["original_format"] == "pdf"
    assert doc.metadata["converted_format"] == "markdown"
    assert "plain_text_fallback" in doc.metadata


def test_pdf_id_is_deterministic_for_same_input() -> None:
    converter = FakePDFConverter("# Same")
    ingestor = PDFDocumentIngestor(converter=converter, page_count_resolver=lambda _: 1)

    rec = {"content": b"fake", "source": "stable-source", "source_name": "x.pdf"}
    first = ingestor.ingest([rec])[0]
    second = ingestor.ingest([rec])[0]

    assert first.id == second.id


def test_markdown_ingestion_is_unchanged() -> None:
    ingestor = MarkdownDocumentIngestor()

    docs = ingestor.ingest(
        [
            {
                "text": "# Existing Markdown\n\nText.",
                "source": "docs/readme.md",
                "source_name": "readme.md",
                "source_type": "git",
            }
        ]
    )

    assert docs[0].text == "# Existing Markdown\n\nText."
    assert docs[0].metadata["original_format"] == "markdown"
    assert docs[0].metadata["converted_format"] == "markdown"
