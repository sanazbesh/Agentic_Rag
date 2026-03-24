# Agentic_Rag

Base scaffold for an agentic RAG system.

## Project structure

```text
src/agentic_rag/
  config/
  evaluation/
  indexing/
  ingestion/
  orchestration/
  prompts/
  retrieval/
  tools/
  types.py
```

Each module includes abstract base interfaces and incremental concrete implementations.

## Ingestion support (PDF + Markdown)

The `ingestion` module now includes local-file ingestion utilities:

- `LocalFileConnector`: reads `.pdf`, `.md`, and `.markdown` files from one or more directories.
- `PDFDocumentIngestor`: extracts text from PDFs (requires `pypdf`).
- `MarkdownDocumentIngestor`: ingests Markdown with structure metadata (`outline` + line numbers) for pre-chunking visibility inspired by [Chunky](https://github.com/GiovanniPasq/chunky).

### Example

```python
from agentic_rag.ingestion import (
    LocalFileConnector,
    MarkdownDocumentIngestor,
    PDFDocumentIngestor,
)

connector = LocalFileConnector(["./docs"])
records = list(connector.fetch())

pdf_documents = PDFDocumentIngestor().ingest(records)
md_documents = MarkdownDocumentIngestor().ingest(records)
```
