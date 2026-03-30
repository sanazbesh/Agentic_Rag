"""Local real-backend wiring for Streamlit legal RAG UI.

Builds an in-memory end-to-end legal RAG dependency set from uploaded/local files,
so users can run non-mock retrieval + grounded answering without external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.chunking import MarkdownParentChildChunker
from agentic_rag.ingestion import MarkdownDocumentIngestor, PDFDocumentIngestor
from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies
from agentic_rag.orchestration.retrieval_graph import RetrievalDependencies
from agentic_rag.retrieval import (
    ChildChunkSearcher,
    ChunkReranker,
    InMemoryChildChunkRepository,
    InMemoryKeywordChunkRepository,
    InMemoryParentChunkRepository,
    KeywordSearchService,
    ParentChildRetrievalTools,
    ParentChunkStore,
)
from agentic_rag.tools import compress_context, extract_legal_entities, rewrite_query


@dataclass(slots=True)
class LocalBackendBuildResult:
    dependencies: LegalRagDependencies
    scope_meta: dict[str, Any]


def build_local_backend_dependencies(selected_documents: list[dict[str, Any]] | None) -> LocalBackendBuildResult:
    """Build LegalRagDependencies from selected local documents.

    Supported file types: `.md`, `.txt`, `.pdf`.
    """

    selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
    loaded_documents, warnings = _load_documents(selected_docs)

    chunker = MarkdownParentChildChunker()
    child_records: list[dict[str, Any]] = []
    parent_lookup: dict[str, dict[str, Any]] = {}

    for doc in loaded_documents:
        chunking_result = chunker.chunk(doc)
        child_records.extend(chunking_result.child_qdrant_records())
        parent_lookup.update(chunking_result.parent_lookup())

    supported_filter_keys: set[str] = set()
    for record in child_records:
        payload = record.get("payload")
        if isinstance(payload, dict):
            supported_filter_keys.update(str(key) for key in payload.keys())

    retrieval_tools = ParentChildRetrievalTools(
        child_searcher=ChildChunkSearcher(
            repository=InMemoryChildChunkRepository(child_records),
            default_limit=20,
        ),
        parent_store=ParentChunkStore(
            repository=InMemoryParentChunkRepository(parent_lookup),
        ),
        keyword_search_service=KeywordSearchService(
            repository=InMemoryKeywordChunkRepository(child_records),
            default_limit=20,
        ),
        chunk_reranker=ChunkReranker(),
    )

    def _hybrid_search(query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[Any]:
        scoped_filters = dict(filters or {})
        selected_ids = _as_string_set(scoped_filters.pop("selected_document_ids", []))
        selected_paths = _as_string_set(scoped_filters.pop("selected_document_paths", []))

        base_filters = {
            key: value for key, value in scoped_filters.items() if key in supported_filter_keys
        }
        results = retrieval_tools.hybrid_search(query=query, filters=base_filters or None, top_k=top_k)
        if not selected_ids and not selected_paths:
            return results

        filtered: list[Any] = []
        for result in results:
            document_id = str(result.document_id)
            metadata = dict(result.metadata or {})
            source_path = str(metadata.get("source", ""))
            if selected_ids and document_id in selected_ids:
                filtered.append(result)
                continue
            if selected_paths and source_path in selected_paths:
                filtered.append(result)
        return filtered

    retrieval_dependencies = RetrievalDependencies(
        rewrite_query=rewrite_query,
        extract_legal_entities=extract_legal_entities,
        hybrid_search=_hybrid_search,
        rerank_chunks=retrieval_tools.rerank_chunks,
        retrieve_parent_chunks=retrieval_tools.retrieve_parent_chunks,
        compress_context=compress_context,
    )

    scope_meta = {
        "backend": "local_in_memory",
        "selected_document_count": len(selected_docs),
        "loaded_document_count": len(loaded_documents),
        "parent_chunk_count": len(parent_lookup),
        "child_chunk_count": len(child_records),
        "warnings": warnings,
    }
    return LocalBackendBuildResult(dependencies=LegalRagDependencies(retrieval=retrieval_dependencies), scope_meta=scope_meta)


def _load_documents(selected_documents: list[dict[str, Any]]) -> tuple[list[Any], list[str]]:
    markdown_records: list[dict[str, Any]] = []
    pdf_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for descriptor in selected_documents:
        path_value = descriptor.get("path")
        if not path_value:
            continue
        path = Path(str(path_value))
        if not path.exists() or not path.is_file():
            warnings.append(f"missing_file:{path}")
            continue

        suffix = path.suffix.lower()
        source_name = str(descriptor.get("name") or path.name)
        base_record = {
            "id": str(descriptor.get("id") or path.name),
            "source": str(path),
            "source_name": source_name,
            "source_type": suffix.lstrip("."),
        }

        if suffix in {".md", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # pragma: no cover - defensive I/O guard
                warnings.append(f"read_failed:{path}:{type(exc).__name__}")
                continue
            markdown_records.append({**base_record, "text": text})
            continue

        if suffix == ".pdf":
            try:
                payload = path.read_bytes()
            except Exception as exc:  # pragma: no cover - defensive I/O guard
                warnings.append(f"read_failed:{path}:{type(exc).__name__}")
                continue
            pdf_records.append({**base_record, "content": payload})
            continue

        warnings.append(f"unsupported_extension:{path.name}")

    docs = []
    if markdown_records:
        docs.extend(MarkdownDocumentIngestor().ingest(markdown_records))

    if pdf_records:
        try:
            docs.extend(PDFDocumentIngestor().ingest(pdf_records))
        except Exception as exc:
            warnings.append(f"pdf_ingestion_failed:{type(exc).__name__}:{exc}")

    return docs, warnings


def _as_string_set(values: Any) -> set[str]:
    if isinstance(values, list):
        return {str(v) for v in values if v}
    return set()
