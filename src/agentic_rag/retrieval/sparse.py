"""Sparse child-chunk retrieval service.

Sparse retrieval complements dense retrieval in legal RAG by improving exact
match recall for statutes, rule citations, and structured legal terminology.
The service returns a strict typed output compatible with future hybrid fusion.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained test envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.indexing.sparse_child_chunks import BM25Index, SparseChunkMetadata


class SparseSearchResult(BaseModel):
    """Typed sparse result record for child-chunk retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    sparse_score: float
    metadata: SparseChunkMetadata = Field(default_factory=SparseChunkMetadata)


class SparseSearchService:
    """Service facade over sparse index with deterministic filtering + ordering."""

    def __init__(self, index: BM25Index, *, default_top_k: int = 20) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be > 0")
        self._index = index
        self._default_top_k = default_top_k

    def search_child_chunks_sparse(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[SparseSearchResult]:
        """Run lexical search over child chunks and return typed sparse results."""

        normalized_query = query.strip()
        if not normalized_query:
            return []

        effective_top_k = self._default_top_k if top_k is None else max(0, top_k)
        if effective_top_k <= 0:
            return []

        raw_hits = self._index.search(normalized_query, top_k=effective_top_k)
        if not raw_hits:
            return []

        filtered_hits: list[SparseSearchResult] = []
        for doc, score in raw_hits:
            if not _passes_filters(doc.metadata, filters):
                continue
            filtered_hits.append(
                SparseSearchResult(
                    child_chunk_id=doc.child_chunk_id,
                    parent_chunk_id=doc.parent_chunk_id,
                    document_id=doc.document_id,
                    text=doc.text,
                    sparse_score=float(score),
                    metadata=doc.metadata,
                )
            )

        filtered_hits.sort(key=lambda item: (-item.sparse_score, item.child_chunk_id))
        return filtered_hits[:effective_top_k]


def search_child_chunks_sparse(
    query: str,
    index: BM25Index,
    filters: Mapping[str, Any] | None = None,
    *,
    top_k: int = 20,
) -> list[SparseSearchResult]:
    """Convenience function wrapper matching pipeline search call shape."""

    return SparseSearchService(index=index, default_top_k=top_k).search_child_chunks_sparse(
        query=query,
        filters=filters,
        top_k=top_k,
    )


def _passes_filters(metadata: SparseChunkMetadata, filters: Mapping[str, Any] | None) -> bool:
    if filters is None:
        return True
    metadata_map = metadata.model_dump()
    for key, expected in filters.items():
        if metadata_map.get(key) != expected:
            return False
    return True
