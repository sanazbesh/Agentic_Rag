"""Deterministic BM25-style sparse indexing for legal child chunks.

Sparse lexical retrieval is critical in legal RAG for exact-term precision on
citations, clause markers, and statutory tokens that dense-only search may miss
(e.g., "§ 2-207" or "Rule 12(b)(6)"). This module provides a backend-swappable
sparse index abstraction and a deterministic in-memory BM25 implementation.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from threading import RLock
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained test envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.chunking.models import ChildChunk

logger = logging.getLogger(__name__)


class SparseChunkMetadata(BaseModel):
    """Typed metadata payload for sparse-indexed child chunks."""

    model_config = ConfigDict(extra="allow", frozen=True)

    source: str | None = None
    source_name: str | None = None
    heading: str | None = None
    jurisdiction: str | None = None
    document_type: str | None = None
    court: str | None = None
    date: str | None = None


class SparseIndexedChildChunk(BaseModel):
    """Canonical sparse-index record linked to parent/child chunk identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    metadata: SparseChunkMetadata = Field(default_factory=SparseChunkMetadata)


class SparseIndexingResult(BaseModel):
    """Diagnostics for sparse indexing runs with failure/skipped tracking."""

    model_config = ConfigDict(extra="forbid")

    total_chunks_indexed: int = 0
    failed_chunk_ids: list[str] = Field(default_factory=list)
    skipped_chunk_ids: list[str] = Field(default_factory=list)


class LegalSparseTokenizer:
    """Deterministic tokenizer preserving legally meaningful notation.

    Tokenization is intentionally conservative: lowercasing + whitespace split.
    This preserves tokens like "§", "12(b)(6)", and "2-207" without stripping
    punctuation that can change legal meaning.
    """

    _whitespace_re = re.compile(r"\s+")

    def tokenize(self, text: str) -> list[str]:
        normalized = self._whitespace_re.sub(" ", text.strip().lower())
        if not normalized:
            return []
        return [token for token in normalized.split(" ") if token]


class SparseIndex(ABC):
    """Swappable sparse-index interface for child chunk indexing + search."""

    @abstractmethod
    def index_child_chunks(self, child_chunks: Iterable[ChildChunk | Mapping[str, Any]]) -> SparseIndexingResult:
        """Index child chunks for sparse retrieval."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 20) -> list[tuple[SparseIndexedChildChunk, float]]:
        """Return ranked sparse hits and raw sparse scores."""


class BM25Index(SparseIndex):
    """Thread-safe deterministic in-memory BM25 index.

    Notes:
    - This baseline backend is non-persistent and process-local.
    - It is designed for easy replacement by persistent/Qdrant sparse backends.
    - Duplicate ``child_chunk_id`` values are resolved deterministically with
      last-write-wins semantics during each indexing call.
    """

    def __init__(self, *, tokenizer: LegalSparseTokenizer | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self._tokenizer = tokenizer or LegalSparseTokenizer()
        self._k1 = k1
        self._b = b
        self._lock = RLock()

        self._documents: dict[str, SparseIndexedChildChunk] = {}
        self._term_freqs: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._document_frequency: Counter[str] = Counter()
        self._total_doc_length = 0

    def index_child_chunks(self, child_chunks: Iterable[ChildChunk | Mapping[str, Any]]) -> SparseIndexingResult:
        result = SparseIndexingResult()
        deduped: dict[str, ChildChunk | Mapping[str, Any]] = {}

        for chunk in child_chunks:
            child_id = _extract_child_chunk_id(chunk)
            if not child_id:
                result.skipped_chunk_ids.append("")
                continue
            deduped[child_id] = chunk

        with self._lock:
            for child_id, raw_chunk in deduped.items():
                try:
                    parsed = _to_sparse_indexed_chunk(raw_chunk)
                except ValueError:
                    result.skipped_chunk_ids.append(child_id)
                    continue
                except Exception:
                    result.failed_chunk_ids.append(child_id)
                    continue

                if not parsed.text.strip():
                    result.skipped_chunk_ids.append(child_id)
                    continue

                terms = self._tokenizer.tokenize(parsed.text)
                if not terms:
                    result.skipped_chunk_ids.append(child_id)
                    continue

                self._remove_if_exists(child_id)

                term_freq = Counter(terms)
                self._documents[child_id] = parsed
                self._term_freqs[child_id] = term_freq
                self._doc_lengths[child_id] = len(terms)
                self._total_doc_length += len(terms)
                for term in term_freq:
                    self._document_frequency[term] += 1

                result.total_chunks_indexed += 1

        logger.info(
            "sparse indexing completed",
            extra={
                "chunks_indexed": result.total_chunks_indexed,
                "failed_count": len(result.failed_chunk_ids),
                "skipped_count": len(result.skipped_chunk_ids),
            },
        )
        return result

    def search(self, query: str, *, top_k: int = 20) -> list[tuple[SparseIndexedChildChunk, float]]:
        if top_k <= 0:
            return []
        query_terms = self._tokenizer.tokenize(query)
        if not query_terms:
            return []

        with self._lock:
            total_docs = len(self._documents)
            if total_docs == 0:
                return []
            avg_doc_len = self._total_doc_length / total_docs
            scored: list[tuple[SparseIndexedChildChunk, float]] = []

            for child_id, doc in self._documents.items():
                score = self._bm25_score(
                    query_terms=query_terms,
                    term_freq=self._term_freqs[child_id],
                    doc_len=self._doc_lengths[child_id],
                    total_docs=total_docs,
                    avg_doc_len=avg_doc_len,
                )
                if score <= 0.0:
                    continue
                scored.append((doc, score))

        scored.sort(key=lambda item: (-item[1], item[0].child_chunk_id))
        return scored[:top_k]

    def _bm25_score(
        self,
        *,
        query_terms: list[str],
        term_freq: Counter[str],
        doc_len: int,
        total_docs: int,
        avg_doc_len: float,
    ) -> float:
        score = 0.0
        unique_query_terms = Counter(query_terms)
        for term, query_tf in unique_query_terms.items():
            tf = term_freq.get(term, 0)
            if tf <= 0:
                continue
            df = self._document_frequency.get(term, 0)
            idf = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))
            denom = tf + self._k1 * (1.0 - self._b + self._b * (float(doc_len) / float(avg_doc_len)))
            score += float(query_tf) * idf * ((tf * (self._k1 + 1.0)) / denom)
        return score

    def _remove_if_exists(self, child_chunk_id: str) -> None:
        existing_terms = self._term_freqs.get(child_chunk_id)
        if existing_terms is None:
            return
        for term in existing_terms:
            self._document_frequency[term] -= 1
            if self._document_frequency[term] <= 0:
                del self._document_frequency[term]
        self._total_doc_length -= self._doc_lengths.get(child_chunk_id, 0)
        self._documents.pop(child_chunk_id, None)
        self._term_freqs.pop(child_chunk_id, None)
        self._doc_lengths.pop(child_chunk_id, None)


def _extract_child_chunk_id(chunk: ChildChunk | Mapping[str, Any]) -> str:
    if isinstance(chunk, ChildChunk):
        return chunk.child_chunk_id
    return str(chunk.get("child_chunk_id", "") or "")


def _to_sparse_indexed_chunk(chunk: ChildChunk | Mapping[str, Any]) -> SparseIndexedChildChunk:
    if isinstance(chunk, ChildChunk):
        metadata = SparseChunkMetadata(
            source=chunk.source,
            source_name=chunk.source_name,
            heading=(" > ".join(chunk.heading_path) if chunk.heading_path else None),
        )
        return SparseIndexedChildChunk(
            child_chunk_id=chunk.child_chunk_id,
            parent_chunk_id=chunk.parent_chunk_id,
            document_id=chunk.document_id,
            text=chunk.text,
            metadata=metadata,
        )

    child_chunk_id = str(chunk.get("child_chunk_id", "") or "")
    parent_chunk_id = str(chunk.get("parent_chunk_id", "") or "")
    document_id = str(chunk.get("document_id", "") or "")
    text = str(chunk.get("text", "") or "")
    metadata_value = chunk.get("metadata", {})
    if not child_chunk_id or not parent_chunk_id or not document_id:
        raise ValueError("missing required sparse chunk fields")
    if isinstance(metadata_value, Mapping):
        metadata = SparseChunkMetadata.model_validate(dict(metadata_value))
    else:
        metadata = SparseChunkMetadata()
    return SparseIndexedChildChunk(
        child_chunk_id=child_chunk_id,
        parent_chunk_id=parent_chunk_id,
        document_id=document_id,
        text=text,
        metadata=metadata,
    )
