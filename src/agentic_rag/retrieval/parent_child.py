"""Parent-child retrieval services and thin tool functions.

Child chunks are searched as retrieval units (Qdrant-ready payloads), while
parent chunks are fetched as larger context units for later LLM use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class ChildSearchResult:
    """A structured child retrieval hit with parent linkage metadata."""

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class HybridSearchResult:
    """Combined vector + keyword hit used by ``hybrid_search``.

    Hybrid retrieval improves legal search quality by mixing semantic recall
    (vector search) with exact-term precision (keyword search for statutes,
    clauses, and term-of-art matches).
    """

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    combined_score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RerankedChunkResult:
    """Re-ranked chunk used before parent expansion in the retrieval pipeline.

    Reranking is critical in legal RAG because candidate chunks can be highly
    similar; stronger lexical/issue-focused scoring improves precision for final
    parent chunk selection.
    """

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    rerank_score: float
    original_score: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ParentChunkResult:
    """A structured parent chunk record fetched by ``parent_chunk_id``."""

    parent_chunk_id: str
    document_id: str
    text: str
    source: str
    source_name: str
    heading_path: tuple[str, ...] = ()
    heading_text: str = ""
    parent_order: int = 0
    part_number: int = 1
    total_parts: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ChildChunkRepository(ABC):
    """Storage abstraction for searching child chunks (e.g., Qdrant)."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[ChildSearchResult]:
        """Return ranked child chunk matches for a query."""


class ParentChunkRepository(ABC):
    """Storage abstraction for fetching parent chunks by id."""

    @abstractmethod
    def get_by_ids(self, parent_ids: Sequence[str]) -> list[ParentChunkResult]:
        """Return parent chunk records for known ids."""


class KeywordChunkRepository(ABC):
    """Storage abstraction for keyword retrieval over child chunks."""

    @abstractmethod
    def search_keyword(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[ChildSearchResult]:
        """Return keyword-ranked child chunk matches for a query."""


@dataclass(slots=True)
class ChildChunkSearcher:
    """Service wrapper that validates queries then delegates to repository search."""

    repository: ChildChunkRepository
    default_limit: int = 10

    def search_child_chunks(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ChildSearchResult]:
        """Search child chunks and return structured hits linked to parent ids."""

        if not query or not query.strip():
            return []
        normalized_query = query.strip()
        return self.repository.search(normalized_query, filters=filters, limit=self.default_limit)


@dataclass(slots=True)
class VectorSearchService:
    """Vector retrieval service wrapper for child chunks."""

    child_searcher: ChildChunkSearcher

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ChildSearchResult]:
        return self.child_searcher.search_child_chunks(query=query, filters=filters)


@dataclass(slots=True)
class KeywordSearchService:
    """Keyword retrieval service abstraction (BM25-compatible interface)."""

    repository: KeywordChunkRepository
    default_limit: int = 10

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ChildSearchResult]:
        if not query or not query.strip():
            return []
        return self.repository.search_keyword(query.strip(), filters=filters, limit=self.default_limit)


@dataclass(slots=True)
class HybridSearchService:
    """Fuses vector and keyword retrieval into a single ranked list."""

    vector_service: VectorSearchService
    keyword_service: KeywordSearchService
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        if not query or not query.strip():
            return []

        vector_hits = self.vector_service.search(query=query, filters=filters)
        keyword_hits = self.keyword_service.search(query=query, filters=filters)
        if not vector_hits and not keyword_hits:
            return []

        normalized_vector = _normalize_scores(vector_hits)
        normalized_keyword = _normalize_scores(keyword_hits)

        merged: dict[str, HybridSearchResult] = {}
        for hit in vector_hits:
            vec_score = normalized_vector.get(hit.child_chunk_id, 0.0)
            merged[hit.child_chunk_id] = HybridSearchResult(
                child_chunk_id=hit.child_chunk_id,
                parent_chunk_id=hit.parent_chunk_id,
                document_id=hit.document_id,
                text=hit.text,
                combined_score=(self.vector_weight * vec_score),
                vector_score=hit.score,
                keyword_score=0.0,
                payload=dict(hit.payload),
            )

        for hit in keyword_hits:
            key_score = normalized_keyword.get(hit.child_chunk_id, 0.0)
            existing = merged.get(hit.child_chunk_id)
            if existing is None:
                merged[hit.child_chunk_id] = HybridSearchResult(
                    child_chunk_id=hit.child_chunk_id,
                    parent_chunk_id=hit.parent_chunk_id,
                    document_id=hit.document_id,
                    text=hit.text,
                    combined_score=(self.keyword_weight * key_score),
                    vector_score=0.0,
                    keyword_score=hit.score,
                    payload=dict(hit.payload),
                )
                continue

            merged[hit.child_chunk_id] = HybridSearchResult(
                child_chunk_id=existing.child_chunk_id,
                parent_chunk_id=existing.parent_chunk_id,
                document_id=existing.document_id,
                text=existing.text,
                combined_score=existing.combined_score + (self.keyword_weight * key_score),
                vector_score=existing.vector_score,
                keyword_score=hit.score,
                payload=existing.payload,
            )

        return sorted(
            merged.values(),
            key=lambda item: (
                -item.combined_score,
                -(1 if item.vector_score > 0 else 0),
                item.child_chunk_id,
            ),
        )


@dataclass(slots=True)
class ChunkReranker:
    """Deterministic lexical reranker placeholder upgradeable to cross-encoder."""

    legal_bonus_terms: tuple[str, ...] = ("means", "shall", "must", "defined", "rule", "clause")

    def rerank(
        self,
        chunks: Sequence[ChildSearchResult | HybridSearchResult | RerankedChunkResult],
        *,
        query: str,
    ) -> list[RerankedChunkResult]:
        if not query or not query.strip() or not chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        reranked: list[RerankedChunkResult] = []
        for chunk in chunks:
            chunk_text_tokens = set(_tokenize(chunk.text))
            lexical_matches = sum(1 for token in query_tokens if token in chunk_text_tokens)
            lexical_score = float(lexical_matches) / float(len(query_tokens))
            legal_bonus = 0.0
            for term in self.legal_bonus_terms:
                if term in chunk_text_tokens:
                    legal_bonus += 0.02
            rerank_score = lexical_score + legal_bonus
            reranked.append(
                RerankedChunkResult(
                    child_chunk_id=chunk.child_chunk_id,
                    parent_chunk_id=chunk.parent_chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    rerank_score=rerank_score,
                    original_score=_extract_original_score(chunk),
                    payload=dict(chunk.payload),
                )
            )

        reranked.sort(
            key=lambda item: (
                -item.rerank_score,
                -item.original_score,
                item.child_chunk_id,
            )
        )
        return reranked


@dataclass(slots=True)
class ParentChunkStore:
    """Service wrapper that fetches unique parent chunks in requested order."""

    repository: ParentChunkRepository

    def retrieve_parent_chunks(self, parent_ids: Sequence[str]) -> list[ParentChunkResult]:
        """Fetch parent chunks by ``parent_chunk_id``, deduplicated and ordered."""

        deduped_ids = [pid for pid in dict.fromkeys(parent_ids) if pid]
        if not deduped_ids:
            return []
        return self.repository.get_by_ids(deduped_ids)


@dataclass(slots=True)
class ParentChildRetrievalTools:
    """Thin tool facade over child-search and parent-fetch services."""

    child_searcher: ChildChunkSearcher
    parent_store: ParentChunkStore
    keyword_search_service: KeywordSearchService | None = None
    chunk_reranker: ChunkReranker = field(default_factory=ChunkReranker)

    def search_child_chunks(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ChildSearchResult]:
        """Tool: retrieve relevant child chunks and keep parent linkage metadata."""

        return self.child_searcher.search_child_chunks(query=query, filters=filters)

    def retrieve_parent_chunks(self, parent_ids: Sequence[str]) -> list[ParentChunkResult]:
        """Tool: fetch parent chunks for context expansion after child retrieval."""

        return self.parent_store.retrieve_parent_chunks(parent_ids=parent_ids)

    def hybrid_search(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """Tool: fuse vector and keyword retrieval for legal-domain recall + precision."""

        if self.keyword_search_service is None:
            return []
        hybrid_service = HybridSearchService(
            vector_service=VectorSearchService(self.child_searcher),
            keyword_service=self.keyword_search_service,
        )
        return hybrid_service.search(query=query, filters=filters)

    def rerank_chunks(
        self,
        chunks: Sequence[ChildSearchResult | HybridSearchResult | RerankedChunkResult],
        query: str,
    ) -> list[RerankedChunkResult]:
        """Tool: rerank child chunks before parent retrieval for higher precision."""

        return self.chunk_reranker.rerank(chunks, query=query)


@dataclass(slots=True)
class InMemoryChildChunkRepository(ChildChunkRepository):
    """Simple deterministic repository over Qdrant-ready child chunk records."""

    records: Sequence[Mapping[str, Any]]

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[ChildSearchResult]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        matched: list[ChildSearchResult] = []
        for record in self.records:
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if not _passes_filters(payload, filters):
                continue

            text = str(record.get("text", payload.get("text", "")))
            score = _score_text(text, tokens)
            if score <= 0:
                continue

            child_chunk_id = str(record.get("id", payload.get("child_chunk_id", "")))
            parent_chunk_id = str(payload.get("parent_chunk_id", ""))
            document_id = str(payload.get("document_id", ""))
            if not child_chunk_id or not parent_chunk_id:
                continue

            matched.append(
                ChildSearchResult(
                    child_chunk_id=child_chunk_id,
                    parent_chunk_id=parent_chunk_id,
                    document_id=document_id,
                    text=text,
                    score=score,
                    payload=dict(payload),
                )
            )

        matched.sort(
            key=lambda item: (
                -item.score,
                item.document_id,
                item.parent_chunk_id,
                item.child_chunk_id,
            )
        )
        return matched[: max(0, limit)]


@dataclass(slots=True)
class InMemoryKeywordChunkRepository(KeywordChunkRepository):
    """Deterministic in-memory keyword repository with BM25-style scoring hooks."""

    records: Sequence[Mapping[str, Any]]

    def search_keyword(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[ChildSearchResult]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        matched: list[ChildSearchResult] = []
        for record in self.records:
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if not _passes_filters(payload, filters):
                continue

            text = str(record.get("text", payload.get("text", "")))
            score = _keyword_score_text(text, tokens)
            if score <= 0:
                continue

            child_chunk_id = str(record.get("id", payload.get("child_chunk_id", "")))
            parent_chunk_id = str(payload.get("parent_chunk_id", ""))
            document_id = str(payload.get("document_id", ""))
            if not child_chunk_id or not parent_chunk_id:
                continue

            matched.append(
                ChildSearchResult(
                    child_chunk_id=child_chunk_id,
                    parent_chunk_id=parent_chunk_id,
                    document_id=document_id,
                    text=text,
                    score=score,
                    payload=dict(payload),
                )
            )

        matched.sort(
            key=lambda item: (
                -item.score,
                item.document_id,
                item.parent_chunk_id,
                item.child_chunk_id,
            )
        )
        return matched[: max(0, limit)]


@dataclass(slots=True)
class InMemoryParentChunkRepository(ParentChunkRepository):
    """Dictionary-backed parent chunk lookup keyed by ``parent_chunk_id``."""

    parent_lookup: Mapping[str, Mapping[str, Any]]

    def get_by_ids(self, parent_ids: Sequence[str]) -> list[ParentChunkResult]:
        output: list[ParentChunkResult] = []
        for parent_id in parent_ids:
            record = self.parent_lookup.get(parent_id)
            if record is None:
                continue
            output.append(_parent_from_record(parent_id=parent_id, record=record))
        return output


def _parent_from_record(parent_id: str, record: Mapping[str, Any]) -> ParentChunkResult:
    heading_path_val = record.get("heading_path", ())
    if isinstance(heading_path_val, Sequence) and not isinstance(heading_path_val, (str, bytes)):
        heading_path = tuple(str(part) for part in heading_path_val)
    else:
        heading_path = ()
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "parent_chunk_id",
            "document_id",
            "text",
            "source",
            "source_name",
            "heading_path",
            "heading_text",
            "parent_order",
            "part_number",
            "total_parts",
        }
    }
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id=str(record.get("document_id", "")),
        text=str(record.get("text", "")),
        source=str(record.get("source", "")),
        source_name=str(record.get("source_name", "")),
        heading_path=heading_path,
        heading_text=str(record.get("heading_text", "")),
        parent_order=int(record.get("parent_order", 0) or 0),
        part_number=int(record.get("part_number", 1) or 1),
        total_parts=int(record.get("total_parts", 1) or 1),
        metadata=metadata,
    )


def _passes_filters(payload: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    if filters is None:
        return True
    for key, expected in filters.items():
        if payload.get(key) != expected:
            return False
    return True


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token.strip()]


def _score_text(text: str, tokens: Sequence[str]) -> float:
    wordset = set(_tokenize(text))
    if not tokens:
        return 0.0
    matches = sum(1 for token in tokens if token in wordset)
    return float(matches) / float(len(tokens))


def _keyword_score_text(text: str, tokens: Sequence[str]) -> float:
    terms = _tokenize(text)
    if not terms or not tokens:
        return 0.0
    term_freq: dict[str, int] = {}
    for term in terms:
        term_freq[term] = term_freq.get(term, 0) + 1
    score = 0.0
    for token in tokens:
        tf = term_freq.get(token, 0)
        if tf > 0:
            score += 1.0 + (float(tf) / float(len(terms)))
    return score / float(len(tokens))


def _normalize_scores(results: Sequence[ChildSearchResult]) -> dict[str, float]:
    if not results:
        return {}
    max_score = max(item.score for item in results)
    min_score = min(item.score for item in results)
    if max_score == min_score:
        return {item.child_chunk_id: 1.0 for item in results}
    denom = max_score - min_score
    return {item.child_chunk_id: (item.score - min_score) / denom for item in results}


def _extract_original_score(
    chunk: ChildSearchResult | HybridSearchResult | RerankedChunkResult,
) -> float:
    if isinstance(chunk, HybridSearchResult):
        return chunk.combined_score
    if isinstance(chunk, RerankedChunkResult):
        return chunk.rerank_score
    return chunk.score
