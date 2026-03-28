"""Parent-child retrieval services and thin tool functions.

Child chunks are searched as retrieval units (Qdrant-ready payloads), while
parent chunks are fetched as larger context units for later LLM use.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained test envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ChildSearchResult:
    """A structured child retrieval hit with parent linkage metadata."""

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)


class HybridSearchResult(BaseModel):
    """Fused child-chunk hit for legal hybrid retrieval.

    Legal search benefits from combining dense semantic retrieval with sparse
    lexical retrieval because citations and term-of-art phrases often require
    exact matching while factual paraphrases benefit from semantic matching.
    This result model preserves both source-specific signals and parent linkage
    metadata required for downstream parent expansion and reranking.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    text: str
    hybrid_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    matched_in_dense: bool = False
    matched_in_sparse: bool = False

    @property
    def payload(self) -> Mapping[str, Any]:
        """Backward-compatible alias for existing pipeline code."""

        return self.metadata


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
class DenseChildSearchService:
    """Dense semantic child-chunk retrieval facade."""

    vector_service: VectorSearchService

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int,
    ) -> list[ChildSearchResult]:
        results = self.vector_service.search(query=query, filters=filters)
        return results[: max(0, limit)]


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
class SparseChildSearchService:
    """Sparse/BM25 child-chunk retrieval facade."""

    keyword_service: KeywordSearchService

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int,
    ) -> list[ChildSearchResult]:
        results = self.keyword_service.search(query=query, filters=filters)
        return results[: max(0, limit)]


@dataclass(slots=True)
class RRFFuser:
    """Reciprocal Rank Fusion (RRF) over dense + sparse child chunk result lists.

    RRF is used instead of raw score addition because dense and sparse scores
    are not naturally comparable across scales. Rank-only fusion is robust,
    deterministic, and easy to reason about.
    """

    rrf_k: int = 60

    def fuse(
        self,
        *,
        dense_results: Sequence[ChildSearchResult],
        sparse_results: Sequence[ChildSearchResult],
        top_k: int,
    ) -> list[HybridSearchResult]:
        if top_k <= 0:
            return []

        dense_ranked = _dedupe_by_child_chunk_id(dense_results)
        sparse_ranked = _dedupe_by_child_chunk_id(sparse_results)

        merged: dict[str, dict[str, Any]] = {}
        self._merge_source(
            merged=merged,
            source_results=dense_ranked,
            source_name="dense",
        )
        self._merge_source(
            merged=merged,
            source_results=sparse_ranked,
            source_name="sparse",
        )

        fused_results = [
            HybridSearchResult(
                child_chunk_id=record["child_chunk_id"],
                parent_chunk_id=record["parent_chunk_id"],
                document_id=record["document_id"],
                text=record["text"],
                metadata=record["metadata"],
                dense_score=record["dense_score"],
                sparse_score=record["sparse_score"],
                dense_rank=record["dense_rank"],
                sparse_rank=record["sparse_rank"],
                matched_in_dense=record["matched_in_dense"],
                matched_in_sparse=record["matched_in_sparse"],
                hybrid_score=float(record["hybrid_score"]),
            )
            for record in merged.values()
        ]

        fused_results.sort(
            key=lambda item: (
                -item.hybrid_score,
                -_matched_source_count(item),
                _best_rank(item),
                item.child_chunk_id,
            )
        )
        return fused_results[:top_k]

    def _merge_source(
        self,
        *,
        merged: dict[str, dict[str, Any]],
        source_results: Sequence[ChildSearchResult],
        source_name: str,
    ) -> None:
        for rank, hit in enumerate(source_results, start=1):
            existing = merged.get(hit.child_chunk_id)
            if existing is None:
                existing = _init_merged_record(hit)
                merged[hit.child_chunk_id] = existing

            _merge_core_fields(existing=existing, incoming=hit)
            if source_name == "dense":
                existing["dense_score"] = hit.score
                existing["dense_rank"] = rank
                existing["matched_in_dense"] = True
            else:
                existing["sparse_score"] = hit.score
                existing["sparse_rank"] = rank
                existing["matched_in_sparse"] = True
            existing["hybrid_score"] += 1.0 / float(self.rrf_k + rank)


@dataclass(slots=True)
class HybridSearchService:
    """Runs dense+sparse retrieval and fuses ranks with RRF.

    ``search`` is intentionally thin: it orchestrates candidate retrieval and
    delegates fusion to :class:`RRFFuser`.
    """

    dense_service: DenseChildSearchService
    sparse_service: SparseChildSearchService
    fuser: RRFFuser = field(default_factory=RRFFuser)
    default_top_k: int = 10
    default_dense_top_k: int = 20
    default_sparse_top_k: int = 20
    max_candidate_pool: int = 500

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> list[HybridSearchResult]:
        if not query or not query.strip():
            return []
        effective_top_k = self.default_top_k if top_k is None else max(0, top_k)
        if effective_top_k <= 0:
            return []

        dense_limit = _resolve_candidate_limit(
            requested=dense_top_k,
            fallback=self.default_dense_top_k,
            top_k=effective_top_k,
            max_limit=self.max_candidate_pool,
        )
        sparse_limit = _resolve_candidate_limit(
            requested=sparse_top_k,
            fallback=self.default_sparse_top_k,
            top_k=effective_top_k,
            max_limit=self.max_candidate_pool,
        )

        normalized_query = query.strip()
        logger.info(
            "hybrid_search_started query_length=%s top_k=%s dense_top_k=%s sparse_top_k=%s",
            len(normalized_query),
            effective_top_k,
            dense_limit,
            sparse_limit,
        )

        dense_hits: list[ChildSearchResult] = []
        sparse_hits: list[ChildSearchResult] = []
        dense_error = False
        sparse_error = False

        try:
            dense_hits = self.dense_service.search(
                query=normalized_query,
                filters=filters,
                limit=dense_limit,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            dense_error = True
            logger.warning("hybrid_search_dense_failed error=%s", exc)

        try:
            sparse_hits = self.sparse_service.search(
                query=normalized_query,
                filters=filters,
                limit=sparse_limit,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            sparse_error = True
            logger.warning("hybrid_search_sparse_failed error=%s", exc)

        logger.info(
            "hybrid_search_candidates dense_count=%s sparse_count=%s",
            len(dense_hits),
            len(sparse_hits),
        )

        if dense_error and sparse_error:
            logger.warning("hybrid_search_both_sources_failed")
            return []

        fused = self.fuser.fuse(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            top_k=effective_top_k,
        )
        logger.info(
            "hybrid_search_completed fused_unique_count=%s returned_count=%s",
            len({item.child_chunk_id for item in fused}),
            len(fused),
        )
        return fused


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
        top_k: int = 10,
    ) -> list[HybridSearchResult]:
        """Tool: fuse vector and keyword retrieval for legal-domain recall + precision."""

        if self.keyword_search_service is None:
            return []
        hybrid_service = HybridSearchService(
            dense_service=DenseChildSearchService(vector_service=VectorSearchService(self.child_searcher)),
            sparse_service=SparseChildSearchService(keyword_service=self.keyword_search_service),
        )
        return hybrid_service.search(query=query, filters=filters, top_k=top_k)

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


def hybrid_search(
    query: str,
    *,
    dense_service: DenseChildSearchService,
    sparse_service: SparseChildSearchService,
    filters: Mapping[str, Any] | None = None,
    top_k: int = 10,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    rrf_k: int = 60,
    max_candidate_pool: int = 500,
) -> list[HybridSearchResult]:
    """Run child-chunk hybrid retrieval with deterministic Reciprocal Rank Fusion.

    Dense and sparse retrieval scores are not directly comparable in legal RAG;
    RRF fuses rank positions from both sources into one stable final list.
    """

    service = HybridSearchService(
        dense_service=dense_service,
        sparse_service=sparse_service,
        fuser=RRFFuser(rrf_k=rrf_k),
        default_top_k=top_k,
        default_dense_top_k=20,
        default_sparse_top_k=20,
        max_candidate_pool=max_candidate_pool,
    )
    return service.search(
        query=query,
        filters=filters,
        top_k=top_k,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
    )


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


def _resolve_candidate_limit(
    *,
    requested: int | None,
    fallback: int,
    top_k: int,
    max_limit: int,
) -> int:
    limit = fallback if requested is None else requested
    limit = max(limit, top_k)
    return min(limit, max_limit)


def _dedupe_by_child_chunk_id(results: Sequence[ChildSearchResult]) -> list[ChildSearchResult]:
    deduped: dict[str, ChildSearchResult] = {}
    for result in results:
        if result.child_chunk_id not in deduped:
            deduped[result.child_chunk_id] = result
    return list(deduped.values())


def _init_merged_record(hit: ChildSearchResult) -> dict[str, Any]:
    return {
        "child_chunk_id": hit.child_chunk_id,
        "parent_chunk_id": hit.parent_chunk_id,
        "document_id": hit.document_id,
        "text": hit.text,
        "metadata": dict(hit.payload),
        "dense_score": None,
        "sparse_score": None,
        "dense_rank": None,
        "sparse_rank": None,
        "matched_in_dense": False,
        "matched_in_sparse": False,
        "hybrid_score": 0.0,
    }


def _merge_core_fields(*, existing: dict[str, Any], incoming: ChildSearchResult) -> None:
    # Deterministic merge rule:
    # 1) Keep first-seen identifiers/text unless missing.
    # 2) Merge metadata dictionaries with first-seen keys preserved.
    if not existing["parent_chunk_id"] and incoming.parent_chunk_id:
        existing["parent_chunk_id"] = incoming.parent_chunk_id
    if not existing["document_id"] and incoming.document_id:
        existing["document_id"] = incoming.document_id
    if not existing["text"] and incoming.text:
        existing["text"] = incoming.text

    incoming_metadata = dict(incoming.payload)
    metadata = dict(incoming_metadata)
    metadata.update(existing["metadata"])
    existing["metadata"] = metadata


def _matched_source_count(item: HybridSearchResult) -> int:
    return int(item.matched_in_dense) + int(item.matched_in_sparse)


def _best_rank(item: HybridSearchResult) -> int:
    ranks = [rank for rank in (item.dense_rank, item.sparse_rank) if rank is not None]
    return min(ranks) if ranks else 10**9


def _extract_original_score(
    chunk: ChildSearchResult | HybridSearchResult | RerankedChunkResult,
) -> float:
    if isinstance(chunk, HybridSearchResult):
        return chunk.hybrid_score
    if isinstance(chunk, RerankedChunkResult):
        return chunk.rerank_score
    return chunk.score
