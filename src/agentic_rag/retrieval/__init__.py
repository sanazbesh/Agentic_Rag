"""Retrieval interfaces and parent-child retrieval services."""

from .interfaces import QueryRewriter, Retriever, Reranker
from .parent_child import (
    ChunkReranker,
    ChildChunkSearcher,
    ChildSearchResult,
    ChildChunkRepository,
    DenseChildSearchService,
    HybridSearchResult,
    HybridSearchService,
    InMemoryChildChunkRepository,
    InMemoryKeywordChunkRepository,
    InMemoryParentChunkRepository,
    KeywordChunkRepository,
    KeywordSearchService,
    ParentChildRetrievalTools,
    ParentChunkRepository,
    ParentChunkResult,
    ParentChunkStore,
    RRFFuser,
    RerankedChunkResult,
    SparseChildSearchService,
    VectorSearchService,
    hybrid_search,
)
from .sparse import SparseSearchResult, SparseSearchService, search_child_chunks_sparse

try:
    from .postgres_chunk_repository import PersistedChunk, PostgresChunkRepository
except Exception:  # pragma: no cover - optional SQLAlchemy dependency in some test envs
    PersistedChunk = None  # type: ignore[assignment]
    PostgresChunkRepository = None  # type: ignore[assignment]

__all__ = [
    "QueryRewriter",
    "Retriever",
    "Reranker",
    "ChildChunkRepository",
    "ParentChunkRepository",
    "ChildChunkSearcher",
    "VectorSearchService",
    "DenseChildSearchService",
    "KeywordChunkRepository",
    "KeywordSearchService",
    "SparseChildSearchService",
    "RRFFuser",
    "HybridSearchService",
    "ChunkReranker",
    "ParentChunkStore",
    "ParentChildRetrievalTools",
    "ChildSearchResult",
    "HybridSearchResult",
    "RerankedChunkResult",
    "ParentChunkResult",
    "InMemoryChildChunkRepository",
    "InMemoryKeywordChunkRepository",
    "InMemoryParentChunkRepository",
    "hybrid_search",
    "SparseSearchResult",
    "SparseSearchService",
    "search_child_chunks_sparse",
]

if PersistedChunk is not None and PostgresChunkRepository is not None:
    __all__.extend(["PersistedChunk", "PostgresChunkRepository"])
