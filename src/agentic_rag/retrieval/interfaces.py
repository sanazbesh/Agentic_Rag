"""Base interfaces for query processing and retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from agentic_rag.types import RetrievedItem


class QueryRewriter(ABC):
    """Optional query transformation prior to retrieval."""

    @abstractmethod
    def rewrite(self, query: str) -> str:
        """Return an optimized retrieval query."""


class Retriever(ABC):
    """Finds relevant items for a query."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedItem]:
        """Return candidate retrieved items."""


class Reranker(ABC):
    """Re-scores retrieval candidates for better relevance."""

    @abstractmethod
    def rerank(self, query: str, items: Sequence[RetrievedItem]) -> list[RetrievedItem]:
        """Return items sorted by improved relevance."""
