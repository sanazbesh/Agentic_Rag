"""Base interfaces for chunking, embedding, and index operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from agentic_rag.types import Chunk, Document


class Chunker(ABC):
    """Splits documents into retrieval-ready chunks."""

    @abstractmethod
    def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
        """Return chunks generated from document text."""


class Embedder(ABC):
    """Converts text into vector embeddings."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per input text."""


class VectorIndex(ABC):
    """Stores and queries vectorized chunk representations."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or update chunk vectors in the index."""

    @abstractmethod
    def search(self, query_embedding: Sequence[float], top_k: int = 5) -> list[str]:
        """Return chunk identifiers ranked by similarity."""
