"""Chunking interfaces for parent-child markdown chunk pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_rag.types import Document
from agentic_rag.chunking.models import ChunkingResult


class Chunker(ABC):
    """Transforms documents into parent/child chunk structures."""

    @abstractmethod
    def chunk(self, document: Document) -> ChunkingResult:
        """Chunk a document into parent and child retrieval units."""
