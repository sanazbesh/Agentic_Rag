"""Shared domain types used across Agentic RAG modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class Document:
    """A canonical content unit used for ingestion and indexing."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A text span created from a document for retrieval indexing."""

    id: str
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedItem:
    """A retrieval result with optional ranking score."""

    chunk_id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Generation:
    """A generated response with optional structured diagnostics."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
