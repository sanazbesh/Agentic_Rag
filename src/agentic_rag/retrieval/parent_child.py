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
