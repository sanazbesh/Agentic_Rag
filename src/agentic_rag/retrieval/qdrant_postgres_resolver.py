"""Resolve Qdrant child hit payloads to persisted Postgres chunks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag.retrieval.parent_child import ChildChunkRepository, ChildSearchResult


class PersistedChunkLike(Protocol):
    chunk_id: str
    document_id: str
    document_version_id: str
    parent_chunk_id: str | None
    qdrant_point_id: str
    text: str
    metadata: Mapping[str, Any]


class ChunkResolverRepository(Protocol):
    def get_chunk_by_id(self, chunk_id: str) -> PersistedChunkLike | None: ...

    def resolve_qdrant_point_id(self, point_id: str) -> PersistedChunkLike | None: ...


class QdrantSearchBackend(Protocol):
    """Minimal backend protocol used for dense child chunk search."""

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[Mapping[str, Any]]: ...


@dataclass(slots=True)
class QdrantResultResolver:
    """Transforms raw Qdrant hits into retrieval-ready child chunk results."""

    chunk_repository: ChunkResolverRepository

    def resolve(
        self,
        raw_hits: Sequence[Mapping[str, Any]],
        *,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ChildSearchResult]:
        resolved: list[ChildSearchResult] = []
        for raw_hit in raw_hits:
            chunk = self._resolve_chunk(raw_hit)
            if chunk is None:
                continue

            payload = _payload(raw_hit)
            merged_payload = _merged_payload(raw_payload=payload, chunk=chunk)
            if not _passes_filters(merged_payload, filters):
                continue

            resolved.append(
                ChildSearchResult(
                    child_chunk_id=chunk.chunk_id,
                    parent_chunk_id=chunk.parent_chunk_id or "",
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=_extract_score(raw_hit),
                    payload=merged_payload,
                )
            )
        return resolved

    def _resolve_chunk(self, raw_hit: Mapping[str, Any]) -> PersistedChunkLike | None:
        payload = _payload(raw_hit)
        child_chunk_id = str(payload.get("chunk_id") or payload.get("child_chunk_id") or "").strip()
        if child_chunk_id:
            return self.chunk_repository.get_chunk_by_id(child_chunk_id)

        qdrant_point_id = str(payload.get("qdrant_point_id") or raw_hit.get("id") or "").strip()
        if not qdrant_point_id:
            return None
        return self.chunk_repository.resolve_qdrant_point_id(qdrant_point_id)


@dataclass(slots=True)
class PostgresResolvedQdrantChildRepository(ChildChunkRepository):
    """Child chunk repository that resolves Qdrant hits to persisted Postgres rows."""

    qdrant_backend: QdrantSearchBackend
    resolver: QdrantResultResolver
    search_mapper: Callable[[str], str] | None = None

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
    ) -> list[ChildSearchResult]:
        effective_query = self.search_mapper(query) if self.search_mapper is not None else query
        raw_hits = self.qdrant_backend.search(effective_query, filters=filters, limit=limit)
        return self.resolver.resolve(raw_hits, filters=filters)


def _payload(hit: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = hit.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return {}


def _extract_score(hit: Mapping[str, Any]) -> float:
    score = hit.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _merged_payload(*, raw_payload: Mapping[str, Any], chunk: PersistedChunkLike) -> dict[str, Any]:
    merged = dict(raw_payload)
    merged.setdefault("chunk_id", chunk.chunk_id)
    merged.setdefault("child_chunk_id", chunk.chunk_id)
    merged.setdefault("document_id", chunk.document_id)
    merged.setdefault("document_version_id", chunk.document_version_id)
    if chunk.parent_chunk_id:
        merged.setdefault("parent_chunk_id", chunk.parent_chunk_id)
    if chunk.qdrant_point_id:
        merged.setdefault("qdrant_point_id", chunk.qdrant_point_id)

    for key, value in chunk.metadata.items():
        merged.setdefault(key, value)
    return merged


def _passes_filters(payload: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    if not filters:
        return True

    selected_document_ids = _as_string_set(filters.get("selected_document_ids"))
    if selected_document_ids and str(payload.get("document_id", "")) not in selected_document_ids:
        return False

    selected_version_ids = _as_string_set(
        filters.get("selected_document_version_ids") or filters.get("selected_version_ids")
    )
    if selected_version_ids and str(payload.get("document_version_id", "")) not in selected_version_ids:
        return False

    for key, expected in filters.items():
        if key in {"selected_document_ids", "selected_document_version_ids", "selected_version_ids"}:
            continue
        if payload.get(key) != expected:
            return False
    return True


def _as_string_set(value: Any) -> set[str]:
    if isinstance(value, (str, bytes)):
        return {str(value)} if str(value) else set()
    if isinstance(value, Sequence):
        return {str(item) for item in value if str(item)}
    return set()


__all__ = [
    "QdrantSearchBackend",
    "QdrantResultResolver",
    "PostgresResolvedQdrantChildRepository",
]
