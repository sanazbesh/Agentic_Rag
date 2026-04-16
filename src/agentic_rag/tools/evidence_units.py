"""Canonical evidence-unit mapping for answerability and answer generation.

This module centralizes how retrieved parent/compressed chunks are normalized
into one deterministic evidence unit. Downstream components must consume this
mapping instead of re-deriving headings/text independently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class EvidenceUnit:
    """Canonical evidence unit used across attribution and strength analysis."""

    evidence_unit_id: str
    parent_chunk_id: str
    document_id: str
    source_name: str
    heading: str
    heading_path: tuple[str, ...]
    text: str
    evidence_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_evidence_units(context: Sequence[object]) -> list[EvidenceUnit]:
    """Normalize answer context into canonical evidence units.

    Canonical unit policy:
    - primary unit is a parent/compressed parent chunk
    - evidence text is exactly the text used for matching/strength scoring
    - heading attribution prefers section-level metadata before document title
    """

    units: list[EvidenceUnit] = []
    for index, item in enumerate(context):
        parent_chunk_id = str(_field(item, "parent_chunk_id") or "").strip()
        document_id = str(_field(item, "document_id") or "").strip()
        source_name = str(_field(item, "source_name") or "").strip()
        heading_path = _tuple_str(_field(item, "heading_path", ()))
        text = str(_field(item, "compressed_text") or _field(item, "text") or "")
        heading = _resolve_heading(
            heading_path=heading_path,
            heading_text=_field(item, "heading_text"),
            explicit_heading=_field(item, "heading"),
            section=_field(item, "section"),
            source_name=source_name,
        )

        stable_parent = parent_chunk_id or f"missing-parent-{index}"
        evidence_unit_id = f"{stable_parent}:{index}"
        units.append(
            EvidenceUnit(
                evidence_unit_id=evidence_unit_id,
                parent_chunk_id=parent_chunk_id,
                document_id=document_id,
                source_name=source_name,
                heading=heading,
                heading_path=heading_path,
                text=text,
                evidence_text=text,
                metadata=_mapping(_field(item, "metadata", {})),
            )
        )
    return units


def _resolve_heading(
    *,
    heading_path: tuple[str, ...],
    heading_text: Any,
    explicit_heading: Any,
    section: Any,
    source_name: str,
) -> str:
    if heading_path:
        most_specific = str(heading_path[-1]).strip()
        if most_specific:
            return most_specific
    for candidate in (heading_text, explicit_heading, section):
        value = str(candidate or "").strip()
        if value:
            return value
    return source_name


def _field(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(part) for part in value)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}
