"""Context-compression tools for post-retrieval legal parent chunks.

Compression is applied after parent retrieval and before answer generation to
reduce token usage while preserving legal fidelity. The default strategy is
extractive and deterministic: it keeps original paragraphs (not paraphrases),
retains heading/section metadata, and preserves parent-level traceability.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class CompressedParentChunk:
    """Compressed, traceable representation of one retrieved parent chunk."""

    parent_chunk_id: str
    document_id: str
    source: str
    source_name: str
    heading_path: tuple[str, ...] = ()
    heading_text: str = ""
    parent_order: int = 0
    part_number: int = 1
    total_parts: int = 1
    compressed_text: str = ""
    original_char_count: int = 0
    compressed_char_count: int = 0


@dataclass(slots=True, frozen=True)
class CompressContextResult:
    """Structured output for compressed legal context packages.

    A typed result keeps the compression step production-safe and makes
    downstream answer-context assembly explicit and traceable by parent id.
    """

    items: tuple[CompressedParentChunk, ...] = ()
    total_original_chars: int = 0
    total_compressed_chars: int = 0


@dataclass(slots=True)
class ParentChunkCompressor:
    """Deterministic extractive compressor tuned for legal parent chunks.

    Why extractive-first for legal RAG:
    - legal clauses are sensitive to qualifiers (e.g., "unless", "except")
    - paraphrasing can distort obligations and carve-outs
    - preserving source wording improves faithfulness and auditability

    The compressor intentionally keeps more text when uncertainty exists.
    """

    small_chunk_char_threshold: int = 900
    medium_chunk_char_threshold: int = 2200
    conservative_target_ratio: float = 0.75
    large_chunk_target_ratio: float = 0.6
    legal_keywords: tuple[str, ...] = (
        "shall",
        "must",
        "may not",
        "except",
        "unless",
        "subject to",
        "provided that",
        "termination",
        "liability",
        "confidential",
        "confidentiality",
        "governing law",
        "jurisdiction",
        "indemn",
        "remedy",
        "damages",
        "breach",
        "means",
        "defined as",
        "definition",
    )
    critical_qualifier_patterns: tuple[re.Pattern[str], ...] = field(
        default_factory=lambda: (
            re.compile(r"\b(unless|except|subject to|provided that|notwithstanding|only if|conditioned upon)\b", re.I),
            re.compile(r"\b(shall not|may not|is not liable|limitation of liability)\b", re.I),
        )
    )

    def compress(self, parent_chunks: Sequence[object]) -> CompressContextResult:
        """Compress retrieved parent chunks while preserving legal meaning.

        Parent boundaries are preserved exactly as received. If compression fails,
        the original chunks are returned in structured form.
        """

        if not parent_chunks:
            return CompressContextResult()

        try:
            compressed_items = [self._compress_single(chunk) for chunk in parent_chunks]
        except Exception:
            compressed_items = [self._fallback_original(chunk) for chunk in parent_chunks]

        total_original = sum(item.original_char_count for item in compressed_items)
        total_compressed = sum(item.compressed_char_count for item in compressed_items)
        return CompressContextResult(
            items=tuple(compressed_items),
            total_original_chars=total_original,
            total_compressed_chars=total_compressed,
        )

    def _compress_single(self, chunk: object) -> CompressedParentChunk:
        parent_chunk_id = _field(chunk, "parent_chunk_id")
        document_id = _field(chunk, "document_id")
        source = _field(chunk, "source")
        source_name = _field(chunk, "source_name")
        heading_text = _field(chunk, "heading_text")
        heading_path = _tuple_str(_field(chunk, "heading_path", ()))
        parent_order = int(_field(chunk, "parent_order", 0) or 0)
        part_number = int(_field(chunk, "part_number", 1) or 1)
        total_parts = int(_field(chunk, "total_parts", 1) or 1)

        text = _field(chunk, "text")
        original_char_count = len(text)

        if not text.strip():
            return CompressedParentChunk(
                parent_chunk_id=parent_chunk_id,
                document_id=document_id,
                source=source,
                source_name=source_name,
                heading_path=heading_path,
                heading_text=heading_text,
                parent_order=parent_order,
                part_number=part_number,
                total_parts=total_parts,
                compressed_text=text,
                original_char_count=original_char_count,
                compressed_char_count=original_char_count,
            )

        if original_char_count <= self.small_chunk_char_threshold:
            return self._build_item(
                parent_chunk_id=parent_chunk_id,
                document_id=document_id,
                source=source,
                source_name=source_name,
                heading_path=heading_path,
                heading_text=heading_text,
                parent_order=parent_order,
                part_number=part_number,
                total_parts=total_parts,
                original_text=text,
                compressed_text=text,
            )

        paragraphs = _split_paragraphs(text)
        if len(paragraphs) <= 1:
            return self._build_item(
                parent_chunk_id=parent_chunk_id,
                document_id=document_id,
                source=source,
                source_name=source_name,
                heading_path=heading_path,
                heading_text=heading_text,
                parent_order=parent_order,
                part_number=part_number,
                total_parts=total_parts,
                original_text=text,
                compressed_text=text,
            )

        ratio = (
            self.large_chunk_target_ratio
            if original_char_count > self.medium_chunk_char_threshold
            else self.conservative_target_ratio
        )
        target_chars = max(int(original_char_count * ratio), self.small_chunk_char_threshold)

        must_keep = {0}
        for idx, paragraph in enumerate(paragraphs):
            if self._contains_critical_qualifier(paragraph):
                must_keep.add(idx)

        ranked_indices = sorted(
            range(len(paragraphs)),
            key=lambda i: (-self._paragraph_score(paragraphs[i]), i),
        )

        selected_indices = set(must_keep)
        for idx in ranked_indices:
            selected_indices.add(idx)
            selected_text = _join_selected_paragraphs(paragraphs, selected_indices)
            if len(selected_text) >= target_chars:
                break

        if not selected_indices:
            selected_indices.add(0)

        compressed_text = _join_selected_paragraphs(paragraphs, selected_indices)
        if len(compressed_text) >= original_char_count:
            compressed_text = text

        return self._build_item(
            parent_chunk_id=parent_chunk_id,
            document_id=document_id,
            source=source,
            source_name=source_name,
            heading_path=heading_path,
            heading_text=heading_text,
            parent_order=parent_order,
            part_number=part_number,
            total_parts=total_parts,
            original_text=text,
            compressed_text=compressed_text,
        )

    def _build_item(
        self,
        *,
        parent_chunk_id: str,
        document_id: str,
        source: str,
        source_name: str,
        heading_path: tuple[str, ...],
        heading_text: str,
        parent_order: int,
        part_number: int,
        total_parts: int,
        original_text: str,
        compressed_text: str,
    ) -> CompressedParentChunk:
        return CompressedParentChunk(
            parent_chunk_id=parent_chunk_id,
            document_id=document_id,
            source=source,
            source_name=source_name,
            heading_path=heading_path,
            heading_text=heading_text,
            parent_order=parent_order,
            part_number=part_number,
            total_parts=total_parts,
            compressed_text=compressed_text,
            original_char_count=len(original_text),
            compressed_char_count=len(compressed_text),
        )

    def _fallback_original(self, chunk: object) -> CompressedParentChunk:
        """Safe fallback preserving original text and metadata on failures."""

        text = _field(chunk, "text")
        return CompressedParentChunk(
            parent_chunk_id=_field(chunk, "parent_chunk_id"),
            document_id=_field(chunk, "document_id"),
            source=_field(chunk, "source"),
            source_name=_field(chunk, "source_name"),
            heading_path=_tuple_str(_field(chunk, "heading_path", ())),
            heading_text=_field(chunk, "heading_text"),
            parent_order=int(_field(chunk, "parent_order", 0) or 0),
            part_number=int(_field(chunk, "part_number", 1) or 1),
            total_parts=int(_field(chunk, "total_parts", 1) or 1),
            compressed_text=text,
            original_char_count=len(text),
            compressed_char_count=len(text),
        )

    def _contains_critical_qualifier(self, paragraph: str) -> bool:
        return any(pattern.search(paragraph) for pattern in self.critical_qualifier_patterns)

    def _paragraph_score(self, paragraph: str) -> float:
        lowered = paragraph.lower()
        word_count = len(lowered.split())
        score = min(float(word_count) / 80.0, 1.0)

        for keyword in self.legal_keywords:
            if keyword in lowered:
                score += 0.35

        if " means " in f" {lowered} " or "defined as" in lowered:
            score += 0.4
        if re.search(r"\b\d{1,2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", lowered):
            score += 0.2

        if word_count <= 35 and self._contains_critical_qualifier(paragraph):
            score += 0.5

        return score


_DEFAULT_CONTEXT_COMPRESSOR = ParentChunkCompressor()


def compress_context(parent_chunks: Sequence[object]) -> CompressContextResult:
    """Compress retrieved parent chunks into a faithful, traceable context pack.

    This tool is called after parent retrieval when context is large/noisy.
    It intentionally avoids retrieval and answer synthesis responsibilities.
    """

    return _DEFAULT_CONTEXT_COMPRESSOR.compress(parent_chunks)


def _field(item: object, key: str, default: str | int | Sequence[str] = "") -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _tuple_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(part) for part in value)
    return ()


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not parts:
        normalized = text.strip()
        return [normalized] if normalized else []
    return parts


def _join_selected_paragraphs(paragraphs: Sequence[str], selected_indices: set[int]) -> str:
    ordered = [paragraphs[idx] for idx in range(len(paragraphs)) if idx in selected_indices]
    return "\n\n".join(ordered)
