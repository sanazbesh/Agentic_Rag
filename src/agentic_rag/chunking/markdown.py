"""Markdown-aware parent-child chunking.

This module intentionally focuses on chunking only:
- parent chunks are context units for downstream LLM answering
- child chunks are retrieval units prepared for later Qdrant upsert
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from agentic_rag.chunking.interfaces import Chunker
from agentic_rag.chunking.models import ChildChunk, ChunkingResult, ParentChunk
from agentic_rag.types import Document


@dataclass(slots=True)
class TokenCounter:
    """Token estimator using tiktoken when available with deterministic fallback."""

    encoding_name: str = "cl100k_base"
    _encoding: object | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._encoding = None
        try:  # pragma: no cover - fallback path tested implicitly when unavailable
            import tiktoken

            self._encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return len(_fallback_tokenize(text))

    def tail(self, text: str, token_count: int) -> str:
        if token_count <= 0 or not text:
            return ""
        if self._encoding is not None:
            ids = self._encoding.encode(text)
            if len(ids) <= token_count:
                return text
            return self._encoding.decode(ids[-token_count:])

        tokens = _fallback_tokenize(text)
        if len(tokens) <= token_count:
            return text
        return "".join(tokens[-token_count:])


@dataclass(slots=True)
class ParentChunker:
    """Build markdown-structure-aware parent chunks."""

    token_counter: TokenCounter
    target_tokens: int = 1200
    hard_cap_tokens: int = 2000

    def chunk(self, document: Document) -> list[ParentChunk]:
        source = str(document.metadata.get("source", "unknown"))
        source_name = str(document.metadata.get("source_name", source))

        sections = _parse_markdown_sections(document.text)
        parent_chunks: list[ParentChunk] = []
        parent_order = 0

        for section in sections:
            section_text = section.full_text
            if self.token_counter.count(section_text) <= self.hard_cap_tokens:
                parent_order += 1
                parent_chunks.append(
                    _make_parent_chunk(
                        document=document,
                        source=source,
                        source_name=source_name,
                        text=section_text,
                        heading_path=section.heading_path,
                        parent_order=parent_order,
                        part_number=1,
                        total_parts=1,
                        token_counter=self.token_counter,
                    )
                )
                continue

            split_parts = _split_large_parent_section(
                text=section.raw_body,
                heading_line=section.heading_line,
                token_counter=self.token_counter,
                max_tokens=self.hard_cap_tokens,
            )
            total_parts = len(split_parts)
            for idx, part_text in enumerate(split_parts, start=1):
                parent_order += 1
                parent_chunks.append(
                    _make_parent_chunk(
                        document=document,
                        source=source,
                        source_name=source_name,
                        text=part_text,
                        heading_path=section.heading_path,
                        parent_order=parent_order,
                        part_number=idx,
                        total_parts=total_parts,
                        token_counter=self.token_counter,
                    )
                )

        return parent_chunks


@dataclass(slots=True)
class RecursiveChildChunker:
    """Create retrieval child chunks with semantic-first recursive splitting."""

    token_counter: TokenCounter
    max_tokens: int = 300
    overlap_tokens: int = 30

    def chunk(self, parent_chunk: ParentChunk) -> list[ChildChunk]:
        base_limit = self.max_tokens - self.overlap_tokens
        base_segments = _recursive_split(
            text=parent_chunk.text,
            token_counter=self.token_counter,
            max_tokens=base_limit,
        )
        base_segments = _rebalance_leftovers(
            base_segments,
            token_counter=self.token_counter,
            max_tokens=base_limit,
        )
        _assert_full_coverage(parent_chunk.text, base_segments)

        children: list[ChildChunk] = []
        for idx, segment in enumerate(base_segments):
            if idx == 0:
                text = segment
            else:
                overlap = self.token_counter.tail(base_segments[idx - 1], self.overlap_tokens)
                text = f"{overlap}{segment}"

            token_count = self.token_counter.count(text)
            if token_count > self.max_tokens:
                # deterministic emergency fallback, keeping no text loss guarantees in base segments
                text = _truncate_to_tokens(text, self.max_tokens, self.token_counter)
                token_count = self.token_counter.count(text)

            child_id = _stable_id(
                "child",
                parent_chunk.document_id,
                parent_chunk.parent_chunk_id,
                str(idx),
                text,
            )
            children.append(
                ChildChunk(
                    child_chunk_id=child_id,
                    parent_chunk_id=parent_chunk.parent_chunk_id,
                    document_id=parent_chunk.document_id,
                    source=parent_chunk.source,
                    source_name=parent_chunk.source_name,
                    text=text,
                    child_order=idx + 1,
                    token_count=token_count,
                    heading_path=parent_chunk.heading_path,
                )
            )

        return children


@dataclass(slots=True)
class MarkdownParentChildChunker(Chunker):
    """End-to-end markdown parent-child chunking for modular RAG pipelines."""

    parent_target_tokens: int = 1200
    parent_hard_cap_tokens: int = 2000
    child_max_tokens: int = 300
    child_overlap_tokens: int = 30

    def __post_init__(self) -> None:
        token_counter = TokenCounter()
        self._parent_chunker = ParentChunker(
            token_counter=token_counter,
            target_tokens=self.parent_target_tokens,
            hard_cap_tokens=self.parent_hard_cap_tokens,
        )
        self._child_chunker = RecursiveChildChunker(
            token_counter=token_counter,
            max_tokens=self.child_max_tokens,
            overlap_tokens=self.child_overlap_tokens,
        )

    def chunk(self, document: Document) -> ChunkingResult:
        parent_chunks = self._parent_chunker.chunk(document)
        child_chunks: list[ChildChunk] = []
        for parent in parent_chunks:
            child_chunks.extend(self._child_chunker.chunk(parent))
        return ChunkingResult(parent_chunks=parent_chunks, child_chunks=child_chunks)


@dataclass(slots=True)
class _MarkdownSection:
    heading_line: str
    heading_path: tuple[str, ...]
    raw_body: str

    @property
    def full_text(self) -> str:
        if self.heading_line:
            if self.raw_body:
                return f"{self.heading_line}\n{self.raw_body}"
            return self.heading_line
        return self.raw_body


def _parse_markdown_sections(text: str) -> list[_MarkdownSection]:
    lines = text.splitlines(keepends=True)
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

    sections: list[_MarkdownSection] = []
    current_heading_line = ""
    current_heading_path: tuple[str, ...] = ()
    current_body: list[str] = []
    stack: list[tuple[int, str]] = []

    def flush() -> None:
        if current_heading_line or current_body:
            sections.append(
                _MarkdownSection(
                    heading_line=current_heading_line,
                    heading_path=current_heading_path,
                    raw_body="".join(current_body),
                )
            )

    for line in lines:
        match = heading_pattern.match(line.rstrip("\n"))
        if match:
            flush()
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))

            current_heading_path = tuple(part for _, part in stack)
            current_heading_line = line.rstrip("\n")
            current_body = []
        else:
            current_body.append(line)

    flush()

    if not sections:
        return [_MarkdownSection(heading_line="", heading_path=(), raw_body=text)]
    return _coalesce_document_start_preamble(sections)


def _coalesce_document_start_preamble(sections: list[_MarkdownSection]) -> list[_MarkdownSection]:
    """Preserve agreement-intro preambles as runtime-visible parent context.

    Some PDF→Markdown converters emit preamble lines (effective-date/party definitions)
    as multiple small heading sections before the first numbered section heading.
    This coalesces those leading fragments into one opening section so parent chunk
    construction keeps the full intro block intact.
    """

    if len(sections) < 2:
        return sections

    first_numbered_idx: int | None = None
    for idx, section in enumerate(sections):
        heading = _heading_text(section.heading_line)
        if heading and _is_numbered_section_heading(heading):
            first_numbered_idx = idx
            break

    if first_numbered_idx is None or first_numbered_idx <= 1:
        return sections

    leading = sections[:first_numbered_idx]
    if not _looks_like_agreement_intro("".join(part.full_text for part in leading)):
        return sections

    first = leading[0]
    if first.heading_line:
        heading_line = first.heading_line
        heading_path = first.heading_path
        raw_body_parts = [first.raw_body]
        raw_body_parts.extend(part.full_text for part in leading[1:])
        raw_body = "".join(raw_body_parts).lstrip("\n")
    else:
        heading_line = ""
        heading_path = ()
        raw_body = "".join(part.full_text for part in leading)

    merged_intro = _MarkdownSection(
        heading_line=heading_line,
        heading_path=heading_path,
        raw_body=raw_body,
    )
    return [merged_intro, *sections[first_numbered_idx:]]


def _heading_text(heading_line: str) -> str:
    if not heading_line:
        return ""
    return re.sub(r"^#{1,6}\s*", "", heading_line).strip()


def _is_numbered_section_heading(heading: str) -> bool:
    normalized = heading.strip().lower()
    return bool(re.match(r"^(section\s+)?\d{1,3}(?:\.\d+)*[\).:\-]?\s+\S", normalized))


def _looks_like_agreement_intro(text: str) -> bool:
    lowered = text.lower()
    intro_markers = (
        "agreement",
        "between",
        "by and between",
        "effective",
        "parties to this agreement",
        "employer",
        "employee",
    )
    return any(marker in lowered for marker in intro_markers)


def _split_large_parent_section(
    text: str,
    heading_line: str,
    token_counter: TokenCounter,
    max_tokens: int,
) -> list[str]:
    effective_limit = max_tokens - token_counter.count(f"{heading_line}\n") if heading_line else max_tokens
    parts = _recursive_split(text, token_counter, max(100, effective_limit))

    out: list[str] = []
    for part in parts:
        if heading_line:
            out.append(f"{heading_line}\n{part}")
        else:
            out.append(part)
    return out


def _recursive_split(text: str, token_counter: TokenCounter, max_tokens: int) -> list[str]:
    if token_counter.count(text) <= max_tokens:
        return [text]

    splitters = (
        _split_by_markdown_boundaries,
        _split_by_paragraphs,
        _split_by_sentences,
    )

    for splitter in splitters:
        parts = splitter(text)
        if len(parts) <= 1:
            continue
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + part
            if current and token_counter.count(candidate) > max_tokens:
                chunks.extend(_recursive_split(current, token_counter, max_tokens))
                current = part
            else:
                current = candidate
        if current:
            chunks.extend(_recursive_split(current, token_counter, max_tokens))

        if chunks:
            return chunks

    return _token_fallback_split(text, token_counter, max_tokens)


def _split_by_markdown_boundaries(text: str) -> list[str]:
    pattern = re.compile(r"(?m)(?=^#{1,6}\s+)")
    points = [m.start() for m in pattern.finditer(text) if m.start() > 0]
    return _split_by_points(text, points)


def _split_by_paragraphs(text: str) -> list[str]:
    points = [m.end() for m in re.finditer(r"\n\s*\n", text)]
    return _split_by_points(text, points)


def _split_by_sentences(text: str) -> list[str]:
    points = [m.end() for m in re.finditer(r"(?<=[.!?])\s+", text)]
    return _split_by_points(text, points)


def _split_by_points(text: str, points: list[int]) -> list[str]:
    if not points:
        return [text]
    out: list[str] = []
    start = 0
    for point in points:
        out.append(text[start:point])
        start = point
    out.append(text[start:])
    return [piece for piece in out if piece]


def _token_fallback_split(text: str, token_counter: TokenCounter, max_tokens: int) -> list[str]:
    if token_counter._encoding is not None:  # noqa: SLF001
        ids = token_counter._encoding.encode(text)  # noqa: SLF001
        parts = [token_counter._encoding.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]  # noqa: SLF001,E203
        return [part for part in parts if part]

    tokens = _fallback_tokenize(text)
    parts = ["".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return [part for part in parts if part]


def _rebalance_leftovers(
    segments: list[str],
    token_counter: TokenCounter,
    max_tokens: int,
    tiny_threshold: int = 25,
) -> list[str]:
    if not segments:
        return []

    result: list[str] = []
    for segment in segments:
        if not result:
            result.append(segment)
            continue

        seg_tokens = token_counter.count(segment)
        if seg_tokens <= tiny_threshold:
            prev = result[-1]
            if token_counter.count(prev + segment) <= max_tokens + 10:
                result[-1] = prev + segment
            else:
                result.append(segment)
        else:
            result.append(segment)

    if len(result) > 1 and token_counter.count(result[0]) <= tiny_threshold:
        candidate = result[0] + result[1]
        if token_counter.count(candidate) <= max_tokens + 10:
            result[1] = candidate
            result = result[1:]

    return result


def _assert_full_coverage(source_text: str, segments: list[str]) -> None:
    joined = "".join(segments)
    if joined != source_text:
        raise ValueError("Child chunking must preserve full text without loss or reordering.")


def _make_parent_chunk(
    document: Document,
    source: str,
    source_name: str,
    text: str,
    heading_path: tuple[str, ...],
    parent_order: int,
    part_number: int,
    total_parts: int,
    token_counter: TokenCounter,
) -> ParentChunk:
    parent_id = _stable_id("parent", document.id, str(parent_order), text)
    heading_text = heading_path[-1] if heading_path else ""
    token_count = token_counter.count(text)
    return ParentChunk(
        parent_chunk_id=parent_id,
        document_id=document.id,
        source=source,
        source_name=source_name,
        text=text,
        heading_path=heading_path,
        heading_text=heading_text,
        parent_order=parent_order,
        parent_token_count=token_count,
        original_heading_context=heading_path,
        part_number=part_number,
        total_parts=total_parts,
    )


def _stable_id(prefix: str, *parts: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(prefix.encode("utf-8"))
    for part in parts:
        hasher.update(b"\x00")
        hasher.update(part.encode("utf-8"))
    return hasher.hexdigest()


def _fallback_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|\s+|[^\w\s]", text, flags=re.UNICODE)


def _truncate_to_tokens(text: str, max_tokens: int, token_counter: TokenCounter) -> str:
    if token_counter._encoding is not None:  # noqa: SLF001
        ids = token_counter._encoding.encode(text)  # noqa: SLF001
        return token_counter._encoding.decode(ids[:max_tokens])  # noqa: SLF001
    return "".join(_fallback_tokenize(text)[:max_tokens])
