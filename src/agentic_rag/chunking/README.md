# Markdown parent-child chunking

This module provides deterministic, structure-aware chunking for Markdown documents.

## Strategy

- **Parent chunks** are section-level context units for answer generation.
- **Child chunks** are smaller retrieval units linked to exactly one parent by `parent_chunk_id`.
- Retrieval should happen over child chunks, then parent chunks should be fetched for final LLM context.

## Parent chunking

- Parent chunking is Markdown-aware and starts from heading boundaries and heading hierarchy.
- Very large sections are split into multiple ordered parent parts.
- When a section is split, the same heading line is repeated at the top of each split parent chunk.
- This keeps each split self-contained and preserves section context downstream.

## Child chunking

Recursive split priority is:
1. Markdown structure boundaries
2. Paragraphs
3. Sentences
4. Token fallback

- Child chunks use overlap (`30` tokens default) for retrieval continuity.
- Child max size is `300` tokens.
- Every child references exactly one parent.

## No-text-loss guarantee

- Chunking verifies full text coverage by reconstructing base child segments and comparing against the original parent text.
- Leftover tiny fragments are first merged into the previous segment when possible, then into the next segment, otherwise kept as their own segment.
- Order is preserved and no text is silently dropped.

## Storage shapes

- Parent chunks are typed dataclasses and can be exported to storage-friendly records keyed by `parent_chunk_id`.
- Child chunks are typed dataclasses and can be exported to Qdrant-ready records (`id`, `text`, `payload`) with parent/document linkage metadata.
