from __future__ import annotations

from agentic_rag.retrieval import ParentChunkResult
from agentic_rag.tools import CompressContextResult, compress_context


def _long_termination_text() -> str:
    return "\n\n".join(
        [
            "Either party may terminate this Agreement for material breach.",
            "The non-breaching party shall provide written notice and a thirty (30) day cure period.",
            "Termination is subject to Section 14 and except as otherwise required by applicable law.",
            "Unless prohibited by statute, either party may seek injunctive relief for ongoing violations.",
            "This Agreement remains in effect until terminated in accordance with this Section.",
            "Headings are for convenience only and shall not affect interpretation.",
            "Termination for insolvency is permitted if proceedings continue for more than sixty (60) days.",
            "Any accrued payment obligations survive termination.",
            "Except for confidentiality and indemnity obligations, duties cease upon effective termination.",
            "Notwithstanding the foregoing, remedies for prior breach remain available.",
            "A party seeking post-termination relief must document losses with reasonable specificity and submit evidence within ninety (90) days.",
            "No waiver of any breach is effective unless in writing and signed by an authorized representative of the non-waiving party.",
            "Termination rights are cumulative and in addition to any rights available at law or in equity, subject to this Agreement's limitations.",
        ]
    )


def _parent(parent_id: str, heading: str, text: str, order: int = 0) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text=text,
        source=f"s3://bucket/{parent_id}.md",
        source_name=f"{parent_id}.md",
        heading_path=("Master Services Agreement", heading),
        heading_text=heading,
        parent_order=order,
        part_number=1,
        total_parts=1,
    )


def test_compress_context_returns_structured_output() -> None:
    result = compress_context([_parent("p-1", "Termination", "Short legal clause.")])

    assert isinstance(result, CompressContextResult)
    assert len(result.items) == 1


def test_compress_context_preserves_parent_and_source_metadata() -> None:
    parent = _parent("p-1", "Termination", "Either party may terminate for material breach.")

    result = compress_context([parent])
    item = result.items[0]

    assert item.parent_chunk_id == "p-1"
    assert item.document_id == "doc-1"
    assert item.source == "s3://bucket/p-1.md"
    assert item.source_name == "p-1.md"


def test_compress_context_handles_empty_input_safely() -> None:
    result = compress_context([])

    assert result.items == ()
    assert result.total_original_chars == 0
    assert result.total_compressed_chars == 0


def test_compress_context_handles_single_parent_chunk() -> None:
    result = compress_context([_parent("p-1", "Confidentiality", "Each party must keep information confidential.")])

    assert len(result.items) == 1
    assert result.items[0].compressed_text


def test_compress_context_handles_multiple_parent_chunks_without_merging() -> None:
    parents = [
        _parent("p-1", "Termination", "Either party may terminate for breach.", order=0),
        _parent("p-2", "Confidentiality", "Each party shall keep Confidential Information secret.", order=1),
    ]

    result = compress_context(parents)

    assert [item.parent_chunk_id for item in result.items] == ["p-1", "p-2"]
    assert len(result.items) == 2


def test_compress_context_preserves_heading_context_when_present() -> None:
    result = compress_context([_parent("p-1", "Limitation of Liability", "Liability is capped at fees paid.")])

    item = result.items[0]
    assert item.heading_text == "Limitation of Liability"
    assert item.heading_path == ("Master Services Agreement", "Limitation of Liability")


def test_compress_context_reduces_size_for_long_parent_text() -> None:
    parent = _parent("p-1", "Termination", _long_termination_text())

    result = compress_context([parent])
    item = result.items[0]

    assert item.compressed_char_count < item.original_char_count


def test_compress_context_avoids_unnecessary_compression_for_short_text() -> None:
    text = "Each party must comply with this confidentiality clause unless disclosure is legally required."
    parent = _parent("p-1", "Confidentiality", text)

    result = compress_context([parent])

    assert result.items[0].compressed_text == text


def test_compress_context_is_deterministic_and_order_preserving() -> None:
    parents = [
        _parent("p-1", "Termination", _long_termination_text(), order=0),
        _parent("p-2", "Confidentiality", _long_termination_text(), order=1),
    ]

    first = compress_context(parents)
    second = compress_context(parents)

    assert first == second
    assert [item.parent_chunk_id for item in first.items] == ["p-1", "p-2"]


def test_compress_context_preserves_legally_important_qualifiers() -> None:
    text = "\n\n".join(
        [
            "The Customer shall pay all undisputed fees within thirty (30) days.",
            "Except as set forth in Section 9, the Supplier shall not disclose Confidential Information.",
            "The Supplier may disclose Confidential Information only if required by law and subject to prior notice.",
            "General business background statement for context.",
            "Unless otherwise required by governing law, remedies are limited to direct damages.",
            "Informational paragraph describing document layout.",
        ]
    )
    parent = _parent("p-legal", "Confidentiality", text)

    result = compress_context([parent])
    compressed = result.items[0].compressed_text.lower()

    assert "except as set forth" in compressed
    assert "subject to prior notice" in compressed
    assert "unless otherwise required" in compressed


def test_compress_context_output_compatible_with_downstream_context_assembly() -> None:
    parents = [
        _parent("p-1", "Termination", "Either party may terminate for material breach."),
        _parent("p-2", "Governing Law", "This Agreement is governed by New York law."),
    ]

    result = compress_context(parents)

    assembled = "\n\n".join(
        f"[{item.parent_chunk_id}] {item.heading_text}\n{item.compressed_text}" for item in result.items
    )

    assert "[p-1] Termination" in assembled
    assert "[p-2] Governing Law" in assembled
    assert "material breach" in assembled.lower()


def test_compress_context_is_idempotent_for_repeated_runs() -> None:
    parent = _parent("p-1", "Termination", _long_termination_text())

    first = compress_context([parent])

    rerun_input = [
        _parent(
            parent_id=item.parent_chunk_id,
            heading=item.heading_text,
            text=item.compressed_text,
        )
        for item in first.items
    ]
    second = compress_context(rerun_input)

    assert second.items[0].compressed_text == first.items[0].compressed_text
