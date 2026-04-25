from __future__ import annotations

from agentic_rag.chunking import MarkdownParentChildChunker
from agentic_rag.types import Document


def _doc(text: str) -> Document:
    return Document(
        id="doc-1",
        text=text,
        metadata={"source": "docs/guide.md", "source_name": "guide.md"},
    )


def test_parent_child_linkage_and_max_child_tokens() -> None:
    text = (
        "# Intro\n\n" + "Intro paragraph. " * 120 + "\n\n"
        "## Details\n\n" + "Details sentence. " * 200
    )
    result = MarkdownParentChildChunker().chunk(_doc(text))

    parent_ids = {parent.parent_chunk_id for parent in result.parent_chunks}
    assert parent_ids
    assert result.child_chunks
    assert all(child.parent_chunk_id in parent_ids for child in result.child_chunks)
    assert all(child.token_count <= 300 for child in result.child_chunks)


def test_parent_size_policy_and_markdown_structure_boundaries() -> None:
    text = "".join(
        [
            "# A\n\n" + ("alpha text. " * 120) + "\n\n",
            "## B\n\n" + ("beta text. " * 100) + "\n\n",
            "# C\n\n" + ("gamma text. " * 90),
        ]
    )
    chunker = MarkdownParentChildChunker()
    result = chunker.chunk(_doc(text))

    assert all(parent.parent_token_count <= 2000 for parent in result.parent_chunks)
    assert any(parent.heading_text == "A" for parent in result.parent_chunks)
    assert any(parent.heading_text == "B" for parent in result.parent_chunks)
    assert any(parent.heading_text == "C" for parent in result.parent_chunks)


def test_no_text_loss_and_order_preserved_per_parent() -> None:
    text = "# T\n\nOne. Two. Three.\n\nFour. Five.\n"
    chunker = MarkdownParentChildChunker()
    result = chunker.chunk(_doc(text))

    for parent in result.parent_chunks:
        children = [c for c in result.child_chunks if c.parent_chunk_id == parent.parent_chunk_id]
        assert children
        reconstructed = children[0].text
        for prev, nxt in zip(children, children[1:]):
            overlap = chunker._child_chunker.token_counter.tail(prev.text, 30)  # noqa: SLF001
            assert nxt.text.startswith(overlap)
            reconstructed += nxt.text[len(overlap) :]
        assert reconstructed == parent.text


def test_leftover_fragment_behavior() -> None:
    text = "# L\n\n" + ("A " * 500) + "tiny"
    chunker = MarkdownParentChildChunker()
    result = chunker.chunk(_doc(text))

    for parent in result.parent_chunks:
        children = [c for c in result.child_chunks if c.parent_chunk_id == parent.parent_chunk_id]
        assert children[-1].text.strip().endswith("tiny")


def test_determinism_same_input_same_ids() -> None:
    text = "# Stable\n\n" + ("repeated sentence. " * 200)
    chunker = MarkdownParentChildChunker()

    first = chunker.chunk(_doc(text))
    second = chunker.chunk(_doc(text))

    assert [p.parent_chunk_id for p in first.parent_chunks] == [p.parent_chunk_id for p in second.parent_chunks]
    assert [c.child_chunk_id for c in first.child_chunks] == [c.child_chunk_id for c in second.child_chunks]


def test_huge_section_split_repeats_heading_and_ordered_parts() -> None:
    text = "# Huge\n\n" + ("long body sentence. " * 4000)
    result = MarkdownParentChildChunker().chunk(_doc(text))

    parents = [p for p in result.parent_chunks if p.heading_text == "Huge"]
    assert len(parents) > 1
    assert [p.part_number for p in parents] == list(range(1, len(parents) + 1))
    assert all(p.total_parts == len(parents) for p in parents)
    assert all(p.text.startswith("# Huge") for p in parents)


def test_output_shapes_for_qdrant_and_parent_lookup() -> None:
    text = "# Out\n\nHello world"
    result = MarkdownParentChildChunker().chunk(_doc(text))

    parent_lookup = result.parent_lookup()
    assert set(parent_lookup.keys()) == {p.parent_chunk_id for p in result.parent_chunks}

    qdrant_records = result.child_qdrant_records()
    assert qdrant_records
    first = qdrant_records[0]
    assert {"id", "text", "payload"}.issubset(first.keys())
    assert "parent_chunk_id" in first["payload"]
    assert "document_id" in first["payload"]


def test_edge_cases_short_no_heading_single_parent_single_child() -> None:
    text = "A short doc without headings."
    result = MarkdownParentChildChunker().chunk(_doc(text))

    assert len(result.parent_chunks) == 1
    assert len(result.child_chunks) == 1
    assert result.child_chunks[0].parent_chunk_id == result.parent_chunks[0].parent_chunk_id


def test_edge_case_tiny_leftover_at_beginning() -> None:
    text = "Hi\n\n" + ("sentence. " * 800)
    result = MarkdownParentChildChunker().chunk(_doc(text))

    first_parent_children = [
        child
        for child in result.child_chunks
        if child.parent_chunk_id == result.parent_chunks[0].parent_chunk_id
    ]
    assert first_parent_children[0].text.startswith("Hi")


def test_document_start_between_and_preamble_is_preserved_in_opening_parent() -> None:
    text = (
        "# EMPLOYMENT AGREEMENT\n"
        "This Employment Agreement is made effective as of January 1, 2025.\n"
        "## BETWEEN:\n"
        "Aurora Data Systems Inc. (the \"Employer\")\n"
        "## AND:\n"
        "Daniel Reza Mohammadi (the \"Employee\")\n"
        "## 1. POSITION AND DUTIES\n"
        "The Employee will perform assigned duties.\n"
    )

    result = MarkdownParentChildChunker().chunk(_doc(text))
    opening_parent = result.parent_chunks[0]
    lowered = opening_parent.text.lower()

    assert "between" in lowered
    assert "aurora data systems inc." in lowered
    assert "daniel reza mohammadi" in lowered
    assert "the \"employer\"" in lowered
    assert "the \"employee\"" in lowered


def test_document_start_between_sentence_is_preserved_in_opening_parent() -> None:
    text = (
        "# EMPLOYMENT AGREEMENT\n"
        "This Employment Agreement is made by and between Acme Holdings LLC and Jane Smith.\n"
        "## 1. POSITION AND DUTIES\n"
        "The Employee will perform assigned duties.\n"
    )

    result = MarkdownParentChildChunker().chunk(_doc(text))
    opening_parent = result.parent_chunks[0]

    assert "made by and between Acme Holdings LLC and Jane Smith" in opening_parent.text


def test_first_numbered_section_chunking_still_produces_numbered_parent() -> None:
    text = (
        "# EMPLOYMENT AGREEMENT\n"
        "Effective Date: January 1, 2025.\n"
        "## BETWEEN:\n"
        "Acme Holdings LLC (the \"Employer\")\n"
        "## AND:\n"
        "Jane Smith (the \"Employee\")\n"
        "## 1. POSITION AND DUTIES\n"
        "The Employee will perform assigned duties.\n"
        "## 2. TERM\n"
        "The initial term is one year.\n"
    )

    result = MarkdownParentChildChunker().chunk(_doc(text))

    assert any(parent.heading_text == "1. POSITION AND DUTIES" for parent in result.parent_chunks)
    assert any(parent.heading_text == "2. TERM" for parent in result.parent_chunks)
