from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.retrieval import ParentChunkResult
from agentic_rag.tools import LegalAnswerSynthesizer, generate_answer


@dataclass
class _CompressedLike:
    parent_chunk_id: str
    document_id: str
    source_name: str
    heading_text: str
    compressed_text: str


def _parent(parent_id: str, heading: str, text: str, document_id: str = "doc-1") -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id=document_id,
        text=text,
        source=f"s3://bucket/{parent_id}.md",
        source_name=f"{parent_id}.md",
        heading_path=("MSA", heading),
        heading_text=heading,
        parent_order=0,
        part_number=1,
        total_parts=1,
    )


def test_generate_answer_grounded_with_valid_citations() -> None:
    context = [
        _parent(
            "p-1",
            "Termination",
            "Either party may terminate for material breach if notice is provided and a 30-day cure period expires.",
        )
    ]

    result = generate_answer(context, "When can a party terminate for breach?")

    assert result.grounded is True
    assert result.sufficient_context is True
    assert result.citations
    assert result.citations[0].parent_chunk_id == "p-1"


def test_generate_answer_insufficient_context_safe_response() -> None:
    context = [_parent("p-1", "Definitions", "This section defines capitalized terms only.")]

    result = generate_answer(context, "What are the indemnification remedies?")

    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert "does not provide enough information" in result.answer_text.lower()


def test_generate_answer_empty_context_fallback() -> None:
    result = generate_answer([], "What is the governing law?")

    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert any("insufficient_context" in warning for warning in result.warnings)


def test_generate_answer_no_hallucination_for_missing_query_fact() -> None:
    context = [_parent("p-1", "Confidentiality", "Confidential information must be protected from disclosure.")]

    result = generate_answer(context, "What is the arbitration seat in Singapore?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert "singapore" not in result.answer_text.lower()


def test_generate_answer_citation_integrity_matches_used_context() -> None:
    context = [
        _parent("p-1", "Termination", "Termination is allowed for uncured material breach."),
        _parent("p-2", "Governing Law", "This Agreement is governed by Delaware law."),
    ]

    result = generate_answer(context, "What law governs the agreement?")

    cited_ids = {citation.parent_chunk_id for citation in result.citations}
    assert cited_ids == {"p-2"}


def test_generate_answer_preserves_qualifier_meaning() -> None:
    text = (
        "Supplier shall deliver services within 10 days. "
        "Except as required by law, Supplier may not disclose Confidential Information."
    )
    result = generate_answer([_parent("p-1", "Confidentiality", text)], "Can supplier disclose confidential information?")

    assert result.grounded is True
    assert "except as required by law" in result.answer_text.lower()


def test_generate_answer_handles_multiple_parent_chunks() -> None:
    context = [
        _parent("p-1", "Payment", "Customer must pay undisputed fees within 30 days."),
        _parent("p-2", "Late Fees", "Late fees apply only to overdue undisputed amounts."),
    ]

    result = generate_answer(context, "What are the payment timing and late fee rules?")

    assert len(result.citations) >= 2
    assert {c.parent_chunk_id for c in result.citations[:2]} == {"p-1", "p-2"}


def test_generate_answer_is_deterministic() -> None:
    context = [_parent("p-1", "Termination", "Either party may terminate for material breach.")]
    first = generate_answer(context, "When can the contract be terminated?")
    second = generate_answer(context, "When can the contract be terminated?")

    assert first == second


def test_generate_answer_failure_fallback() -> None:
    class _BrokenSynthesizer(LegalAnswerSynthesizer):
        def _rank_relevant_chunks(self, context, query):  # type: ignore[override]
            raise RuntimeError("boom")

    broken = _BrokenSynthesizer()
    result = broken.generate([_parent("p-1", "X", "Y")], "Q")

    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert any(warning.startswith("failure:") for warning in result.warnings)


def test_generate_answer_query_not_in_context_no_fabrication() -> None:
    context = [
        _CompressedLike(
            parent_chunk_id="p-c1",
            document_id="doc-7",
            source_name="doc-7.md",
            heading_text="Scope",
            compressed_text="The agreement defines service scope and acceptance criteria only.",
        )
    ]

    result = generate_answer(context, "What is the limitation of liability cap?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert "limitation of liability cap" not in result.answer_text.lower()
