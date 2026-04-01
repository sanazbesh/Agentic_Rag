from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, text: str, heading: str = "") -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-1",
        text=text,
        source="test",
        source_name="test-source",
        heading_text=heading,
    )


def test_definition_failure_on_title_only_match() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [_parent("p1", "Employment Agreement", heading="Employment Agreement")]

    result = assess_answerability(query, understanding, context)

    assert result.has_relevant_context is True
    assert result.sufficient_context is False
    assert result.partially_supported is True
    assert result.insufficiency_reason in {"definition_not_supported", "only_title_or_heading_match"}


def test_clause_lookup_success() -> None:
    query = "what does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "The receiving party shall keep all Confidential Information strictly confidential and may disclose it only as required by law.",
            heading="Confidentiality",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_summary_insufficiency_on_single_clause() -> None:
    query = "summarize this agreement"
    understanding = understand_query(query)
    context = [_parent("p1", "Either party may terminate this agreement with thirty (30) days written notice.", heading="Termination")]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.partially_supported is True
    assert result.insufficiency_reason in {"summary_not_supported", "partial_evidence_only"}


def test_fact_extraction_success() -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)
    context = [_parent("p1", "This agreement is governed by the laws of Ontario.", heading="Governing Law")]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_fact_extraction_failure_when_fact_missing() -> None:
    query = "who are the parties?"
    understanding = understand_query(query)
    context = [_parent("p1", "The agreement contains a confidentiality clause and an indemnity clause.", heading="Confidentiality")]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"


def test_comparison_insufficiency_with_one_sided_evidence() -> None:
    query = "compare this agreement with Ontario law"
    understanding = understand_query(query)
    context = [_parent("p1", "This agreement allows termination for convenience on thirty days notice.", heading="Termination")]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "comparison_not_supported"


def test_empty_context() -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    result = assess_answerability(query, understanding, [])

    assert result.has_relevant_context is False
    assert result.sufficient_context is False
    assert result.should_answer is False


def test_determinism() -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)
    context = [_parent("p1", "This agreement is governed by the laws of Ontario.", heading="Governing Law")]

    a = assess_answerability(query, understanding, context)
    b = assess_answerability(query, understanding, context)

    assert a == b


def test_no_heading_only_false_positive() -> None:
    query = "what does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [_parent("p1", "Confidentiality", heading="Confidentiality")]

    result = assess_answerability(query, understanding, context)

    assert result.has_relevant_context is True
    assert result.sufficient_context is False
    assert result.insufficiency_reason == "only_title_or_heading_match"
