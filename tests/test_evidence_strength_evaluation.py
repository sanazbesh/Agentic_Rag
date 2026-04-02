from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answerability import evaluate_evidence_strength


def _parent(pid: str, text: str, heading: str = "", source_name: str = "test-source") -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-1",
        text=text,
        source="test",
        source_name=source_name,
        heading_text=heading,
    )


def test_evidence_strength_empty_context() -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    result = evaluate_evidence_strength(query, understanding, [])

    assert result.evidence_strength == "none"
    assert result.strength_reason == "no_context"


def test_evidence_strength_title_only() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [_parent("p1", "Employment Agreement", heading="Employment Agreement", source_name="Employment Agreement")]

    result = evaluate_evidence_strength(query, understanding, context)

    assert result.evidence_strength == "weak"
    assert result.has_title_only_match is True
    assert result.strength_reason == "title_only_match"


def test_evidence_strength_heading_only() -> None:
    query = "what does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [_parent("p1", "Confidentiality", heading="Confidentiality")]

    result = evaluate_evidence_strength(query, understanding, context)

    assert result.evidence_strength == "weak"
    assert result.has_heading_only_match is True
    assert result.strength_reason == "heading_only_match"


def test_evidence_strength_thin_single_clause() -> None:
    query = "what does the document say about notice?"
    understanding = understand_query(query)
    context = [_parent("p1", "Party may terminate with 30 days written notice.", heading="Termination")]

    result = evaluate_evidence_strength(query, understanding, context)

    assert result.evidence_strength != "strong"
    assert result.strength_reason == "thin_single_clause"


def test_evidence_strength_single_substantive_clause() -> None:
    query = "what does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "The receiving party shall keep all Confidential Information strictly confidential, use it only for the stated business purpose, and disclose it solely when required by law after prompt written notice.",
            heading="Confidentiality",
        )
    ]

    result = evaluate_evidence_strength(query, understanding, context)

    assert result.evidence_strength == "moderate"
    assert result.has_substantive_clause_text is True


def test_evidence_strength_multiple_substantive_sections() -> None:
    query = "summarize this agreement"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "Either party may terminate this agreement for convenience by giving at least thirty days prior written notice, subject to payment of all accrued fees and return of materials.",
            heading="Termination",
        ),
        _parent(
            "p1",
            "This agreement and any dispute arising out of it shall be governed by the laws of Ontario, excluding conflict of law rules, and the courts of Toronto have exclusive jurisdiction.",
            heading="Governing Law",
        ),
    ]

    result = evaluate_evidence_strength(query, understanding, context)

    assert result.evidence_strength == "strong"
    assert result.has_multiple_substantive_sections is True


def test_evidence_strength_broad_support_multi_parent() -> None:
    query = "summarize this agreement"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "Either party may terminate this agreement for convenience by giving at least thirty days prior written notice, subject to payment of all accrued fees and return of materials.",
            heading="Termination",
        ),
        _parent(
            "p2",
            "This agreement and any dispute arising out of it shall be governed by the laws of Ontario, excluding conflict of law rules, and the courts of Toronto have exclusive jurisdiction.",
            heading="Governing Law",
        ),
    ]

    result = evaluate_evidence_strength(query, understanding, context)

    assert result.evidence_strength == "strong"
    assert result.distinct_parent_chunk_count > 1


def test_evidence_strength_determinism() -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This agreement and any dispute arising out of it shall be governed by the laws of Ontario and interpreted in accordance with those laws without regard to conflict principles.",
            heading="Governing Law",
        )
    ]

    a = evaluate_evidence_strength(query, understanding, context)
    b = evaluate_evidence_strength(query, understanding, context)

    assert a == b
