from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, heading: str, text: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-chrono",
        text=text,
        source="test",
        source_name="employment_timeline.md",
        heading_path=("Employment Agreement", heading),
        heading_text=heading,
    )


def _chronology_context() -> list[ParentChunkResult]:
    return [
        _parent(
            "p-1",
            "Introduction",
            (
                "This Employment Agreement is made effective January 1, 2020. "
                "Employment commenced on January 15, 2020, and the introductory section confirms that this date marks the start of active duties under the agreement."
            ),
        ),
        _parent(
            "p-2",
            "Termination Notice",
            (
                "On March 15, 2023, a termination letter was sent to the employee by email, "
                "and this written communication served as formal notice under the notice clause."
            ),
        ),
        _parent(
            "p-3",
            "Termination Effective",
            "Employment terminated on April 30, 2023 after the notice period, and this date reflects the final separation event in the chronology.",
        ),
    ]


def test_chronology_question_recognized_as_legal_date_event_query() -> None:
    result = understand_query("What happened after the termination letter?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:chronology_date_event" in result.routing_notes


def test_dated_event_evidence_can_support_start_date_question() -> None:
    query = "When did employment start?"
    understanding = understand_query(query)

    result = assess_answerability(query, understanding, _chronology_context())

    assert result.sufficient_context is True
    assert "employment_start_date_supported" in result.evidence_notes


def test_dated_event_evidence_can_support_termination_notice_question() -> None:
    query = "When was the termination notice given?"
    understanding = understand_query(query)

    result = assess_answerability(query, understanding, _chronology_context())

    assert result.sufficient_context is True
    assert "termination_notice_date_supported" in result.evidence_notes


def test_first_event_question_uses_event_ordering_when_supported() -> None:
    result = generate_answer(_chronology_context(), "What happened first?")

    assert result.sufficient_context is True
    assert "January 1, 2020" in result.answer_text


def test_after_event_question_uses_chronology_responsive_evidence_when_supported() -> None:
    result = generate_answer(_chronology_context(), "What happened after the termination letter?")

    assert result.sufficient_context is True
    assert "April 30, 2023" in result.answer_text


def test_date_range_question_uses_only_events_within_range_when_supported() -> None:
    result = generate_answer(
        _chronology_context(),
        "What happened between January 1, 2023 and April 1, 2023?",
    )

    assert result.sufficient_context is True
    assert "March 15, 2023" in result.answer_text
    assert "January 1, 2020" not in result.answer_text
    assert "April 30, 2023" not in result.answer_text


def test_missing_or_ambiguous_chronology_evidence_fails_safely() -> None:
    context = [
        _parent(
            "p-1",
            "Termination",
            "Termination requires written notice but this clause includes no dated events.",
        )
    ]
    query = "What happened first?"
    understanding = understand_query(query)

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"


def test_non_chronology_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-1",
            "Confidentiality",
            (
                "The employee shall keep Confidential Information confidential and only disclose as required by law, "
                "and this clause applies during employment and after termination unless disclosure is legally required."
            ),
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
