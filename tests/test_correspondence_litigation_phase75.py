from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, heading: str, text: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-procedural",
        text=text,
        source="test",
        source_name="litigation_file.md",
        heading_path=("Litigation", heading),
        heading_text=heading,
    )


def _procedural_context() -> list[ParentChunkResult]:
    return [
        _parent(
            "p-1",
            "Correspondence",
            "Demand letter sent on January 10, 2024 by email with subject line Final Demand and response deadline of January 20, 2024.",
        ),
        _parent(
            "p-2",
            "Pleadings",
            "Statement of Claim was filed on February 1, 2024 and served on February 3, 2024.",
        ),
        _parent(
            "p-3",
            "Defence",
            "Statement of Defence was due on February 20, 2024 and filed on February 18, 2024.",
        ),
        _parent(
            "p-4",
            "Court Filings",
            "Default notice issued on March 5, 2024 after no timely response and court filing reference entered in the docket.",
        ),
        _parent(
            "p-5",
            "Settlement",
            "Settlement discussion letter dated March 12, 2024 proposed without prejudice resolution terms.",
        ),
    ]


def test_procedural_question_recognized_as_correspondence_or_litigation_milestone_query() -> None:
    result = understand_query("What happened procedurally in the litigation?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:correspondence_litigation_milestone" in result.routing_notes


def test_sent_letter_question_uses_dated_communication_evidence_when_present() -> None:
    query = "What letters were sent and when?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _procedural_context())
    answer = generate_answer(_procedural_context(), query)

    assert result.sufficient_context is True
    assert "correspondence_dated_communication_evidence_detected" in result.evidence_notes
    assert "January 10, 2024" in answer.answer_text


def test_demanded_deadline_question_uses_deadline_language_when_present() -> None:
    query = "What deadlines were demanded?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _procedural_context())
    answer = generate_answer(_procedural_context(), query)

    assert result.sufficient_context is True
    assert "procedural_demand_deadline_evidence_detected" in result.evidence_notes
    assert "deadline" in answer.answer_text.lower() or "January 20, 2024" in answer.answer_text


def test_claim_filed_question_uses_filing_reference_when_present() -> None:
    query = "When was the claim filed?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _procedural_context())
    answer = generate_answer(_procedural_context(), query)

    assert result.sufficient_context is True
    assert "procedural_claim_filing_evidence_detected" in result.evidence_notes
    assert "February 1, 2024" in answer.answer_text


def test_defence_due_question_uses_deadline_or_procedural_evidence_when_present() -> None:
    query = "When was the defence due?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _procedural_context())
    answer = generate_answer(_procedural_context(), query)

    assert result.sufficient_context is True
    assert "procedural_defence_deadline_or_filing_evidence_detected" in result.evidence_notes
    assert "February 20, 2024" in answer.answer_text


def test_procedural_history_question_uses_responsive_milestone_evidence_when_present() -> None:
    query = "What happened procedurally in the litigation?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _procedural_context())
    answer = generate_answer(_procedural_context(), query)

    assert result.sufficient_context is True
    assert "procedural_milestone_responsive_evidence_detected" in result.evidence_notes
    assert "Statement of Claim" in answer.answer_text


def test_missing_or_ambiguous_procedural_evidence_fails_safely() -> None:
    context = [
        _parent(
            "p-x",
            "Confidentiality",
            "The parties must keep information confidential and comply with non-disclosure obligations.",
        )
    ]
    query = "When was the claim filed?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, context)
    answer = generate_answer(context, query)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"
    assert answer.sufficient_context is False


def test_non_procedural_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-c",
            "Confidentiality",
            "The receiving party shall keep Confidential Information confidential and use it only for permitted purposes unless disclosure is required by law.",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
