from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, heading: str, text: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-employment",
        text=text,
        source="test",
        source_name="employment_agreement.md",
        heading_path=("Employment Agreement", heading),
        heading_text=heading,
    )


def _lifecycle_context() -> list[ParentChunkResult]:
    return [
        _parent(
            "p-1",
            "Offer and Acceptance",
            "Offer accepted on January 3, 2023, and this acceptance created the employment relationship.",
        ),
        _parent(
            "p-2",
            "Term and Commencement",
            "This Employment Agreement is effective January 2, 2023 and employment commences on January 9, 2023.",
        ),
        _parent(
            "p-3",
            "Probation",
            "The Employee is subject to a probationary period of three (3) months ending April 9, 2023.",
        ),
        _parent(
            "p-4",
            "Compensation",
            "Base salary is $90,000 per year, payable bi-weekly, with eligibility for an annual performance bonus.",
        ),
        _parent(
            "p-5",
            "Benefits",
            "Benefits coverage begins after 30 days of employment and includes health, dental, and vision plans.",
        ),
        _parent(
            "p-6",
            "Termination",
            "Termination is effective on August 1, 2024 after delivery of written notice.",
        ),
        _parent(
            "p-7",
            "Severance",
            "Severance payable is eight (8) weeks of base salary, subject to statutory minimums.",
        ),
        _parent(
            "p-8",
            "Record of Employment",
            "The Employer will issue the Record of Employment (ROE) within five (5) days of interruption of earnings.",
        ),
        _parent(
            "p-9",
            "Confidentiality",
            (
                "The employee must keep Confidential Information confidential during and after employment, "
                "must not disclose proprietary information to third parties, and must follow all confidentiality "
                "controls listed in the agreement unless disclosure is required by law."
            ),
        ),
    ]


def test_lifecycle_question_recognized_as_employment_contract_lifecycle_query() -> None:
    result = understand_query("When did the employment relationship begin?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:employment_contract_lifecycle" in result.routing_notes


def test_start_date_question_uses_effective_or_commencement_evidence_when_present() -> None:
    query = "When did employment start?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_start_or_commencement_evidence_detected" in result.evidence_notes
    assert "January 9, 2023" in answer.answer_text


def test_compensation_question_uses_compensation_section_when_present() -> None:
    query = "What were the compensation terms?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_compensation_evidence_detected" in result.evidence_notes
    assert "$90,000" in answer.answer_text


def test_probation_question_uses_probation_language_when_present() -> None:
    query = "When did probation end?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_probation_evidence_detected" in result.evidence_notes
    assert "probationary period" in answer.answer_text.lower()


def test_benefits_question_uses_benefits_section_when_present() -> None:
    query = "What benefits applied?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_benefits_evidence_detected" in result.evidence_notes
    assert "health, dental, and vision" in answer.answer_text


def test_termination_effective_date_question_uses_termination_evidence_when_present() -> None:
    query = "When did termination take effect?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_termination_effective_evidence_detected" in result.evidence_notes
    assert "August 1, 2024" in answer.answer_text


def test_severance_question_uses_severance_language_when_present() -> None:
    query = "What severance was offered?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_severance_evidence_detected" in result.evidence_notes
    assert "eight (8) weeks" in answer.answer_text


def test_roe_question_uses_roe_reference_when_present() -> None:
    query = "When was the ROE issued?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())
    answer = generate_answer(_lifecycle_context(), query)

    assert result.sufficient_context is True
    assert "employment_lifecycle_roe_evidence_detected" in result.evidence_notes
    assert "record of employment" in answer.answer_text.lower()


def test_missing_or_ambiguous_lifecycle_evidence_fails_safely() -> None:
    context = [_parent("p-x", "Confidentiality", "Confidentiality obligations survive termination.")]
    query = "What severance was offered?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, context)
    answer = generate_answer(context, query)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"
    assert answer.sufficient_context is False


def test_non_lifecycle_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _lifecycle_context())

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
