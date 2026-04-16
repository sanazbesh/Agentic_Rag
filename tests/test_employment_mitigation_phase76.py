from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, heading: str, text: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-mitigation",
        text=text,
        source="test",
        source_name="employment_mitigation_file.md",
        heading_path=("Employment File", heading),
        heading_text=heading,
    )


def _mitigation_context() -> list[ParentChunkResult]:
    return [
        _parent(
            "m-1",
            "Mitigation Journal",
            (
                "Mitigation efforts included a weekly job search log with outreach, resume updates, and networking contacts, "
                "and the journal records each application and follow-up communication made by the employee after termination."
            ),
        ),
        _parent(
            "m-2",
            "Application Records",
            (
                "12 job applications were submitted between January and March 2024, including roles in operations and payroll, "
                "with application confirmations and resume submissions tracked in the mitigation spreadsheet."
            ),
        ),
        _parent(
            "m-3",
            "Interview Invitations",
            (
                "Interview conducted on February 14, 2024 and a second interview scheduled for March 3, 2024, "
                "as confirmed by invitation emails and calendar records attached to counsel correspondence."
            ),
        ),
        _parent(
            "m-4",
            "Offer Update",
            (
                "Offer letter received on March 20, 2024 and declined due to compensation mismatch, "
                "with an email chain documenting the receipt and rejection decision."
            ),
        ),
        _parent(
            "m-5",
            "Employment Update",
            (
                "Alternative employment secured with Northline Co.; new employment started on April 15, 2024, "
                "as shown in onboarding correspondence and a signed start-date confirmation."
            ),
        ),
        _parent(
            "m-6",
            "Correspondence",
            (
                "Email attaches mitigation journal, application record spreadsheet, and interview invitation screenshots, "
                "and references additional mitigation evidence including offer correspondence and employment updates."
            ),
        ),
        _parent(
            "m-7",
            "Confidentiality",
            (
                "The employee must keep Confidential Information confidential during and after employment, "
                "must not disclose proprietary information to third parties, and must follow all confidentiality controls "
                "listed in the agreement unless disclosure is required by law."
            ),
        ),
    ]


def test_mitigation_question_recognized_as_employment_mitigation_query() -> None:
    result = understand_query("What mitigation efforts were made?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:employment_mitigation" in result.routing_notes


def test_mitigation_efforts_question_uses_mitigation_responsive_evidence_when_present() -> None:
    query = "What mitigation efforts were made?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _mitigation_context())
    answer = generate_answer(_mitigation_context(), query)

    assert result.sufficient_context is True
    assert "employment_mitigation_efforts_evidence_detected" in result.evidence_notes
    assert "mitigation" in answer.answer_text.lower()


def test_application_count_question_uses_application_records_when_present() -> None:
    query = "How many job applications were submitted?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _mitigation_context())
    answer = generate_answer(_mitigation_context(), query)

    assert result.sufficient_context is True
    assert "employment_mitigation_application_record_evidence_detected" in result.evidence_notes
    assert "12 job applications" in answer.answer_text


def test_interview_date_question_uses_interview_evidence_when_present() -> None:
    query = "When were interviews conducted?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _mitigation_context())
    answer = generate_answer(_mitigation_context(), query)

    assert result.sufficient_context is True
    assert "employment_mitigation_interview_evidence_detected" in result.evidence_notes
    assert "February 14, 2024" in answer.answer_text


def test_alternative_employment_question_uses_new_employment_or_offer_evidence_when_present() -> None:
    query = "Was alternative employment found?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _mitigation_context())
    answer = generate_answer(_mitigation_context(), query)

    assert result.sufficient_context is True
    assert "employment_mitigation_alternative_employment_evidence_detected" in result.evidence_notes
    assert "Alternative employment secured" in answer.answer_text


def test_mitigation_evidence_question_uses_responsive_support_when_present() -> None:
    query = "What mitigation evidence exists?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _mitigation_context())
    answer = generate_answer(_mitigation_context(), query)

    assert result.sufficient_context is True
    assert "employment_mitigation_evidence_source_detected" in result.evidence_notes
    assert "mitigation evidence" in answer.answer_text.lower()


def test_missing_or_ambiguous_mitigation_evidence_fails_safely() -> None:
    query = "What mitigation efforts were made?"
    understanding = understand_query(query)
    context = [
        _parent(
            "m-x",
            "Confidentiality",
            "The employee must keep confidential information confidential during and after employment.",
        )
    ]

    result = assess_answerability(query, understanding, context)
    answer = generate_answer(context, query)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"
    assert answer.sufficient_context is False


def test_non_mitigation_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)

    result = assess_answerability(query, understanding, _mitigation_context())

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
