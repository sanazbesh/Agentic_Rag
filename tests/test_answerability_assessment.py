from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answerability import (
    PARTY_ROLE_PREVIEW_END_CHARS,
    PARTY_ROLE_PREVIEW_START_CHARS,
    CoverageEvaluation,
    EvidenceStrengthEvaluation,
    assess_answerability,
    evaluate_coverage,
)


def _parent(pid: str, text: str, heading: str = "", heading_path: tuple[str, ...] = ()) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-1",
        text=text,
        source="test",
        source_name="test-source",
        heading_path=heading_path,
        heading_text=heading,
    )


def test_definition_failure_on_title_only_match() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [_parent("p1", "Employment Agreement", heading="Employment Agreement")]

    result = assess_answerability(query, understanding, context)

    assert result.has_relevant_context is True
    assert result.sufficient_context is False
    assert result.partially_supported is False
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


def test_definition_success_with_explanatory_language() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "An employment agreement is a contract between an employer and an employee that defines terms of employment.",
            heading="Definitions",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.coverage_status == "sufficient"
    assert result.sufficient_coverage is True


def test_definition_success_with_clause_heading_and_substantive_body_without_explicit_definition_phrase() -> None:
    query = "what is Termination Without Cause?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "The Company may terminate Employee's employment at any time without Cause by providing thirty (30) days written notice and any required severance under this Agreement.",
            heading="Termination Without Cause",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.coverage_status == "sufficient"
    assert result.sufficient_coverage is True
    assert "operative_clause_language_detected" in result.supporting_signals


def test_definition_success_with_leading_clause_label_when_heading_metadata_is_coarse() -> None:
    query = "what is Termination Without Cause?"
    understanding = understand_query(
        query,
        active_documents=[{"id": "doc-1", "name": "Employment Agreement"}],
    )
    context = [
        _parent(
            "p1",
            "Termination Without Cause. The Company may terminate Employee's employment without Cause by giving thirty (30) days written notice.",
            heading="employment_agreement.pdf",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.coverage_status == "sufficient"
    assert result.sufficient_coverage is True
    assert "operative_clause_language_detected" in result.supporting_signals


def test_clause_lookup_override_succeeds_with_substantive_body_topic_match_and_coarse_heading() -> None:
    query = "what is Termination Without Cause?"
    understanding = understand_query(
        query,
        active_documents=[{"id": "doc-1", "name": "Employment Agreement"}],
    )
    context = [
        _parent(
            "p1",
            "The Company may terminate Employee's employment without Cause upon thirty (30) days written notice and payment of any accrued amounts.",
            heading="employment_agreement.pdf",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert "debug:clause_hint_match=true" in understanding.routing_notes
    assert "debug:clause_override_triggered=true" in understanding.routing_notes
    assert result.sufficient_context is True
    assert result.should_answer is True
    assert result.insufficiency_reason is None


def test_clause_lookup_override_heading_only_still_fails() -> None:
    query = "what is Termination Without Cause?"
    understanding = understand_query(
        query,
        active_documents=[{"id": "doc-1", "name": "Employment Agreement"}],
    )
    context = [_parent("p1", "employment_agreement.pdf", heading="employment_agreement.pdf")]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "only_title_or_heading_match"


def test_clause_lookup_override_unrelated_substantive_text_still_fails_without_definition_reason() -> None:
    query = "what is Termination Without Cause?"
    understanding = understand_query(
        query,
        active_documents=[{"id": "doc-1", "name": "Employment Agreement"}],
    )
    context = [
        _parent(
            "p1",
            "All notices under this Agreement must be in writing and sent by certified mail to the addresses listed above.",
            heading="employment_agreement.pdf",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is False
    assert result.insufficiency_reason == "topic_match_but_not_answer"


def test_clause_lookup_override_generic_multiword_token_overlap_is_not_sufficient() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(
        query,
        active_documents=[{"id": "doc-1", "name": "Employment Agreement"}],
    )
    context = [
        _parent(
            "p1",
            (
                "Employment begins on the Effective Date and compensation is set out in Section 4. "
                "Either party may amend this agreement in writing for bonus and benefits terms."
            ),
            heading="Compensation",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.sufficient_coverage is False
    assert result.coverage_reason == "definition_not_supported"


def test_clause_lookup_governing_law_heading_support_still_sufficient() -> None:
    query = "what is governing law?"
    understanding = understand_query(query, active_documents=[{"id": "doc-1", "name": "MSA.pdf"}])
    context = [
        _parent(
            "p1",
            (
                "This Agreement is governed by and construed under the laws of the State of New York, "
                "and each party irrevocably submits to the exclusive jurisdiction of New York courts for "
                "all disputes arising out of or related to this Agreement."
            ),
            heading="Governing Law",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_definition_multiword_clause_not_satisfied_by_broad_single_word_match() -> None:
    query = "what is Termination Without Cause?"
    understanding = understand_query(
        query,
        active_documents=[{"id": "doc-1", "name": "Employment Agreement"}],
    )
    context = [
        _parent(
            "p1",
            "Termination: Either party may terminate this Agreement with thirty (30) days written notice.",
            heading="Termination",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.sufficient_coverage is False
    assert result.coverage_reason == "no_relevant_support"


def test_definition_regression_title_and_clauses_without_definition_not_sufficient() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [
        _parent("p1", "Employment Agreement", heading="Employment Agreement"),
        _parent(
            "p2",
            "Either party may terminate this agreement with thirty (30) days written notice. This agreement is governed by New York law.",
            heading="Termination",
        ),
        _parent(
            "p3",
            "The employee must keep Confidential Information confidential during and after employment.",
            heading="Confidentiality",
        ),
    ]

    result = assess_answerability(query, understanding, context)

    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"
    assert result.has_relevant_context is True
    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.support_level in {"weak", "partial"}
    assert result.insufficiency_reason in {"definition_not_supported", "only_title_or_heading_match"}


def test_definition_required_unrelated_substantive_with_term_phrase_is_not_sufficient_coverage() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "Either party may terminate this employment agreement with thirty (30) days written notice. "
                "The Company may also place the employee on garden leave during the notice period."
            ),
            heading="Termination",
        )
    ]

    coverage = evaluate_coverage(query, understanding, context)

    assert understanding.answerability_expectation == "definition_required"
    assert coverage.sufficient_coverage is False
    assert coverage.coverage_reason == "definition_not_supported"


def test_definition_required_unrelated_substantive_with_term_phrase_is_not_sufficient_assessment() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "Either party may terminate this employment agreement with thirty (30) days written notice. "
                "The Company may also place the employee on garden leave during the notice period."
            ),
            heading="Termination",
        )
    ]

    assessment = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "definition_required"
    assert assessment.sufficient_context is False
    assert assessment.should_answer is False
    assert assessment.insufficiency_reason == "definition_not_supported"


def test_definition_required_operational_fallback_still_allows_label_anchored_clause_definition() -> None:
    query = "what is termination without cause?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "The Company may terminate Employee's employment without Cause by giving thirty (30) days written notice.",
            heading="Termination Without Cause",
        )
    ]

    coverage = evaluate_coverage(query, understanding, context)

    assert understanding.answerability_expectation == "definition_required"
    assert coverage.sufficient_coverage is True
    assert coverage.coverage_status == "sufficient"
    assert "operative_clause_language_detected" in coverage.supporting_signals


def test_definition_title_only_is_not_sufficient() -> None:
    query = "what is employment agreement?"
    understanding = understand_query(query)
    context = [_parent("p1", "Employment Agreement", heading="Employment Agreement")]

    coverage = evaluate_coverage(query, understanding, context)

    assert coverage.has_any_coverage is True
    assert coverage.sufficient_coverage is False
    assert coverage.coverage_reason == "definition_not_supported"


def test_definition_generic_substantive_text_without_clause_match_still_fails() -> None:
    query = "what is confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This agreement sets out the rights and obligations of the parties and includes provisions related to employment, compensation, and workplace conduct.",
            heading="General Terms",
        )
    ]

    coverage = evaluate_coverage(query, understanding, context)

    assert coverage.has_any_coverage is True
    assert coverage.sufficient_coverage is False
    assert coverage.coverage_reason == "definition_not_supported"


def test_definition_termination_without_cause_body_phrase_variant_still_sufficient() -> None:
    query = "what is termination without cause?"
    understanding = understand_query(query, active_documents=[{"id": "doc-1", "name": "Employment Agreement"}])
    context = [
        _parent(
            "p1",
            (
                "The Company may terminate Employee's employment without Cause by giving thirty (30) days written "
                "notice and paying accrued compensation."
            ),
            heading="employment_agreement.pdf",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.sufficient_coverage is True
    assert result.coverage_status == "sufficient"
    assert "operative_clause_language_detected" in result.supporting_signals


def test_summary_insufficiency_on_single_clause() -> None:
    query = "summarize this agreement"
    understanding = understand_query(query)
    context = [_parent("p1", "Either party may terminate this agreement with thirty (30) days written notice.", heading="Termination")]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.partially_supported is True
    assert result.insufficiency_reason in {"summary_not_supported", "partial_evidence_only"}


def test_summary_multi_section_is_sufficient() -> None:
    query = "summarize this agreement"
    understanding = understand_query(query)
    context = [
        _parent("p1", "Either party may terminate this agreement with thirty (30) days written notice.", heading="Termination"),
        _parent("p2", "This agreement is governed by the laws of Ontario.", heading="Governing Law"),
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.coverage_status == "sufficient"
    assert result.sufficient_coverage is True


def test_fact_extraction_success() -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This agreement and all disputes, claims, and controversies arising out of or related to it are governed by the laws of Ontario, Canada.",
            heading="Governing Law",
        )
    ]

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
    assert result.partially_supported is True
    assert result.insufficiency_reason == "partial_evidence_only"


def test_comparison_both_sides_evidence_is_sufficient() -> None:
    query = "compare this agreement versus ontario law"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This agreement requires arbitration for disputes, while Ontario law permits civil proceedings absent an arbitration clause.",
            heading="Dispute Resolution",
        )
    ]

    result = evaluate_coverage(query, understanding, context)

    assert result.coverage_status == "sufficient"
    assert result.coverage_reason == "comparison_supported"


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


def test_matched_headings_prefer_section_level_heading_path_over_document_title() -> None:
    query = "what does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "The receiving party shall keep Confidential Information strictly confidential.",
            heading="Master Agreement",
            heading_path=("Master Agreement", "Article II", "Confidentiality"),
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.matched_headings == ["Confidentiality"]


def test_meta_response_returns_strict_no_context_shape() -> None:
    query = "what files are loaded?"
    understanding = understand_query(query)

    result = evaluate_coverage(query, understanding, context=[])

    assert result.coverage_status == "none"
    assert result.has_any_coverage is False
    assert result.sufficient_coverage is False
    assert result.partial_coverage is False
    assert result.coverage_reason == "no_context"


def test_relevance_requires_token_match_not_substring() -> None:
    from agentic_rag.tools.answerability import AnswerabilityAssessor

    assessor = AnswerabilityAssessor()
    query_terms = {"law"}
    item = {"heading": "Drafting Notes", "text": "This section discusses a flaw in drafting quality controls."}

    assert assessor._is_relevant(item, query_terms) is False


def test_assess_answerability_delegates_to_evaluate_coverage(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="weak",
            has_any_coverage=True,
            sufficient_coverage=False,
            partial_coverage=False,
            coverage_reason="no_relevant_support",
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Governing Law"],
            supporting_signals=[],
            missing_requirements=["explicit_fact_statement_in_context"],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="strong",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=True,
            has_multiple_substantive_sections=True,
            distinct_parent_chunk_count=2,
            distinct_heading_count=2,
            approximate_text_span_count=2,
            strength_reason="broad_multi_section_support",
            supporting_signals=["substantive_body_blocks:2"],
            weakness_signals=[],
            warnings=[],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is False
    assert result.should_answer is False


def test_assess_answerability_no_silent_upgrade_for_partial_coverage(monkeypatch) -> None:
    query = "summarize this agreement"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="partial",
            has_any_coverage=True,
            sufficient_coverage=False,
            partial_coverage=True,
            coverage_reason="summary_partially_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Termination"],
            supporting_signals=["single_section_context_only"],
            missing_requirements=["multi_section_or_multi_parent_coverage"],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="strong",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=True,
            has_multiple_substantive_sections=True,
            distinct_parent_chunk_count=2,
            distinct_heading_count=2,
            approximate_text_span_count=2,
            strength_reason="multiple_substantive_sections",
            supporting_signals=["substantive_body_blocks:2"],
            weakness_signals=[],
            warnings=[],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is False
    assert result.partially_supported is True
    assert result.should_answer is False


def test_assess_answerability_coverage_sufficient_but_heading_only_weak_strength_is_not_sufficient(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="sufficient",
            has_any_coverage=True,
            sufficient_coverage=True,
            partial_coverage=False,
            coverage_reason="fact_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Governing Law"],
            supporting_signals=["explicit_fact_statement_detected"],
            missing_requirements=[],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="weak",
            has_title_only_match=False,
            has_heading_only_match=True,
            has_substantive_clause_text=False,
            has_multiple_substantive_sections=False,
            distinct_parent_chunk_count=1,
            distinct_heading_count=1,
            approximate_text_span_count=0,
            strength_reason="heading_only_match",
            supporting_signals=[],
            weakness_signals=["heading_only_signal_without_body"],
            warnings=["no_substantive_clause_text_detected"],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.support_level == "weak"


def test_assess_answerability_coverage_sufficient_plus_thin_weak_strength_stays_insufficient(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="sufficient",
            has_any_coverage=True,
            sufficient_coverage=True,
            partial_coverage=False,
            coverage_reason="fact_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Governing Law"],
            supporting_signals=["explicit_fact_statement_detected"],
            missing_requirements=[],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="weak",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=False,
            has_multiple_substantive_sections=False,
            distinct_parent_chunk_count=1,
            distinct_heading_count=1,
            approximate_text_span_count=0,
            strength_reason="thin_single_clause",
            supporting_signals=[],
            weakness_signals=["only_thin_body_fragments_detected"],
            warnings=["no_substantive_clause_text_detected"],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.support_level == "weak"


def test_assess_answerability_coverage_sufficient_plus_moderate_strength_is_sufficient(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="sufficient",
            has_any_coverage=True,
            sufficient_coverage=True,
            partial_coverage=False,
            coverage_reason="fact_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Governing Law"],
            supporting_signals=["explicit_fact_statement_detected"],
            missing_requirements=[],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="moderate",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=True,
            has_multiple_substantive_sections=False,
            distinct_parent_chunk_count=1,
            distinct_heading_count=1,
            approximate_text_span_count=1,
            strength_reason="single_substantive_clause",
            supporting_signals=["substantive_body_blocks:1"],
            weakness_signals=[],
            warnings=[],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert result.support_level == "sufficient"


def test_assess_answerability_moderate_policy_is_deterministic(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="sufficient",
            has_any_coverage=True,
            sufficient_coverage=True,
            partial_coverage=False,
            coverage_reason="fact_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Governing Law"],
            supporting_signals=["explicit_fact_statement_detected"],
            missing_requirements=[],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="moderate",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=True,
            has_multiple_substantive_sections=False,
            distinct_parent_chunk_count=1,
            distinct_heading_count=1,
            approximate_text_span_count=1,
            strength_reason="single_substantive_clause",
            supporting_signals=["substantive_body_blocks:1"],
            weakness_signals=[],
            warnings=[],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)

    a = assess_answerability(query, understanding, [])
    b = assess_answerability(query, understanding, [])

    assert a == b
    assert a.sufficient_context is True
    assert a.should_answer is True


def test_assess_answerability_coverage_sufficient_plus_strong_strength_is_sufficient(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _stub_coverage(*args, **kwargs) -> CoverageEvaluation:
        return CoverageEvaluation(
            original_query=query,
            answerability_expectation=understanding.answerability_expectation,
            coverage_status="sufficient",
            has_any_coverage=True,
            sufficient_coverage=True,
            partial_coverage=False,
            coverage_reason="fact_supported",
            matched_parent_chunk_ids=["p1", "p2"],
            matched_headings=["Governing Law", "Definitions"],
            supporting_signals=["explicit_fact_statement_detected"],
            missing_requirements=[],
            warnings=[],
        )

    def _stub_strength(*args, **kwargs) -> EvidenceStrengthEvaluation:
        return EvidenceStrengthEvaluation(
            original_query=query,
            evidence_strength="strong",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=True,
            has_multiple_substantive_sections=True,
            distinct_parent_chunk_count=2,
            distinct_heading_count=2,
            approximate_text_span_count=2,
            strength_reason="broad_multi_section_support",
            supporting_signals=["substantive_body_blocks:2"],
            weakness_signals=[],
            warnings=[],
        )

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _stub_coverage)
    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _stub_strength)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert result.support_level == "sufficient"


def test_assess_answerability_coverage_failure_safe_fallback(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _raise(*args, **kwargs):
        raise RuntimeError("coverage engine unavailable")

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_coverage", _raise)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert any(w.startswith("coverage_evaluation_failed:") for w in result.warnings)


def test_assess_answerability_strength_failure_safe_fallback(monkeypatch) -> None:
    query = "which law governs the agreement?"
    understanding = understand_query(query)

    def _raise(*args, **kwargs):
        raise RuntimeError("strength engine unavailable")

    monkeypatch.setattr("agentic_rag.tools.answerability.evaluate_evidence_strength", _raise)
    result = assess_answerability(query, understanding, [])

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert any(w.startswith("strength_evaluation_failed:") for w in result.warnings)


def test_agreement_intro_line_can_support_party_question() -> None:
    query = "who are the parties to this agreement?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "This Employment Agreement is made effective as of January 1, 2025, by and between Acme Corp and Jane Smith. "
                "The opening recital identifies the contracting parties for all obligations in this Agreement."
            ),
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert "party_role_responsive_evidence_detected" in result.evidence_notes


def test_who_is_the_employer_uses_party_role_evidence_not_unrelated_clause_text() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Agreement is governed by New York law and includes confidentiality obligations for both parties.",
            heading="Confidentiality",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"


def test_who_are_the_parties_returns_party_evidence_when_present() -> None:
    query = "who are the parties?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "The parties to this Agreement are Acme Corp (the Employer) and Jane Smith (the Employee). "
                "These identified parties are the signatories and are bound by the operative provisions below."
            ),
            heading="Parties",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_which_company_is_this_agreement_for_uses_entity_evidence_when_present() -> None:
    query = "which company is this agreement for?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "This Agreement is made by and between Acme Holdings, Inc. and John Roe. "
                "The introductory statement names the company entity associated with this agreement."
            ),
            heading="Introductory Statement",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_missing_party_role_evidence_fails_safely() -> None:
    query = "who is the employee?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "Either party may terminate this Agreement with thirty (30) days written notice.",
            heading="Termination",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "fact_not_found"


def test_successful_employer_role_resolution_counts_as_fact_support() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "This Employment Agreement is made by and between Acme Corp and Jane Smith. "
                "The introductory recital identifies the two contracting parties for all obligations in this agreement."
            ),
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert "employer_role_assignment_resolved" in result.evidence_notes


def test_successful_employee_role_resolution_counts_as_fact_support() -> None:
    query = "who is the employee?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "This Employment Agreement is made by and between Acme Corp and Jane Smith. "
                "The introductory recital identifies the two contracting parties for all obligations in this agreement."
            ),
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert "employee_role_assignment_resolved" in result.evidence_notes


def test_between_and_multiline_intro_with_explicit_role_labels_is_sufficient_for_party_role() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "BETWEEN:\nAcme Holdings LLC (the “Employer”)\nAND:\nJane Smith (the “Employee”)",
            heading="Parties",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert "employer_role_assignment_resolved" in result.evidence_notes


def test_as_role_intro_assignment_is_sufficient_for_parties_question() -> None:
    query = "who are the parties?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "Acme Holdings LLC as Employer and Jane Smith as Employee enter this Employment Agreement.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert "party_set_resolved" in result.evidence_notes


def test_company_side_and_individual_side_queries_use_explicit_intro_role_evidence() -> None:
    company_query = "which party is the company side?"
    company_understanding = understand_query(company_query)
    individual_query = "which party is the individual side?"
    individual_understanding = understand_query(individual_query)
    context = [
        _parent(
            "p1",
            'This Employment Agreement is between Acme Holdings LLC ("Employer") and Jane Smith ("Employee").',
            heading="Parties",
        )
    ]

    company_result = assess_answerability(company_query, company_understanding, context)
    individual_result = assess_answerability(individual_query, individual_understanding, context)

    assert company_result.sufficient_context is True
    assert individual_result.sufficient_context is True
    assert "company_side_party_identified" in company_result.evidence_notes
    assert "individual_side_party_identified" in individual_result.evidence_notes


def test_successful_parties_resolution_counts_as_fact_support() -> None:
    query = "who are the parties?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert "party_set_resolved" in result.evidence_notes


def test_agreement_between_x_and_y_uses_extracted_party_set_when_supported_in_answerability() -> None:
    query = "is this agreement between acme corp and jane smith?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert "agreement_between_pair_confirmed_from_extracted_parties" in result.evidence_notes


def test_agreement_between_query_uses_extracted_party_set_as_fact_support() -> None:
    query = "is this agreement with acme corp?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True
    assert "agreement_between_pair_confirmed_from_extracted_parties" in result.evidence_notes


def test_agreement_between_query_fails_safely_when_party_set_is_mismatched() -> None:
    query = "is this agreement between acme corp and john roe?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "fact_not_found"


def test_agreement_between_query_fails_safely_when_party_set_is_ambiguous_or_incomplete() -> None:
    query = "is this agreement between acme corp and jane smith?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "This Employment Agreement is made between the Company and the Employee. "
                "The introductory paragraph uses role labels but does not resolve named parties."
            ),
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "fact_not_found"


def test_ambiguous_or_missing_role_resolution_still_fails_safely_in_answerability() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made between the Company and the Employee.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "fact_not_found"





def test_party_role_resolution_debug_payload_includes_per_parent_previews_for_runtime_checked_parents() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "INTRODUCTION\nThis is a short heading-only section without party intro details.",
            heading="Introduction",
        ),
        _parent(
            "p2",
            (
                "This Employment Agreement is entered into by and between Acme Corp (\"Employer\") "
                "and Jane Smith (\"Employee\").\nAdditional terms follow in later sections."
            ),
            heading="Agreement Overview",
        ),
    ]

    result = assess_answerability(query, understanding, context)

    assert result.party_role_resolution_debug is not None
    debug = result.party_role_resolution_debug
    assert debug.party_role_resolution_checked_parent_count == 2
    assert debug.party_role_resolution_checked_parent_ids == ["p1", "p2"]
    assert len(debug.checked_parent_previews) == 2
    assert debug.checked_parent_previews[0].parent_chunk_id == "p1"
    assert debug.checked_parent_previews[0].heading == "Introduction"
    assert debug.checked_parent_previews[0].text_length_chars > 0
    assert debug.checked_parent_previews[0].preview_start
    assert debug.checked_parent_previews[0].preview_end
    assert debug.checked_parent_previews[1].intro_pattern_detected is True
    assert debug.checked_parent_previews[1].resolver_considered_usable_intro_text is True


def test_party_role_resolution_debug_flags_missing_intro_pattern_when_runtime_context_lacks_intro_text() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "EMPLOYMENT AGREEMENT\nSection 1 - Compensation\nSection 2 - Benefits",
            heading="Cover Page",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.party_role_resolution_debug is not None
    debug = result.party_role_resolution_debug
    assert debug.party_role_resolution_debug_outcome == "not_found"
    assert debug.party_role_resolution_intro_pattern_parent_ids == []
    assert debug.checked_parent_previews[0].intro_pattern_detected is False
    assert debug.checked_parent_previews[0].resolver_considered_usable_intro_text is False


def test_party_role_resolution_debug_preview_text_is_bounded_and_not_full_dump() -> None:
    query = "who is the employer?"
    understanding = understand_query(query)
    long_text = (
        "This Employment Agreement is made by and between Acme Corp (Employer) and Jane Smith (Employee).\n"
        + ("Body paragraph with many details.\n" * 80)
    )
    context = [_parent("p1", long_text, heading="Introduction")]

    result = assess_answerability(query, understanding, context)

    assert result.party_role_resolution_debug is not None
    preview = result.party_role_resolution_debug.checked_parent_previews[0]
    assert len(preview.preview_start) <= PARTY_ROLE_PREVIEW_START_CHARS
    assert len(preview.preview_end) <= PARTY_ROLE_PREVIEW_END_CHARS
    assert preview.text_length_chars > (PARTY_ROLE_PREVIEW_START_CHARS + PARTY_ROLE_PREVIEW_END_CHARS)
    assert preview.preview_start != long_text
    assert preview.preview_end != long_text

def test_agreement_between_query_extracts_query_side_entity_set_for_comparison() -> None:
    query = "is this agreement between Aurora Data Systems Inc. and Daniel Reza Mohammadi?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made by and between Aurora Data Systems Inc and Daniel Reza Mohammadi.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert "agreement_between_query_entity_set_matched_extracted_party_set" in result.evidence_notes


def test_agreement_between_query_fails_safely_when_query_entity_set_is_incomplete_or_ambiguous() -> None:
    query = "is this agreement between Acme Corp and the employee?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
            heading="Introduction",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "fact_not_found"
def test_non_party_clause_lookup_behavior_remains_unchanged_in_answerability() -> None:
    query = "what does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p1",
            (
                "The receiving party shall keep confidential information confidential and may disclose it only when required "
                "by law, with prior written notice where legally permitted."
            ),
            heading="Confidentiality",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
