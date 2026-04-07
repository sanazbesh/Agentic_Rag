from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answerability import (
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
