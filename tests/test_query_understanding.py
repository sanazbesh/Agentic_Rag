from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query


def test_meta_query() -> None:
    result = understand_query("how many documents do you have?")
    assert result.question_type == "meta_query"
    assert result.should_retrieve is False
    assert result.answerability_expectation == "meta_response"


def test_definition_query() -> None:
    result = understand_query("what is employment agreement?")
    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"


def test_what_is_clause_lookup_only_when_document_grounded() -> None:
    result = understand_query(
        "what is confidentiality?",
        active_documents=[{"id": "doc-nda", "name": "Mutual NDA"}],
    )
    assert result.question_type == "document_content_query"
    assert result.answerability_expectation == "clause_lookup"
    assert result.is_document_scoped is True


def test_what_is_title_only_match_does_not_trigger_clause_lookup_override() -> None:
    result = understand_query(
        "what is employment agreement?",
        active_documents=[{"id": "doc-employment", "title": "Employment Agreement"}],
    )
    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"
    assert "what_is_override_blocked_by_non_independent_hint_evidence" in result.routing_notes


def test_what_is_filename_like_label_does_not_trigger_clause_lookup_override() -> None:
    result = understand_query(
        "what is employment agreement?",
        active_documents=[{"id": "doc-employment", "name": "employment_agreement_019.md"}],
    )
    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"
    assert "what_is_override_blocked_by_non_independent_hint_evidence" in result.routing_notes


def test_what_is_generic_agreement_label_does_not_trigger_clause_lookup_override() -> None:
    result = understand_query(
        "what is agreement?",
        active_documents=[{"id": "doc-employment", "name": "Employment Agreement"}],
    )
    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"


def test_what_is_remains_definition_without_active_document_context() -> None:
    result = understand_query("what is confidentiality?")
    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"


def test_what_is_weak_hint_match_stays_definition_and_logs_ambiguity() -> None:
    result = understand_query(
        "what is law?",
        active_documents=[{"id": "doc-nda", "name": "Mutual NDA"}],
    )
    assert result.question_type == "definition_query"
    assert result.answerability_expectation == "definition_required"
    assert "ambiguous_definition_vs_clause" in result.ambiguity_notes


def test_what_is_multiword_clause_subject_preserved_for_hint_matching() -> None:
    result = understand_query(
        "what is Termination Without Cause?",
        active_documents=[{"id": "doc-employment", "name": "Employment Agreement"}],
    )
    assert "termination without cause" in result.resolved_clause_hints
    assert result.question_type == "document_content_query"
    assert "debug:clause_hint_match=true" in result.routing_notes


def test_what_is_governing_law_clause_lookup_still_triggers_with_independent_topic_evidence() -> None:
    result = understand_query(
        "what is governing law?",
        active_documents=[{"id": "doc-nda", "name": "Mutual NDA"}],
    )
    assert result.question_type == "document_content_query"
    assert result.answerability_expectation == "clause_lookup"
    assert "debug:clause_hint_match=true" in result.routing_notes


def test_document_content_query() -> None:
    result = understand_query("what does the document say about confidentiality?")
    assert result.question_type == "document_content_query"
    assert result.is_document_scoped is True
    assert result.should_retrieve is True


def test_followup_query_context_overlay() -> None:
    result = understand_query(
        "what about governing law?",
        conversation_summary="Prior turn discussed NDA confidentiality clause.",
        recent_messages=[{"role": "assistant", "content": "We reviewed NDA confidentiality."}],
    )
    assert result.is_followup is True
    assert result.is_context_dependent is True
    assert result.use_conversation_context is True


def test_pronoun_resolution_hint_single_document() -> None:
    result = understand_query(
        "what does it say about confidentiality?",
        selected_documents=[{"id": "doc-nda", "name": "Mutual NDA"}],
    )
    assert result.use_conversation_context is True
    assert result.refers_to_prior_document_scope is True or result.is_document_scoped is True
    assert result.resolved_document_hints


def test_ambiguous_pronoun_with_multiple_documents() -> None:
    result = understand_query(
        "what does it say about governing law?",
        active_documents=[{"id": "a", "name": "NDA"}, {"id": "b", "name": "Lease"}],
    )
    assert result.question_type == "ambiguous_query" or result.ambiguity_notes
    assert result.resolved_document_hints == []


def test_comparison_query() -> None:
    result = understand_query("compare that with Ontario law")
    assert result.question_type == "comparison_query"
    assert result.may_need_decomposition is True


def test_non_canonical_query_understanding_behavior_remains_unchanged() -> None:
    result = understand_query("compare that with Ontario law")
    assert result.question_type == "comparison_query"
    assert result.answerability_expectation == "comparison"
    assert result.may_need_decomposition is True


def test_self_contained_override_explicit_document() -> None:
    result = understand_query(
        "What does the NDA say about confidentiality?",
        conversation_summary="Earlier we discussed a lease.",
        recent_messages=[{"role": "assistant", "content": "lease analysis"}],
        active_documents=[{"name": "NDA"}, {"name": "Lease"}],
    )
    assert result.question_type == "document_content_query"
    assert result.use_conversation_context is False
    assert "NDA" in result.resolved_document_hints


def test_determinism() -> None:
    kwargs = dict(
        query="what about governing law?",
        conversation_summary="We discussed an NDA.",
        recent_messages=[{"role": "assistant", "content": "NDA scope", "metadata": {"resolved_document_ids": ["doc-nda"]}}],
        selected_documents=[{"id": "doc-nda", "name": "NDA"}],
    )
    result_a = understand_query(**kwargs)
    result_b = understand_query(**kwargs)
    assert result_a == result_b


def test_canonical_tricky_followup_example() -> None:
    result = understand_query(
        "what about governing law?",
        conversation_summary="Prior turn analyzed the NDA confidentiality clause.",
        recent_messages=[{"role": "assistant", "content": "NDA summary"}],
        selected_documents=[{"id": "doc-nda", "name": "Mutual NDA"}],
    )
    assert result.question_type in {"other_query", "document_content_query", "extractive_fact_query"}
    assert result.is_followup is True
    assert result.is_context_dependent is True
    assert result.use_conversation_context is True
    assert result.resolved_document_hints == ["Mutual NDA"]


def test_party_question_recognized_as_legal_party_role_query() -> None:
    result = understand_query("who is the employer?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:party_role_entity" in result.routing_notes
    assert result.should_rewrite is True


def test_agreement_with_verification_is_recognized_as_party_role_entity_query() -> None:
    result = understand_query("is this agreement with Acme Corp?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:party_role_entity" in result.routing_notes


def test_non_party_clause_lookup_behavior_remains_unchanged() -> None:
    result = understand_query("what does the document say about confidentiality?")

    assert result.question_type == "document_content_query"
    assert result.answerability_expectation == "clause_lookup"
    assert "legal_question_family:party_role_entity" not in result.routing_notes


def test_matter_metadata_question_recognized_as_distinct_legal_question_family() -> None:
    result = understand_query("What court is involved?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:matter_document_metadata" in result.routing_notes
