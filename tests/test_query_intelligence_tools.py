from __future__ import annotations

from agentic_rag.tools import decompose_query, extract_legal_entities, rewrite_query
from agentic_rag.tools.query_intelligence import QueryTransformationService


class _FakeLLM:
    def __init__(self, response: str, raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm failure")
        return self.response


def test_rewrite_query_returns_structured_result() -> None:
    result = rewrite_query("What is the negligence standard in California?")

    assert result.original_query == "What is the negligence standard in California?"
    assert isinstance(result.rewritten_query, str)
    assert isinstance(result.used_conversation_context, bool)
    assert result.rewrite_notes


def test_rewrite_query_preserves_legal_meaning() -> None:
    query = "Under New York law, what are the elements of breach of contract in 2024?"

    result = rewrite_query(query)

    assert result.rewritten_query == query
    assert "New York" in result.rewritten_query
    assert "2024" in result.rewritten_query


def test_rewrite_query_handles_empty_input_safely() -> None:
    assert rewrite_query("").rewritten_query == ""
    assert rewrite_query("   ").rewritten_query == ""


def test_rewrite_query_resolves_references_when_context_is_provided() -> None:
    summary = "We discussed Section 2.1 of the MSA termination rights."

    result = rewrite_query("How is that clause enforced?", conversation_summary=summary)

    assert result.used_conversation_context is True
    assert "Section 2.1" in result.rewritten_query


def test_rewrite_query_does_not_inject_context_when_unnecessary() -> None:
    query = "Interpret Section 10 indemnity language under Delaware law."

    result = rewrite_query(
        query,
        conversation_summary="Earlier we reviewed Section 2.1 in a different contract.",
        recent_messages=["Please keep discussing Section 2.1."],
    )

    assert result.rewritten_query == query
    assert result.used_conversation_context is False


def test_decompose_query_returns_structured_result() -> None:
    result = decompose_query("What is consideration in contract law?")

    assert result.original_query == "What is consideration in contract law?"
    assert isinstance(result.sub_queries, tuple)
    assert isinstance(result.used_conversation_context, bool)
    assert result.decomposition_notes


def test_decompose_query_returns_single_subquery_for_simple_query() -> None:
    result = decompose_query("What is promissory estoppel?")

    assert result.sub_queries == ("What is promissory estoppel?",)


def test_decompose_query_returns_multiple_for_complex_queries() -> None:
    query = "What is the rule and exception for hearsay in New York; definition and remedy for violation"

    result = decompose_query(query)

    assert len(result.sub_queries) > 1
    assert any("rule" in part.lower() for part in result.sub_queries)
    assert any("exception" in part.lower() for part in result.sub_queries)


def test_decompose_query_preserves_deterministic_ordering() -> None:
    query = "definition and enforcement and remedy of liquidated damages"

    first = decompose_query(query)
    second = decompose_query(query)

    assert first.sub_queries == second.sub_queries


def test_both_tools_correctly_use_or_ignore_context() -> None:
    ambiguous = rewrite_query(
        "Does this case discuss duty?",
        conversation_summary="We are analyzing Smith v. Jones from the California Court of Appeal.",
    )
    self_contained = decompose_query(
        "Compare procedural and substantive unconscionability in California.",
        conversation_summary="Older topic about hearsay exceptions.",
        recent_messages=["Prior case was unrelated."],
    )

    assert ambiguous.used_conversation_context is True
    assert "Smith v. Jones" in ambiguous.rewritten_query
    assert self_contained.used_conversation_context is False


def test_llm_is_used_only_when_needed() -> None:
    llm = _FakeLLM('{\"rewritten_query\": \"How is Section 3 enforced?\"}')
    service = QueryTransformationService(llm_client=llm)

    clear = service.rewrite_query("Interpret Section 10 under Delaware law.")
    ambiguous = service.rewrite_query(
        "How is that clause enforced?",
        conversation_summary="We discussed Section 3 in the MSA.",
    )

    assert clear.rewritten_query == "Interpret Section 10 under Delaware law."
    assert ambiguous.rewritten_query == "How is Section 3 enforced?"
    assert llm.calls == 1


def test_safe_fallbacks_on_llm_failure() -> None:
    service = QueryTransformationService(llm_client=_FakeLLM("{}", raises=True))

    rewrite = service.rewrite_query(
        "How is that clause enforced?",
        conversation_summary="We discussed Section 3 in the MSA.",
    )
    decompose = service.decompose_query("definition and remedy for unconscionability")

    assert rewrite.rewritten_query == "How is that clause enforced?"
    assert rewrite.rewrite_notes.startswith("llm_failure_fallback_original_query")
    assert decompose.sub_queries == ("definition and remedy for unconscionability",)
    assert decompose.decomposition_notes.startswith("llm_failure_fallback_original_query")


def test_extract_legal_entities_contract_clause_query() -> None:
    result = extract_legal_entities("Review NDA confidentiality clause obligations within 30 days.")

    assert result.normalized_query == "Review non-disclosure agreement confidentiality clause obligations within 30 days."
    assert "non-disclosure agreement" in result.document_types
    assert "confidentiality clause" in result.clause_types
    assert "within 30 days" in result.time_constraints


def test_extract_legal_entities_jurisdiction_query() -> None:
    result = extract_legal_entities("Delaware breach of contract requirements after 2020.")

    assert result.jurisdictions == ["Delaware"]
    assert result.legal_topics == ["breach"]
    assert result.filters.jurisdiction == ["Delaware"]
    assert result.filters.date_from == "2020-01-01"


def test_extract_legal_entities_court_query() -> None:
    result = extract_legal_entities("Recent Supreme Court appeal on negligence.")

    assert result.courts == ["Supreme Court"]
    assert "appeal" in result.procedural_posture
    assert "recent_without_explicit_timeframe" in result.ambiguity_notes


def test_extract_legal_entities_statute_query() -> None:
    result = extract_legal_entities("Interpret GDPR Section 5 for customer data policy.")

    assert "GDPR" in result.laws_or_regulations
    assert "Section 5" in result.legal_citations
    assert "policy" in result.document_types


def test_extract_legal_entities_clause_extraction() -> None:
    result = extract_legal_entities("Need indemnity clause language in a contract.")

    assert result.clause_types == ["indemnity clause"]
    assert result.document_types == ["contract"]
    assert result.filters.clause_type == ["indemnity clause"]


def test_extract_legal_entities_procedural_query() -> None:
    result = extract_legal_entities("Motion to dismiss for fraud in federal district court.")

    assert "motion to dismiss" in result.procedural_posture
    assert "fraud" in result.causes_of_action
    assert result.jurisdictions == ["Federal"]
    assert result.courts == ["District Court"]


def test_extract_legal_entities_ambiguous_query() -> None:
    result = extract_legal_entities("What is the governing law clause?")

    assert "governing law clause" in result.clause_types
    assert "governing_law_without_jurisdiction" in result.ambiguity_notes
    assert result.jurisdictions == []


def test_extract_legal_entities_empty_query() -> None:
    result = extract_legal_entities("  ")

    assert result.original_query == "  "
    assert result.warnings == ["empty_input"]
    assert result.document_types == []


def test_extract_legal_entities_determinism() -> None:
    query = "California summary judgment negligence before 2022."
    first = extract_legal_entities(query)
    second = extract_legal_entities(query)

    assert first == second


def test_extract_legal_entities_filter_correctness() -> None:
    result = extract_legal_entities("New York Court of Appeal termination clause after 2021.")

    assert result.filters.jurisdiction == result.jurisdictions
    assert result.filters.court == result.courts
    assert result.filters.document_type == result.document_types
    assert result.filters.clause_type == result.clause_types
    assert result.filters.date_from == "2021-01-01"
    assert result.filters.date_to is None


def test_extract_legal_entities_non_conflation_validation() -> None:
    result = extract_legal_entities("Find UCC Rule 12 in Delaware Chancery Court cases.")

    assert "UCC" in result.laws_or_regulations
    assert "Rule 12" in result.legal_citations
    assert "Delaware" in result.jurisdictions
    assert "Chancery Court" in result.courts
    assert "UCC" not in result.legal_citations


def test_rewrite_query_expands_party_role_entity_query_for_intro_line_retrieval() -> None:
    query = "who is the employer?"

    result = rewrite_query(query)

    assert result.rewrite_notes == "party_role_entity_query_expansion"
    assert "by and between" in result.rewritten_query.lower()
    assert "employer" in result.rewritten_query.lower()


def test_rewrite_query_expands_matter_document_metadata_query_for_caption_header_retrieval() -> None:
    query = "what is the file number?"

    result = rewrite_query(query)

    assert result.rewrite_notes == "matter_document_metadata_query_expansion"
    assert "caption" in result.rewritten_query.lower()
    assert "matter information" in result.rewritten_query.lower()


def test_rewrite_query_expands_employment_lifecycle_query_for_lifecycle_clause_retrieval() -> None:
    query = "When did employment start?"

    result = rewrite_query(query)

    assert result.rewrite_notes == "employment_contract_lifecycle_query_expansion"
    assert "commencement" in result.rewritten_query.lower()
    assert "compensation" in result.rewritten_query.lower()


def test_rewrite_query_expands_employment_mitigation_query_for_mitigation_evidence_retrieval() -> None:
    query = "What mitigation efforts were made?"

    result = rewrite_query(query)

    assert result.rewrite_notes == "employment_mitigation_query_expansion"
    assert "job search log" in result.rewritten_query.lower()
    assert "offer letter" in result.rewritten_query.lower()
