from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import (
    FinalAnswerModel,
    LegalRagDependencies,
    build_answer_graph,
    build_full_legal_rag_graph,
    default_legal_rag_state,
    run_legal_rag_turn,
    run_legal_rag_turn_with_state,
)
from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.orchestration.retrieval_graph import (
    DecompositionPlan,
    QueryRoutingDecision,
    RetrievalDependencies,
    RetrievalGraphConfig,
    SubQueryPlan,
    SubqueryCoverageRecord,
    SubquerySupportClassification,
)
from agentic_rag.chunking import MarkdownParentChildChunker
from agentic_rag.types import Document
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answerability import AnswerabilityAssessment, assess_answerability
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.context_processing import CompressContextResult, CompressedParentChunk
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult


@dataclass
class FakeServices:
    classifier: QueryRoutingDecision
    rewritten_query: str = ""
    hybrid_results: list[HybridSearchResult] = field(default_factory=list)
    hybrid_results_by_query: dict[str, list[HybridSearchResult]] = field(default_factory=dict)
    reranked_results: list[RerankedChunkResult] = field(default_factory=list)
    parent_results: list[ParentChunkResult] = field(default_factory=list)
    compressed_items: list[CompressedParentChunk] = field(default_factory=list)

    answer_result: object | None = None
    answer_raises: bool = False

    answer_calls: list[dict[str, Any]] = field(default_factory=list)
    hybrid_calls: list[dict[str, Any]] = field(default_factory=list)
    answerability_raises: bool = False
    answerability_result: AnswerabilityAssessment | None = None

    def retrieval_dependencies(self) -> RetrievalDependencies:
        return RetrievalDependencies(
            rewrite_query=self.rewrite_query,
            extract_legal_entities=self.extract_legal_entities,
            hybrid_search=self.hybrid_search,
            rerank_chunks=self.rerank_chunks,
            retrieve_parent_chunks=self.retrieve_parent_chunks,
            compress_context=self.compress_context,
            classify_query_state=self.classify,
        )

    def as_dependencies(self) -> LegalRagDependencies:
        return LegalRagDependencies(
            retrieval=self.retrieval_dependencies(),
            generate_grounded_answer=self.generate_answer,
            assess_answerability=self.assess_answerability,
        )

    def classify(self, *_: Any, **__: Any) -> QueryRoutingDecision:
        return self.classifier

    def rewrite_query(self, query: str, **_: Any) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=self.rewritten_query or query,
            used_conversation_context=False,
            rewrite_notes="test",
        )

    def extract_legal_entities(self, query: str) -> LegalEntityExtractionResult:
        return LegalEntityExtractionResult(
            original_query=query,
            normalized_query=query,
            document_types=[],
            legal_topics=[],
            jurisdictions=[],
            courts=[],
            laws_or_regulations=[],
            legal_citations=[],
            clause_types=[],
            parties=[],
            dates=[],
            time_constraints=[],
            obligations=[],
            remedies=[],
            procedural_posture=[],
            causes_of_action=[],
            factual_entities=[],
            keywords=[],
            filters=LegalEntityFilters(),
            ambiguity_notes=[],
            warnings=[],
            extraction_notes=[],
        )

    def hybrid_search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[HybridSearchResult]:
        self.hybrid_calls.append({"query": query, "filters": filters, "top_k": top_k})
        if query in self.hybrid_results_by_query:
            return self.hybrid_results_by_query[query]
        return self.hybrid_results

    def rerank_chunks(self, chunks: list[HybridSearchResult], query: str) -> list[RerankedChunkResult]:
        _ = (chunks, query)
        return self.reranked_results

    def retrieve_parent_chunks(self, parent_ids: list[str]) -> list[ParentChunkResult]:
        _ = parent_ids
        return self.parent_results

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        return CompressContextResult(items=tuple(self.compressed_items), total_original_chars=100, total_compressed_chars=80)

    def generate_answer(self, context: list[object], query: str) -> object:
        self.answer_calls.append({"context": context, "query": query})
        if self.answer_raises:
            raise RuntimeError("boom")
        if self.answer_result is not None:
            return self.answer_result
        return GenerateAnswerResult(
            answer_text="Grounded answer",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id="p1",
                    document_id="doc-1",
                    source_name="source",
                    heading="Section 1",
                    supporting_excerpt="Text",
                )
            ],
            warnings=[],
        )

    def assess_answerability(self, query: str, query_understanding: Any, retrieved_context: list[object]) -> AnswerabilityAssessment:
        if self.answerability_raises:
            raise RuntimeError("answerability_fail")
        if self.answerability_result is not None:
            return self.answerability_result
        return assess_answerability(query=query, query_understanding=query_understanding, retrieved_context=retrieved_context)


def _decision(*, rewrite: bool = False) -> QueryRoutingDecision:
    return QueryRoutingDecision(
        original_query="q",
        normalized_query="q",
        question_type="other_query",
        is_followup=False,
        is_context_dependent=False,
        use_conversation_context=False,
        is_document_scoped=False,
        should_rewrite=rewrite,
        should_extract_entities=False,
        should_retrieve=True,
        may_need_decomposition=False,
        resolved_document_hints=[],
        resolved_topic_hints=[],
        resolved_clause_hints=[],
        answerability_expectation="general_grounded_response",
        refers_to_prior_document_scope=False,
        refers_to_prior_clause_or_topic=False,
        ambiguity_notes=[],
        routing_notes=["test"],
        warnings=[],
    )


def _hybrid(child_id: str, parent_id: str) -> HybridSearchResult:
    return HybridSearchResult(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text="termination clause",
        hybrid_score=0.8,
    )


def _reranked(child_id: str, parent_id: str) -> RerankedChunkResult:
    return RerankedChunkResult(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text="reranked text",
        rerank_score=0.9,
        original_score=0.8,
    )


def _parent(
    parent_id: str,
    text: str = "This section states that either party may terminate the agreement with written notice, subject to accrued obligations and applicable governing law requirements.",
) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text=text,
        source="test",
        source_name="test-source",
        heading_text="Section A",
    )


def test_happy_path_end_to_end() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    result = run_legal_rag_turn(query="What is the termination rule?", dependencies=services.as_dependencies())
    assert isinstance(result, FinalAnswerModel)
    assert result.grounded is True
    assert result.citations[0].parent_chunk_id == "p1"


def test_compressed_context_preference() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="raw parent")],
        compressed_items=[
            CompressedParentChunk(
                parent_chunk_id="p1",
                document_id="doc-1",
                source="test",
                source_name="compressed-source",
                compressed_text="This compressed section states that either party may terminate with written notice while honoring accrued obligations under governing law.",
            )
        ],
    )
    run_legal_rag_turn(
        query="Q",
        dependencies=services.as_dependencies(),
        retrieval_config=RetrievalGraphConfig(compress_if_parent_chunks_gte=1),
    )
    assert services.answer_calls
    used_context = services.answer_calls[0]["context"]
    assert len(used_context) == 1
    assert isinstance(used_context[0], CompressedParentChunk)


def test_parent_chunks_fallback_when_compressed_empty() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[
            _parent(
                "p1",
                text="Raw parent section states that either party may terminate with written notice and must satisfy accrued obligations under the agreement.",
            )
        ],
        compressed_items=[],
    )
    run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    used_context = services.answer_calls[0]["context"]
    assert isinstance(used_context[0], ParentChunkResult)


def test_party_role_queries_use_parent_context_not_heading_only_compressed_context() -> None:
    agreement_intro = (
        "EMPLOYMENT AGREEMENT\n\n"
        "BETWEEN:\n"
        "Acme Holdings LLC (the \"Employer\")\n"
        "AND:\n"
        "Jane Smith (the \"Employee\")\n\n"
        "1. POSITION AND DUTIES\n"
        "The Employee will perform assigned duties."
    )
    services = FakeServices(
        classifier=understand_query("who is the employer?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text=agreement_intro)],
        compressed_items=[
            CompressedParentChunk(
                parent_chunk_id="p1",
                document_id="doc-1",
                source="test",
                source_name="compressed-source",
                heading_text="EMPLOYMENT AGREEMENT",
                compressed_text="EMPLOYMENT AGREEMENT\n1. POSITION AND DUTIES",
            )
        ],
    )
    result, state = run_legal_rag_turn_with_state(
        query="who is the employer?",
        dependencies=services.as_dependencies(),
        retrieval_config=RetrievalGraphConfig(compress_if_parent_chunks_gte=1),
    )
    used_context = services.answer_calls[0]["context"]
    assert isinstance(used_context[0], ParentChunkResult)
    assert "Acme Holdings LLC" in used_context[0].text
    assert result.sufficient_context is True
    assert any(note == "party_role_assignment_resolved" for note in state["answerability_result"].evidence_notes)


def test_party_role_runtime_family_answers_short_agreement_intro() -> None:
    agreement_intro = (
        "This Employment Agreement is made effective as of January 1, 2025, by and between "
        "Acme Holdings LLC (the \"Employer\") and Jane Smith (the \"Employee\").\n"
        "1. POSITION AND DUTIES\n"
        "The Employee will perform assigned duties."
    )
    services = FakeServices(
        classifier=understand_query("who is the employer?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text=agreement_intro)],
    )
    runtime_dependencies = LegalRagDependencies(
        retrieval=services.retrieval_dependencies(),
        generate_grounded_answer=generate_answer,
        assess_answerability=assess_answerability,
    )
    checks = {
        "who is the employer?": "Acme Holdings LLC",
        "who is the employee?": "Jane Smith",
        "who are the parties?": "Acme Holdings LLC and Jane Smith",
        "who is the hiring company?": "Acme Holdings LLC",
    }
    for query, expected in checks.items():
        result, state = run_legal_rag_turn_with_state(query=query, dependencies=runtime_dependencies)
        assert result.sufficient_context is True
        assert expected in result.answer_text
        assert any(note == "party_role_assignment_resolved" for note in state["answerability_result"].evidence_notes)
        assert any(warning == "party_role_resolution_invoked" for warning in result.warnings)

    verification_result, verification_state = run_legal_rag_turn_with_state(
        query="Is this agreement between Acme Holdings LLC and Jane Smith?",
        dependencies=runtime_dependencies,
    )
    assert verification_result.sufficient_context is True
    assert "Yes" in verification_result.answer_text
    assert any(warning == "party_role_resolution_invoked" for warning in verification_result.warnings)
    assert any(
        note == "agreement_between_pair_confirmed_from_extracted_parties"
        for note in verification_state["answerability_result"].evidence_notes
    )


def test_party_role_runtime_parent_expansion_includes_intro_parent_chunk() -> None:
    services = FakeServices(
        classifier=understand_query("who is the employer?"),
        hybrid_results=[
            _hybrid("c-heading", "p1"),
            _hybrid("c-section", "p3"),
        ],
        reranked_results=[
            _reranked("c-heading", "p1"),
            _reranked("c-section", "p3"),
        ],
        parent_results=[
            _parent("p1", text="EMPLOYMENT AGREEMENT\nEffective Date: January 1, 2025."),
            _parent(
                "p2",
                text=(
                    "This Employment Agreement is between Acme Holdings LLC (the \"Employer\") and "
                    "Jane Smith (the \"Employee\")."
                ),
            ),
            _parent("p3", text="1. POSITION AND DUTIES\nThe Employee will perform assigned duties."),
        ],
    )
    services.hybrid_results_by_query[
        "who is the employer? agreement preamble intro between and employer employee parties definitions"
    ] = [
        _hybrid(
            "c-intro",
            "p2",
        )
    ]

    runtime_dependencies = LegalRagDependencies(
        retrieval=services.retrieval_dependencies(),
        generate_grounded_answer=generate_answer,
        assess_answerability=assess_answerability,
    )
    result, state = run_legal_rag_turn_with_state(query="who is the employer?", dependencies=runtime_dependencies)

    assert result.sufficient_context is True
    assert "Acme Holdings LLC" in result.answer_text
    debug = state["answerability_result"].party_role_resolution_debug
    assert debug is not None
    assert "p2" in debug.party_role_resolution_checked_parent_ids
    assert "p2" in debug.party_role_resolution_intro_pattern_parent_ids
    assert any("Acme Holdings LLC" in preview.preview_start for preview in debug.checked_parent_previews)


def test_party_role_runtime_from_chunked_agreement_preserves_intro_preamble_in_checked_preview() -> None:
    text = (
        "# EMPLOYMENT AGREEMENT\n"
        "This Employment Agreement is made effective as of January 1, 2025.\n"
        "## BETWEEN:\n"
        "Aurora Data Systems Inc. (the \"Employer\")\n"
        "## AND:\n"
        "Daniel Reza Mohammadi (the \"Employee\")\n"
        "## 1. POSITION AND DUTIES\n"
        "The Employee will perform assigned duties.\n"
        "## 2. TERM\n"
        "The initial term is one year.\n"
    )
    chunked = MarkdownParentChildChunker().chunk(
        Document(id="doc-1", text=text, metadata={"source": "fixtures/agreement.md", "source_name": "agreement.md"})
    )
    parents = [
        ParentChunkResult(
            parent_chunk_id=parent.parent_chunk_id,
            document_id=parent.document_id,
            text=parent.text,
            source=parent.source,
            source_name=parent.source_name,
            heading_text=parent.heading_text,
            heading_path=parent.heading_path,
            metadata={},
        )
        for parent in chunked.parent_chunks
    ]
    intro_parent = parents[0]
    first_numbered_parent = next(parent for parent in parents if parent.heading_text == "1. POSITION AND DUTIES")

    services = FakeServices(
        classifier=understand_query("who is the employer?"),
        hybrid_results=[_hybrid("c-section", first_numbered_parent.parent_chunk_id)],
        reranked_results=[_reranked("c-section", first_numbered_parent.parent_chunk_id)],
        parent_results=parents,
    )
    services.hybrid_results_by_query[
        "who is the employer? agreement preamble intro between and employer employee parties definitions"
    ] = [
        HybridSearchResult(
            child_chunk_id="c-intro",
            parent_chunk_id=intro_parent.parent_chunk_id,
            document_id="doc-1",
            text=intro_parent.text,
            hybrid_score=0.95,
        )
    ]

    runtime_dependencies = LegalRagDependencies(
        retrieval=services.retrieval_dependencies(),
        generate_grounded_answer=generate_answer,
        assess_answerability=assess_answerability,
    )
    result, state = run_legal_rag_turn_with_state(query="who is the employer?", dependencies=runtime_dependencies)

    assert result.sufficient_context is True
    assert "Aurora Data Systems Inc." in result.answer_text
    debug = state["answerability_result"].party_role_resolution_debug
    assert debug is not None
    assert debug.party_role_resolution_intro_pattern_parent_ids
    assert any("Aurora Data Systems Inc." in preview.preview_start for preview in debug.checked_parent_previews)
    assert any("Daniel Reza Mohammadi" in preview.preview_start for preview in debug.checked_parent_previews)

    employee_result, _ = run_legal_rag_turn_with_state(query="who is the employee?", dependencies=runtime_dependencies)
    parties_result, _ = run_legal_rag_turn_with_state(query="who are the parties?", dependencies=runtime_dependencies)

    assert employee_result.sufficient_context is True
    assert "Daniel Reza Mohammadi" in employee_result.answer_text
    assert parties_result.sufficient_context is True
    assert "Aurora Data Systems Inc." in parties_result.answer_text
    assert "Daniel Reza Mohammadi" in parties_result.answer_text


def test_party_role_runtime_heading_only_context_fails_safely() -> None:
    services = FakeServices(
        classifier=understand_query("who is the employer?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="EMPLOYMENT AGREEMENT\n1. POSITION AND DUTIES")],
    )
    result, state = run_legal_rag_turn_with_state(query="who is the employer?", dependencies=services.as_dependencies())
    assert result.sufficient_context is False
    assert any(note.startswith("party_role_resolution_outcome:") for note in state["answerability_result"].evidence_notes)
    assert any(
        warning == "answerability_gate:fact_not_found" or warning.startswith("answerability_gate:")
        for warning in result.warnings
    )


def test_empty_context_insufficiency_response() -> None:
    services = FakeServices(classifier=_decision(), hybrid_results=[], reranked_results=[], parent_results=[])
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert "does not contain enough information" in result.answer_text


def test_partial_context_flags_preserved() -> None:
    partial = GenerateAnswerResult(
        answer_text="Partial grounded answer",
        grounded=True,
        sufficient_context=False,
        citations=[AnswerCitation(parent_chunk_id="p1", document_id=None, source_name=None, heading=None, supporting_excerpt="x")],
        warnings=["partial"],
    )
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
        answer_result=partial,
    )
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert result.grounded is False
    assert result.sufficient_context is False
    assert any("fallback_after_sufficient_gate:generate_answer_returned_insufficient" in warning for warning in result.warnings)


def test_citation_preservation() -> None:
    citation = AnswerCitation(
        parent_chunk_id="p-preserve",
        document_id="d1",
        source_name="src",
        heading="H1",
        supporting_excerpt="Excerpt",
    )
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
        answer_result=GenerateAnswerResult(
            answer_text="A",
            grounded=True,
            sufficient_context=True,
            citations=[citation],
            warnings=[],
        ),
    )
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert result.citations == [citation]


def test_answer_generation_failure_fallback() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
        answer_raises=True,
    )
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert any("answer_generation_failed" in warning for warning in result.warnings)


def test_finalization_robustness_on_malformed_answer_output() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
        answer_result={"answer_text": "bad", "grounded": True, "sufficient_context": True, "citations": [{"document_id": "d1"}]},
    )
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []


def test_runner_returns_only_final_typed_model() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert isinstance(result, FinalAnswerModel)


def test_determinism_sanity_same_input_equivalent_output() -> None:
    services = FakeServices(
        classifier=_decision(rewrite=True),
        rewritten_query="rewritten",
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    result_a = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    result_b = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert result_a == result_b


def test_build_full_graph_composition_works() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    graph = build_full_legal_rag_graph(services.as_dependencies())
    initial = default_legal_rag_state(query="Q")
    final_state = graph.invoke(initial)
    assert final_state["final_response_ready"] is True
    assert isinstance(final_state["final_answer"], FinalAnswerModel)


def test_answerability_gate_blocks_generation_on_insufficient_context() -> None:
    definition_decision = understand_query("what is employment agreement?")
    services = FakeServices(
        classifier=definition_decision,
        hybrid_results=[_hybrid("c1", "p1"), _hybrid("c2", "p2")],
        reranked_results=[_reranked("c1", "p1"), _reranked("c2", "p2")],
        parent_results=[
            _parent("p1", text="Employment Agreement"),
            _parent(
                "p2",
                text="This agreement is governed by New York law and may be terminated with thirty days written notice.",
            ),
        ],
    )
    result = run_legal_rag_turn(query="what is employment agreement?", dependencies=services.as_dependencies())
    assert result.sufficient_context is False
    assert len(services.answer_calls) == 0
    assert any("answerability_gate" in warning for warning in result.warnings)


def test_clause_lookup_success_routes_to_generation() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Confidentiality obligations survive termination.")],
        answerability_result=AnswerabilityAssessment(
            original_query="what does the document say about confidentiality?",
            question_type="document_content_query",
            answerability_expectation="clause_lookup",
            has_relevant_context=True,
            sufficient_context=True,
            partially_supported=False,
            should_answer=True,
            support_level="sufficient",
            insufficiency_reason=None,
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Confidentiality"],
            evidence_notes=[],
            warnings=[],
        ),
    )
    result = run_legal_rag_turn(query="what does the document say about confidentiality?", dependencies=services.as_dependencies())
    assert len(services.answer_calls) == 1
    assert result.grounded is True


def test_response_route_is_traceable_for_generate_path() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Confidentiality obligations survive termination.")],
        answerability_result=AnswerabilityAssessment(
            original_query="what does the document say about confidentiality?",
            question_type="document_content_query",
            answerability_expectation="clause_lookup",
            has_relevant_context=True,
            sufficient_context=True,
            partially_supported=False,
            should_answer=True,
            support_level="sufficient",
            insufficiency_reason=None,
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Confidentiality"],
            evidence_notes=[],
            warnings=[],
        ),
    )
    _, state = run_legal_rag_turn_with_state(query="what does the document say about confidentiality?", dependencies=services.as_dependencies())
    assert state["response_route"] == "generate_answer"


def test_response_route_is_traceable_for_insufficient_path() -> None:
    definition_decision = understand_query("what is employment agreement?")
    services = FakeServices(
        classifier=definition_decision,
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Employment Agreement")],
    )
    _, state = run_legal_rag_turn_with_state(query="what is employment agreement?", dependencies=services.as_dependencies())
    assert state["response_route"] == "build_insufficient_response"


def test_sufficient_gate_cannot_silently_downgrade_to_insufficient() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Confidentiality obligations survive termination.")],
        answerability_result=AnswerabilityAssessment(
            original_query="what does the document say about confidentiality?",
            question_type="document_content_query",
            answerability_expectation="clause_lookup",
            has_relevant_context=True,
            sufficient_context=True,
            partially_supported=False,
            should_answer=True,
            support_level="sufficient",
            insufficiency_reason=None,
            matched_parent_chunk_ids=["p1"],
            matched_headings=["Confidentiality"],
            evidence_notes=[],
            warnings=[],
        ),
        answer_result=GenerateAnswerResult(
            answer_text="unexpected insufficient",
            grounded=False,
            sufficient_context=False,
            citations=[],
            warnings=[],
        ),
    )
    result, state = run_legal_rag_turn_with_state(
        query="what does the document say about confidentiality?",
        dependencies=services.as_dependencies(),
    )
    assert result.sufficient_context is False
    assert state["response_route"] == "fallback_finalizer:generate_answer_returned_insufficient"
    assert any("fallback_after_sufficient_gate" in warning for warning in result.warnings)


def test_fact_extraction_insufficiency_skips_generation() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="General terms only")],
        answerability_result=AnswerabilityAssessment(
            original_query="who are the parties?",
            question_type="extractive_fact_query",
            answerability_expectation="fact_extraction",
            has_relevant_context=True,
            sufficient_context=False,
            partially_supported=False,
            should_answer=False,
            support_level="weak",
            insufficiency_reason="fact_not_found",
            matched_parent_chunk_ids=["p1"],
            matched_headings=[],
            evidence_notes=["requested_fact_missing_from_context"],
            warnings=[],
        ),
    )
    result = run_legal_rag_turn(query="who are the parties?", dependencies=services.as_dependencies())
    assert len(services.answer_calls) == 0
    assert result.sufficient_context is False


def test_summary_insufficiency_is_not_treated_as_sufficient() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Only one clause available")],
        answerability_result=AnswerabilityAssessment(
            original_query="summarize this agreement",
            question_type="document_summary_query",
            answerability_expectation="summary",
            has_relevant_context=True,
            sufficient_context=False,
            partially_supported=True,
            should_answer=False,
            support_level="partial",
            insufficiency_reason="summary_not_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=[],
            evidence_notes=["single_chunk_insufficient_for_summary"],
            warnings=[],
        ),
    )
    result = run_legal_rag_turn(query="summarize this agreement", dependencies=services.as_dependencies())
    assert len(services.answer_calls) == 0
    assert result.sufficient_context is False
    assert any("answerability_gate:summary_not_supported" == warning for warning in result.warnings)


def test_partial_support_routes_to_insufficient_response_for_v1_policy() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Topic mention without full answer")],
        answerability_result=AnswerabilityAssessment(
            original_query="compare obligations",
            question_type="comparison_query",
            answerability_expectation="comparison",
            has_relevant_context=True,
            sufficient_context=False,
            partially_supported=True,
            should_answer=False,
            support_level="partial",
            insufficiency_reason="comparison_not_supported",
            matched_parent_chunk_ids=["p1"],
            matched_headings=[],
            evidence_notes=["one_sided_evidence_only"],
            warnings=[],
        ),
    )
    result = run_legal_rag_turn(query="compare obligations", dependencies=services.as_dependencies())
    assert len(services.answer_calls) == 0
    assert result.sufficient_context is False


def test_answerability_failure_falls_back_to_safe_insufficiency() -> None:
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="some context")],
        answerability_raises=True,
    )
    result = run_legal_rag_turn(query="Q", dependencies=services.as_dependencies())
    assert len(services.answer_calls) == 0
    assert result.sufficient_context is False
    assert any("answerability_assessment_failed" in warning for warning in result.warnings)


def test_regression_definition_title_only_does_not_generate_clause_answer() -> None:
    definition_decision = understand_query("what is employment agreement?")
    services = FakeServices(
        classifier=definition_decision,
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Employment Agreement\nTermination Without Cause...")],
    )
    result = run_legal_rag_turn(query="what is employment agreement?", dependencies=services.as_dependencies())
    assert len(services.answer_calls) == 0
    assert "Termination Without Cause" not in result.answer_text


def test_warning_aggregation_dedupes_identical_definition_not_supported_warnings() -> None:
    services = FakeServices(
        classifier=understand_query("what is employment agreement?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Employment Agreement\nTermination Without Cause...")],
    )
    result = run_legal_rag_turn(query="what is employment agreement?", dependencies=services.as_dependencies())
    assert len(result.warnings) == len(set(result.warnings))


def test_definition_required_insufficient_response_is_more_specific_for_title_only_or_generic_label_case() -> None:
    title_only_parent = ParentChunkResult(
        parent_chunk_id="p1",
        document_id="doc-1",
        text="Employment Agreement",
        source="test",
        source_name="Employment Agreement",
        heading_text="Employment Agreement",
    )
    services = FakeServices(
        classifier=understand_query("what is employment agreement?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[title_only_parent],
    )
    result = run_legal_rag_turn(query="what is employment agreement?", dependencies=services.as_dependencies())
    assert "does not define the term itself" not in result.answer_text
    assert (
        result.answer_text
        == "Direct answer: I do not see a definition of 'employment agreement' in the retrieved context. "
        "It appears as a document title or label, not as a defined term or clause."
    )
    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []




def test_non_definition_heading_only_insufficient_response_does_not_use_definition_wording() -> None:
    non_definition_heading_only = AnswerabilityAssessment(
        original_query="what are notice requirements?",
        question_type="document_content_query",
        answerability_expectation="clause_lookup",
        has_relevant_context=True,
        sufficient_context=False,
        partially_supported=False,
        should_answer=False,
        support_level="insufficient",
        insufficiency_reason="only_title_or_heading_match",
        matched_parent_chunk_ids=["p1"],
        matched_headings=["Notice"],
        evidence_notes=["weakness_signal:title_only_signal_without_body"],
        warnings=["heading_only_context"],
    )
    title_only_parent = ParentChunkResult(
        parent_chunk_id="p1",
        document_id="doc-1",
        text="Notice",
        source="test",
        source_name="test",
        heading_text="Notice",
    )
    services = FakeServices(
        classifier=understand_query("what are notice requirements?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[title_only_parent],
        answerability_result=non_definition_heading_only,
    )

    result = run_legal_rag_turn(query="what are notice requirements?", dependencies=services.as_dependencies())

    assert "does not define the term itself" not in result.answer_text
    assert "I do not see a definition" not in result.answer_text


def test_definition_intent_heading_only_insufficient_response_still_uses_improved_definition_wording() -> None:
    title_only_parent = ParentChunkResult(
        parent_chunk_id="p1",
        document_id="doc-1",
        text="Employment Agreement",
        source="test",
        source_name="Employment Agreement",
        heading_text="Employment Agreement",
    )
    services = FakeServices(
        classifier=understand_query("what is employment agreement?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[title_only_parent],
    )

    result = run_legal_rag_turn(query="what is employment agreement?", dependencies=services.as_dependencies())

    assert result.answer_text == (
        "Direct answer: I do not see a definition of 'employment agreement' in the retrieved context. "
        "It appears as a document title or label, not as a defined term or clause."
    )


def test_warning_dedupe_still_holds_after_followup_fix() -> None:
    services = FakeServices(
        classifier=understand_query("what is employment agreement?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Employment Agreement\nTermination Without Cause...")],
    )

    result = run_legal_rag_turn(query="what is employment agreement?", dependencies=services.as_dependencies())

    assert len(result.warnings) == len(set(result.warnings))

def test_existing_valid_clause_lookup_response_remains_unchanged() -> None:
    sufficient_assessment = AnswerabilityAssessment(
        original_query="what is confidentiality?",
        question_type="document_content_query",
        answerability_expectation="clause_lookup",
        has_relevant_context=True,
        sufficient_context=True,
        partially_supported=False,
        should_answer=True,
        support_level="sufficient",
        insufficiency_reason=None,
        matched_parent_chunk_ids=["p1"],
        matched_headings=["Confidentiality"],
        evidence_notes=["operative_clause_language_detected"],
        warnings=[],
    )
    services = FakeServices(
        classifier=_decision(),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
        answerability_result=sufficient_assessment,
    )
    result = run_legal_rag_turn(query="what is confidentiality?", dependencies=services.as_dependencies())
    assert result.sufficient_context is True
    assert result.grounded is True
    assert result.answer_text == "Grounded answer"


def test_existing_unsupported_definition_routing_remains_unchanged_except_wording_and_warning_dedupe() -> None:
    title_only_parent = ParentChunkResult(
        parent_chunk_id="p1",
        document_id="doc-1",
        text="Employment Agreement",
        source="test",
        source_name="Employment Agreement",
        heading_text="Employment Agreement",
    )
    services = FakeServices(
        classifier=understand_query("what is employment agreement?"),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[title_only_parent],
    )
    result, state = run_legal_rag_turn_with_state(
        query="what is employment agreement?",
        dependencies=services.as_dependencies(),
    )
    assessment = state["answerability_result"]
    assert assessment.should_answer is False
    assert assessment.sufficient_context is False
    assert assessment.insufficiency_reason == "only_title_or_heading_match"
    assert state["response_route"] == "build_insufficient_response"
    assert len(result.warnings) == len(set(result.warnings))


def _sufficient_assessment(query: str = "compare clauses") -> AnswerabilityAssessment:
    return AnswerabilityAssessment(
        original_query=query,
        question_type="comparison_query",
        answerability_expectation="comparison",
        has_relevant_context=True,
        sufficient_context=True,
        partially_supported=False,
        should_answer=True,
        support_level="sufficient",
        insufficiency_reason=None,
        matched_parent_chunk_ids=["p1"],
        matched_headings=["Comparison"],
        evidence_notes=["baseline_sufficient_answerability"],
        warnings=[],
    )


def _plan() -> DecompositionPlan:
    return DecompositionPlan(
        should_decompose=True,
        root_question="Compare governing law and dispute resolution.",
        strategy="comparison",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="Find governing law text.",
                purpose="required",
                required=True,
                expected_answer_type="cross_reference",
            ),
            SubQueryPlan(
                id="sq-2",
                question="Find dispute resolution text.",
                purpose="required",
                required=True,
                expected_answer_type="cross_reference",
            ),
            SubQueryPlan(
                id="sq-3",
                question="Find optional venue text.",
                purpose="optional",
                required=False,
                expected_answer_type="cross_reference",
            ),
        ],
    )


def _coverage(
    sq1: SubquerySupportClassification,
    sq2: SubquerySupportClassification,
    sq3: SubquerySupportClassification = "supported",
) -> list[SubqueryCoverageRecord]:
    return [
        SubqueryCoverageRecord(
            subquery_id="sq-1",
            required=True,
            support_classification=sq1,
            evidence_child_chunk_ids=["c1"] if sq1 == "supported" else [],
            evidence_parent_chunk_ids=["p1"] if sq1 == "supported" else [],
            insufficiency_reason=None if sq1 == "supported" else "no_retrieval_evidence",
        ),
        SubqueryCoverageRecord(
            subquery_id="sq-2",
            required=True,
            support_classification=sq2,
            evidence_child_chunk_ids=["c2"] if sq2 == "supported" else [],
            evidence_parent_chunk_ids=["p2"] if sq2 == "supported" else [],
            insufficiency_reason=None if sq2 == "supported" else "no_retrieval_evidence",
        ),
        SubqueryCoverageRecord(
            subquery_id="sq-3",
            required=False,
            support_classification=sq3,
            evidence_child_chunk_ids=["c3"] if sq3 == "supported" else [],
            evidence_parent_chunk_ids=["p3"] if sq3 == "supported" else [],
            insufficiency_reason=None if sq3 == "supported" else "no_retrieval_evidence",
        ),
    ]


def _invoke_answer_graph(
    *,
    assessment: AnswerabilityAssessment,
    decomposition_plan: DecompositionPlan | None,
    subquery_coverage: list[SubqueryCoverageRecord],
) -> dict[str, Any]:
    app = build_answer_graph(
        answer_generator=lambda *_args, **_kwargs: GenerateAnswerResult(
            answer_text="Grounded answer",
            grounded=True,
            sufficient_context=True,
            citations=[AnswerCitation(parent_chunk_id="p1", document_id=None, source_name=None, heading=None, supporting_excerpt="x")],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: assessment,
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1")]
    state["decomposition_plan"] = decomposition_plan
    state["subquery_coverage"] = subquery_coverage
    return app.invoke(state)


def test_decomposed_query_with_all_required_subqueries_supported_can_be_sufficient() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=_coverage("supported", "supported", "supported"),
    )
    assert state["final_answer"].sufficient_context is True
    assert state["response_route"] == "generate_answer"


def test_decomposed_query_with_missing_required_subquery_is_not_fully_sufficient() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=[
            _coverage("supported", "supported")[0],
        ],
    )
    assessment = state["answerability_result"]
    assert assessment.sufficient_context is False
    assert assessment.should_answer is False
    assert assessment.support_level == "partial"
    assert state["response_route"] == "build_insufficient_response"


def test_decomposed_query_with_optional_subquery_missing_can_still_be_sufficient() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=_coverage("supported", "supported", "unsupported"),
    )
    assert state["answerability_result"].sufficient_context is True
    assert state["response_route"] == "generate_answer"


def test_decomposed_query_with_weak_required_support_routes_to_partial_or_insufficient_in_repo_consistent_way() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=_coverage("supported", "weak", "supported"),
    )
    assessment = state["answerability_result"]
    assert assessment.sufficient_context is False
    assert assessment.should_answer is False
    assert assessment.partially_supported is True
    assert assessment.support_level == "partial"
    assert assessment.insufficiency_reason == "partial_evidence_only"
    assert state["response_route"] == "build_insufficient_response"


def test_no_coverage_or_no_valid_decomposition_plan_falls_back_to_existing_answerability_behavior() -> None:
    no_coverage_state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=[],
    )
    no_plan_state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=None,
        subquery_coverage=_coverage("unsupported", "unsupported"),
    )
    assert no_coverage_state["answerability_result"].sufficient_context is True
    assert no_coverage_state["response_route"] == "generate_answer"
    assert no_plan_state["answerability_result"].sufficient_context is True
    assert no_plan_state["response_route"] == "generate_answer"


def test_non_decomposition_answerability_behavior_remains_unchanged() -> None:
    baseline = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=None,
        subquery_coverage=[],
    )
    assert baseline["answerability_result"] == _sufficient_assessment()
    assert baseline["response_route"] == "generate_answer"


def test_fallback_path_matches_main_answerability_behavior_if_applicable(monkeypatch: Any) -> None:
    import agentic_rag.orchestration.legal_rag_graph as legal_graph_module

    assessment = _sufficient_assessment()
    decomposition_plan = _plan()
    subquery_coverage = _coverage("supported", "weak", "supported")
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1")]
    state["decomposition_plan"] = decomposition_plan
    state["subquery_coverage"] = subquery_coverage

    main_app = build_answer_graph(
        answer_generator=lambda *_args, **_kwargs: GenerateAnswerResult(
            answer_text="Grounded answer",
            grounded=True,
            sufficient_context=True,
            citations=[AnswerCitation(parent_chunk_id="p1", document_id=None, source_name=None, heading=None, supporting_excerpt="x")],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: assessment,
    )
    main_state = main_app.invoke(state)

    monkeypatch.setattr(legal_graph_module, "StateGraph", None)
    fallback_app = legal_graph_module.build_answer_graph(
        answer_generator=lambda *_args, **_kwargs: GenerateAnswerResult(
            answer_text="Grounded answer",
            grounded=True,
            sufficient_context=True,
            citations=[AnswerCitation(parent_chunk_id="p1", document_id=None, source_name=None, heading=None, supporting_excerpt="x")],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: assessment,
    )
    fallback_state = fallback_app.invoke(state)

    assert main_state["answerability_result"] == fallback_state["answerability_result"]
    assert main_state["response_route"] == fallback_state["response_route"]


def test_supported_subquery_generates_grounded_subanswer_with_citations() -> None:
    app = build_answer_graph(
        answer_generator=lambda context, query: GenerateAnswerResult(
            answer_text=f"Grounded answer for {query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt="Text",
                )
            ],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment(),
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "supported", "unsupported")
    state = app.invoke(state)

    subanswers = state["subquery_subanswers"]
    assert len(subanswers) == 3
    supported = next(item for item in subanswers if item.subquery_id == "sq-1")
    assert supported.grounded is True
    assert supported.support_classification == "supported"
    assert supported.citations
    assert supported.citations[0].parent_chunk_id == "p1"


def test_multiple_supported_subqueries_generate_separate_subanswers() -> None:
    app = build_answer_graph(
        answer_generator=lambda context, query: GenerateAnswerResult(
            answer_text=f"Grounded answer for {query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt="Text",
                )
            ],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment(),
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2"), _parent("p3")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "supported", "unsupported")
    state = app.invoke(state)

    subanswers = state["subquery_subanswers"]
    assert len(subanswers) == 3
    assert [item.subquery_id for item in subanswers] == ["sq-1", "sq-2", "sq-3"]
    assert [item.support_classification for item in subanswers] == ["supported", "supported", "unsupported"]
    assert all(item.subquery_question for item in subanswers)


def test_unsupported_subquery_does_not_generate_fabricated_grounded_subanswer() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=_coverage("unsupported", "supported", "unsupported"),
    )

    unsupported = next(item for item in state["subquery_subanswers"] if item.subquery_id == "sq-1")
    assert unsupported.grounded is False
    assert unsupported.citations == []
    assert unsupported.support_classification == "unsupported"
    assert unsupported.answer_text


def test_subanswer_uses_only_subquery_linked_evidence_in_repo_consistent_way() -> None:
    app = build_answer_graph(
        answer_generator=lambda context, query: GenerateAnswerResult(
            answer_text=f"Scoped answer for {query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt="Scoped text",
                ),
                AnswerCitation(
                    parent_chunk_id="p-out-of-scope",
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt="Out-of-scope text",
                ),
            ],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment(),
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "supported", "unsupported")

    final_state = app.invoke(state)
    sq1 = next(item for item in final_state["subquery_subanswers"] if item.subquery_id == "sq-1")
    sq2 = next(item for item in final_state["subquery_subanswers"] if item.subquery_id == "sq-2")
    assert [citation.parent_chunk_id for citation in sq1.citations] == ["p1"]
    assert [citation.parent_chunk_id for citation in sq2.citations] == ["p2"]
    assert "subquery_citation_scope_adjusted" in sq1.warnings


def test_no_valid_decomposition_or_no_supported_subqueries_skips_subanswer_generation_cleanly() -> None:
    no_plan = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=None,
        subquery_coverage=_coverage("supported", "supported", "supported"),
    )
    unsupported_only = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=_coverage("unsupported", "unsupported", "unsupported"),
    )

    assert no_plan["subquery_subanswers"] == []
    assert len(unsupported_only["subquery_subanswers"]) == 3
    assert all(item.grounded is False for item in unsupported_only["subquery_subanswers"])


def test_non_decomposition_behavior_remains_unchanged_with_subanswers_layer() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=None,
        subquery_coverage=[],
    )
    assert state["final_answer"].answer_text == "Grounded answer"
    assert state["final_answer"].grounded is True
    assert state["subquery_subanswers"] == []


def test_fallback_or_shared_path_matches_main_behavior_if_applicable_for_subanswers(monkeypatch: Any) -> None:
    import agentic_rag.orchestration.legal_rag_graph as legal_graph_module

    assessment = _sufficient_assessment()
    decomposition_plan = _plan()
    subquery_coverage = _coverage("supported", "unsupported", "weak")
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2")]
    state["decomposition_plan"] = decomposition_plan
    state["subquery_coverage"] = subquery_coverage

    main_app = build_answer_graph(
        answer_generator=lambda *_args, **_kwargs: GenerateAnswerResult(
            answer_text="Grounded answer",
            grounded=True,
            sufficient_context=True,
            citations=[AnswerCitation(parent_chunk_id="p1", document_id=None, source_name=None, heading=None, supporting_excerpt="x")],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: assessment,
    )
    main_state = main_app.invoke(state)

    monkeypatch.setattr(legal_graph_module, "StateGraph", None)
    fallback_app = legal_graph_module.build_answer_graph(
        answer_generator=lambda *_args, **_kwargs: GenerateAnswerResult(
            answer_text="Grounded answer",
            grounded=True,
            sufficient_context=True,
            citations=[AnswerCitation(parent_chunk_id="p1", document_id=None, source_name=None, heading=None, supporting_excerpt="x")],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: assessment,
    )
    fallback_state = fallback_app.invoke(state)

    assert [item.model_dump() for item in main_state["subquery_subanswers"]] == [
        item.model_dump() for item in fallback_state["subquery_subanswers"]
    ]


def test_all_supported_required_subqueries_produce_fully_synthesized_final_answer() -> None:
    calls: list[str] = []

    def _generator(context: list[object], query: str) -> GenerateAnswerResult:
        calls.append(query)
        return GenerateAnswerResult(
            answer_text=f"Subanswer::{query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt=f"Excerpt::{query}",
                )
            ],
            warnings=[],
        )

    app = build_answer_graph(answer_generator=_generator, answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment())
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2"), _parent("p3")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "supported", "supported")

    final_state = app.invoke(state)
    final = final_state["final_answer"]
    assert final.sufficient_context is True
    assert "Direct answer (synthesized from grounded subanswers)" in final.answer_text
    assert "Find governing law text." in final.answer_text
    assert "Find dispute resolution text." in final.answer_text
    assert "Support gaps:" not in final.answer_text
    assert calls == ["Find governing law text.", "Find dispute resolution text.", "Find optional venue text."]


def test_missing_required_subquery_produces_partial_or_insufficient_final_answer_in_repo_consistent_way() -> None:
    app = build_answer_graph(
        answer_generator=lambda context, query: GenerateAnswerResult(
            answer_text=f"Subanswer::{query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt=f"Excerpt::{query}",
                )
            ],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment(),
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2"), _parent("p3")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "unsupported", "supported")

    final_state = app.invoke(state)
    final = final_state["final_answer"]
    assert final.sufficient_context is False
    assert final.grounded is True
    assert "Support gaps:" in final.answer_text
    assert "Required gap: Find dispute resolution text." in final.answer_text
    assert final_state["response_route"] == "build_insufficient_response"


def test_missing_optional_subquery_does_not_block_full_final_answer_when_required_support_is_complete() -> None:
    app = build_answer_graph(
        answer_generator=lambda context, query: GenerateAnswerResult(
            answer_text=f"Subanswer::{query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt=f"Excerpt::{query}",
                )
            ],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment(),
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2"), _parent("p3")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "supported", "unsupported")

    final = app.invoke(state)["final_answer"]
    assert final.sufficient_context is True
    assert "Optional gap: Find optional venue text." in final.answer_text


def test_no_supported_subanswers_produces_insufficient_final_answer() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=_plan(),
        subquery_coverage=_coverage("unsupported", "unsupported", "unsupported"),
    )
    final = state["final_answer"]
    assert final.sufficient_context is False
    assert final.grounded is False
    assert "does not contain enough information" in final.answer_text


def test_final_citations_are_aggregated_from_grounded_subanswers_only() -> None:
    app = build_answer_graph(
        answer_generator=lambda context, query: GenerateAnswerResult(
            answer_text=f"Subanswer::{query}",
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt=f"Excerpt::{query}",
                ),
                AnswerCitation(
                    parent_chunk_id="p-extra",
                    document_id="doc-1",
                    source_name="source",
                    heading="Section",
                    supporting_excerpt="extra",
                ),
            ],
            warnings=[],
        ),
        answerability_evaluator=lambda *_args, **_kwargs: _sufficient_assessment(),
    )
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "supported", "unsupported")
    final = app.invoke(state)["final_answer"]
    assert [citation.parent_chunk_id for citation in final.citations] == ["p1", "p2"]


def test_non_decomposition_final_answer_behavior_remains_unchanged() -> None:
    state = _invoke_answer_graph(
        assessment=_sufficient_assessment(),
        decomposition_plan=None,
        subquery_coverage=[],
    )
    assert state["final_answer"].answer_text == "Grounded answer"
    assert state["response_route"] == "generate_answer"


def test_fallback_or_shared_path_matches_main_behavior_if_applicable_for_final_synthesis(monkeypatch: Any) -> None:
    import agentic_rag.orchestration.legal_rag_graph as legal_graph_module

    assessment = _sufficient_assessment()
    state = default_legal_rag_state(query="compare clauses")
    state["query_classification"] = _decision()
    state["parent_chunks"] = [_parent("p1"), _parent("p2")]
    state["decomposition_plan"] = _plan()
    state["subquery_coverage"] = _coverage("supported", "unsupported", "supported")
    generator = lambda context, query: GenerateAnswerResult(
        answer_text=f"Subanswer::{query}",
        grounded=True,
        sufficient_context=True,
        citations=[
            AnswerCitation(
                parent_chunk_id=str(getattr(context[0], "parent_chunk_id", "missing")),
                document_id="doc-1",
                source_name="source",
                heading="Section",
                supporting_excerpt=f"Excerpt::{query}",
            )
        ],
        warnings=[],
    )
    main_app = build_answer_graph(answer_generator=generator, answerability_evaluator=lambda *_args, **_kwargs: assessment)
    main_state = main_app.invoke(state)

    monkeypatch.setattr(legal_graph_module, "StateGraph", None)
    fallback_app = legal_graph_module.build_answer_graph(
        answer_generator=generator,
        answerability_evaluator=lambda *_args, **_kwargs: assessment,
    )
    fallback_state = fallback_app.invoke(state)
    assert main_state["final_answer"] == fallback_state["final_answer"]
