from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import (
    FinalAnswerModel,
    LegalRagDependencies,
    build_full_legal_rag_graph,
    default_legal_rag_state,
    run_legal_rag_turn,
)
from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.orchestration.retrieval_graph import QueryRoutingDecision, RetrievalDependencies, RetrievalGraphConfig
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answerability import AnswerabilityAssessment, assess_answerability
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult
from agentic_rag.tools.context_processing import CompressContextResult, CompressedParentChunk
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult


@dataclass
class FakeServices:
    classifier: QueryRoutingDecision
    rewritten_query: str = ""
    hybrid_results: list[HybridSearchResult] = field(default_factory=list)
    reranked_results: list[RerankedChunkResult] = field(default_factory=list)
    parent_results: list[ParentChunkResult] = field(default_factory=list)
    compressed_items: list[CompressedParentChunk] = field(default_factory=list)

    answer_result: object | None = None
    answer_raises: bool = False

    answer_calls: list[dict[str, Any]] = field(default_factory=list)
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
        _ = (query, filters, top_k)
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
    assert result.grounded is True
    assert result.sufficient_context is False


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
