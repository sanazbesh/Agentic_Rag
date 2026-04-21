from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn, run_legal_rag_turn_with_state
from agentic_rag.orchestration.retrieval_graph import QueryRoutingDecision, RetrievalDependencies
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult
from agentic_rag.tools.answerability import AnswerabilityAssessment
from agentic_rag.tools.context_processing import CompressContextResult
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult


@dataclass
class _TracingServices:
    no_results: bool = False
    answerable: bool = True
    answer_text: str = "Grounded answer"
    hybrid_hits: list[HybridSearchResult] = field(default_factory=lambda: [_hybrid("c1", "p1")])
    reranked_hits: list[RerankedChunkResult] = field(default_factory=lambda: [_reranked("c1", "p1")])
    parents: list[ParentChunkResult] = field(default_factory=lambda: [_parent("p1")])

    def deps(self) -> LegalRagDependencies:
        retrieval = RetrievalDependencies(
            rewrite_query=self.rewrite_query,
            extract_legal_entities=self.extract_legal_entities,
            hybrid_search=self.hybrid_search,
            rerank_chunks=self.rerank_chunks,
            retrieve_parent_chunks=self.retrieve_parent_chunks,
            compress_context=self.compress_context,
            classify_query_state=self.classify,
        )
        return LegalRagDependencies(
            retrieval=retrieval,
            generate_grounded_answer=self.generate_answer,
            assess_answerability=self.assess_answerability,
        )

    def classify(self, *_: Any, **__: Any) -> QueryRoutingDecision:
        return QueryRoutingDecision(
            original_query="q",
            normalized_query="q",
            question_type="other_query",
            is_followup=False,
            is_context_dependent=False,
            use_conversation_context=False,
            is_document_scoped=True,
            should_rewrite=False,
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
            routing_notes=["legal_question_family:party_role_verification"],
            warnings=[],
        )

    def rewrite_query(self, query: str, **_: Any) -> QueryRewriteResult:
        return QueryRewriteResult(original_query=query, rewritten_query=query, used_conversation_context=False, rewrite_notes="none")

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

    def hybrid_search(self, _: str, *, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[HybridSearchResult]:
        _ = (filters, top_k)
        return [] if self.no_results else self.hybrid_hits

    def rerank_chunks(self, _: list[HybridSearchResult], __: str) -> list[RerankedChunkResult]:
        return [] if self.no_results else self.reranked_hits

    def retrieve_parent_chunks(self, _: list[str]) -> list[ParentChunkResult]:
        return [] if self.no_results else self.parents

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        return CompressContextResult(items=(), total_original_chars=0, total_compressed_chars=0)

    def assess_answerability(self, query: str, query_understanding: Any, retrieved_context: list[object]) -> AnswerabilityAssessment:
        _ = (query, query_understanding, retrieved_context)
        if self.answerable and not self.no_results:
            return AnswerabilityAssessment(
                original_query="q",
                question_type="other_query",
                answerability_expectation="general_grounded_response",
                has_relevant_context=True,
                sufficient_context=True,
                partially_supported=False,
                support_level="sufficient",
                insufficiency_reason=None,
                matched_parent_chunk_ids=["p1"],
                matched_headings=["h"],
                evidence_notes=[],
                warnings=[],
                should_answer=True,
            )
        return AnswerabilityAssessment(
            original_query="q",
            question_type="other_query",
            answerability_expectation="general_grounded_response",
            has_relevant_context=False,
            sufficient_context=False,
            partially_supported=False,
            support_level="none",
            insufficiency_reason="fact_not_found",
            matched_parent_chunk_ids=[],
            matched_headings=[],
            evidence_notes=[],
            warnings=["definition_not_supported"],
            should_answer=False,
        )

    def generate_answer(self, _: list[object], __: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=self.answer_text,
            grounded=True,
            sufficient_context=True,
            citations=[
                AnswerCitation(
                    parent_chunk_id="p1",
                    document_id="doc-1",
                    source_name="s",
                    heading=None,
                    supporting_excerpt="e",
                )
            ],
            warnings=[],
        )


def _hybrid(child_id: str, parent_id: str) -> HybridSearchResult:
    return HybridSearchResult(child_chunk_id=child_id, parent_chunk_id=parent_id, document_id="doc-1", text="t", hybrid_score=0.8)


def _reranked(child_id: str, parent_id: str) -> RerankedChunkResult:
    return RerankedChunkResult(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text="t",
        rerank_score=0.9,
        original_score=0.8,
    )


def _parent(parent_id: str) -> ParentChunkResult:
    return ParentChunkResult(parent_chunk_id=parent_id, document_id="doc-1", text="parent", source="s", source_name="s", heading_text="h")


def _span_by_stage(trace: dict[str, Any], stage: str) -> dict[str, Any]:
    return next(item for item in trace["spans"] if item["stage"] == stage)


def test_one_request_produces_one_linked_trace_and_stage_spans() -> None:
    services = _TracingServices()
    final_answer, state = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=services.deps())

    assert final_answer.answer_text == "Grounded answer"
    trace = state["trace"]
    assert trace is not None
    assert isinstance(trace["trace_id"], str) and trace["trace_id"]
    assert trace["request_id"]
    assert trace["active_family"] == "party_role_verification"
    stages = [span["stage"] for span in trace["spans"]]
    for required in (
        "query_understanding",
        "decomposition",
        "retrieval",
        "rerank",
        "parent_expansion",
        "answerability",
        "answer_generation",
        "final_synthesis",
    ):
        assert required in stages

    rerank = _span_by_stage(trace, "rerank")
    parent_expansion = _span_by_stage(trace, "parent_expansion")
    retrieval = _span_by_stage(trace, "retrieval")
    assert rerank["parent_span_id"] == retrieval["span_id"]
    assert parent_expansion["parent_span_id"] == retrieval["span_id"]

    for span in trace["spans"]:
        assert span["start_time_utc"] is not None
        assert span["end_time_utc"] is not None
        assert span["duration_ms"] is not None

    answerability = _span_by_stage(trace, "answerability")
    assert "sufficient_context" in answerability["outputs_summary"]
    assert "support_level" in answerability["outputs_summary"]
    assert "insufficiency_reason" in answerability["outputs_summary"]

    answer_generation = _span_by_stage(trace, "answer_generation")
    assert "path_taken" in answer_generation["outputs_summary"]
    assert "citation_count" in answer_generation["outputs_summary"]

    final_synthesis = _span_by_stage(trace, "final_synthesis")
    assert "final_output_status" in final_synthesis["outputs_summary"]
    assert "grounded" in final_synthesis["outputs_summary"]
    assert "sufficient_context" in final_synthesis["outputs_summary"]
    assert "synthesis_path" in final_synthesis["outputs_summary"]


def test_skipped_stages_are_stable_and_outputs_unchanged() -> None:
    services = _TracingServices(no_results=True, answerable=False, answer_text="unused")
    answer_only = run_legal_rag_turn(query="Who is the employer?", dependencies=services.deps())
    answer_with_state, state = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=services.deps())

    assert answer_only.model_dump() == answer_with_state.model_dump()
    trace = state["trace"]
    assert trace is not None
    assert len({trace["trace_id"]}) == 1

    rerank = _span_by_stage(trace, "rerank")
    assert rerank["status"] == "skipped"
    answer_generation = _span_by_stage(trace, "answer_generation")
    assert answer_generation["status"] == "skipped"
    assert _span_by_stage(trace, "retrieval")["outputs_summary"]["retrieved_child_count"] == 0
