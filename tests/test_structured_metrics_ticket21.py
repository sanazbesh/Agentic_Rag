from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import run_legal_rag_turn, run_legal_rag_turn_with_state
from agentic_rag.orchestration.metrics import (
    RequestMetricsRecord,
    aggregate_metrics,
    aggregate_metrics_by_document_type,
    aggregate_metrics_by_family,
    emit_request_metrics,
)
from agentic_rag.orchestration.retrieval_graph import QueryRoutingDecision, RetrievalDependencies
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult
from agentic_rag.tools.answerability import AnswerabilityAssessment
from agentic_rag.tools.context_processing import CompressContextResult
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult
from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies


@dataclass
class _MetricsServices:
    answerable: bool = True
    grounded: bool = True
    include_citations: bool = True
    legal_family: str = "party_role_verification"
    document_type: str = "contract"

    hybrid_hits: list[HybridSearchResult] = field(default_factory=lambda: [_hybrid("c1", "p1")])
    reranked_hits: list[RerankedChunkResult] = field(default_factory=lambda: [_reranked("c1", "p1")])

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
            routing_notes=[f"legal_question_family:{self.legal_family}"],
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
        return self.hybrid_hits

    def rerank_chunks(self, _: list[HybridSearchResult], __: str) -> list[RerankedChunkResult]:
        return self.reranked_hits

    def retrieve_parent_chunks(self, _: list[str]) -> list[ParentChunkResult]:
        return [_parent("p1", self.document_type)]

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        return CompressContextResult(items=(), total_original_chars=0, total_compressed_chars=0)

    def assess_answerability(self, query: str, query_understanding: Any, retrieved_context: list[object]) -> AnswerabilityAssessment:
        _ = (query, query_understanding, retrieved_context)
        if self.answerable:
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
        citations = (
            [
                AnswerCitation(
                    parent_chunk_id="p1",
                    document_id="doc-1",
                    source_name="src",
                    heading=None,
                    supporting_excerpt="excerpt",
                )
            ]
            if self.include_citations
            else []
        )
        return GenerateAnswerResult(
            answer_text="answer",
            grounded=self.grounded,
            sufficient_context=self.answerable,
            citations=citations,
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


def _parent(parent_id: str, document_type: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text="parent",
        source="s",
        source_name="s",
        heading_text="h",
        metadata={"document_type": document_type},
    )


def test_per_request_metrics_emitted_and_structured() -> None:
    _, state = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=_MetricsServices().deps())

    metrics = state["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["grounded_answer"] == 1
    assert metrics["insufficient_answer"] == 0
    assert metrics["family_route_present"] == 1
    assert metrics["legal_family"] == "party_role_verification"
    assert metrics["document_type"] == "contract"
    assert isinstance(metrics["latency_ms"], int)
    assert metrics["cost_usd"] is None
    assert metrics["input_tokens"] is None
    assert metrics["output_tokens"] is None
    assert metrics["total_tokens"] is None


def test_insufficient_metric_and_false_confident_proxy_are_deterministic() -> None:
    # grounded answer + no citations should deterministically set proxy=1.
    services = _MetricsServices(answerable=True, grounded=True, include_citations=False)
    _, state = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=services.deps())
    assert state["metrics"]["false_confident_proxy"] == 1

    insufficient_services = _MetricsServices(answerable=False, grounded=False, include_citations=False)
    _, insufficient_state = run_legal_rag_turn_with_state(
        query="Who is the employer?",
        dependencies=insufficient_services.deps(),
    )
    assert insufficient_state["metrics"]["insufficient_answer"] == 1


def test_cost_and_token_metrics_when_available_and_when_missing() -> None:
    answer, state = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=_MetricsServices().deps())
    assert state["metrics"]["cost_usd"] is None
    assert state["metrics"]["total_tokens"] is None

    enriched = dict(state)
    enriched["cost_usd"] = 0.034
    enriched["token_usage"] = {"input_tokens": 100, "output_tokens": 40}
    record = emit_request_metrics(final_answer=answer, state=enriched)
    assert record.cost_usd == 0.034
    assert record.total_tokens == 140


def test_aggregation_by_family_and_document_type_and_latency_p95() -> None:
    records = [
        RequestMetricsRecord(
            request_id="r1",
            trace_id="t1",
            legal_family="party_role_verification",
            document_type="contract",
            grounded_answer=1,
            insufficient_answer=0,
            false_confident_proxy=0,
            family_route_present=1,
            latency_ms=100,
            cost_usd=0.01,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
        RequestMetricsRecord(
            request_id="r2",
            trace_id="t2",
            legal_family="party_role_verification",
            document_type="contract",
            grounded_answer=0,
            insufficient_answer=1,
            false_confident_proxy=1,
            family_route_present=1,
            latency_ms=250,
            cost_usd=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        ),
        RequestMetricsRecord(
            request_id="r3",
            trace_id="t3",
            legal_family="chronology",
            document_type=None,
            grounded_answer=1,
            insufficient_answer=0,
            false_confident_proxy=0,
            family_route_present=1,
            latency_ms=400,
            cost_usd=0.02,
            input_tokens=20,
            output_tokens=8,
            total_tokens=28,
        ),
    ]

    all_rollup = aggregate_metrics(records)
    assert all_rollup.request_count == 3
    assert all_rollup.p95_latency_ms == 400

    by_family = aggregate_metrics_by_family(records)
    assert by_family["party_role_verification"].request_count == 2
    assert by_family["chronology"].request_count == 1

    by_doc_type = aggregate_metrics_by_document_type(records)
    assert by_doc_type["contract"].request_count == 2
    assert by_doc_type["unknown"].request_count == 1


def test_instrumentation_does_not_change_core_final_answer_output() -> None:
    services = _MetricsServices()
    answer_only = run_legal_rag_turn(query="Who is the employer?", dependencies=services.deps())
    answer_with_state, _ = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=services.deps())
    assert answer_only.model_dump() == answer_with_state.model_dump()
