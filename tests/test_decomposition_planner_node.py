from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_rag.orchestration.retrieval_graph import (
    DecompositionPlan,
    QueryRoutingDecision,
    RetrievalDependencies,
    RetrievalGraphConfig,
    RetrievalGraphNodes,
    SubQueryPlan,
    default_retrieval_state,
)
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.context_processing import CompressContextResult
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult


@dataclass
class _PlannerOnlyDeps:
    query_decision: QueryRoutingDecision

    def as_dependencies(self) -> RetrievalDependencies:
        return RetrievalDependencies(
            rewrite_query=self.rewrite_query,
            extract_legal_entities=self.extract_legal_entities,
            hybrid_search=self.hybrid_search,
            rerank_chunks=self.rerank_chunks,
            retrieve_parent_chunks=self.retrieve_parent_chunks,
            compress_context=self.compress_context,
            classify_query_state=self.classify,
        )

    def classify(self, *_: Any, **__: Any) -> QueryRoutingDecision:
        return self.query_decision

    def rewrite_query(self, query: str, **_: Any) -> QueryRewriteResult:
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=query,
            used_conversation_context=False,
            rewrite_notes="noop",
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
            filters=LegalEntityFilters(
                jurisdiction=[],
                court=[],
                document_type=[],
                date_from=None,
                date_to=None,
                clause_type=[],
            ),
            ambiguity_notes=[],
            warnings=[],
            extraction_notes=[],
        )

    def hybrid_search(self, *_: Any, **__: Any) -> list[HybridSearchResult]:
        return []

    def rerank_chunks(self, *_: Any, **__: Any) -> list[RerankedChunkResult]:
        return []

    def retrieve_parent_chunks(self, *_: Any, **__: Any) -> list[ParentChunkResult]:
        return []

    def compress_context(self, *_: Any, **__: Any) -> CompressContextResult:
        return CompressContextResult(items=tuple(), total_original_chars=0, total_compressed_chars=0)


def _decision() -> QueryRoutingDecision:
    return QueryRoutingDecision(
        original_query="q",
        normalized_query="q",
        question_type="comparison_query",
        is_followup=False,
        is_context_dependent=False,
        use_conversation_context=False,
        is_document_scoped=True,
        refers_to_prior_document_scope=False,
        refers_to_prior_clause_or_topic=False,
        should_rewrite=False,
        should_extract_entities=False,
        should_retrieve=True,
        may_need_decomposition=True,
        answerability_expectation="comparison",
        resolved_document_hints=[],
        resolved_topic_hints=["termination"],
        resolved_clause_hints=["termination"],
        ambiguity_notes=[],
        routing_notes=["test"],
        warnings=[],
    )


def _nodes() -> RetrievalGraphNodes:
    deps = _PlannerOnlyDeps(query_decision=_decision()).as_dependencies()
    return RetrievalGraphNodes(dependencies=deps, config=RetrievalGraphConfig())


def test_maybe_build_decomposition_plan_creates_typed_plan_for_decomposable_query() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="Compare the governing law and dispute resolution clauses.")
    state["query_classification"] = _decision()
    state["needs_decomposition"] = True
    state["decomposition_gate_reasons"] = ["comparison_query"]

    updated = nodes.maybe_build_decomposition_plan(state)

    assert isinstance(updated["decomposition_plan"], DecompositionPlan)
    assert updated["decomposition_plan"] is not None
    assert updated["decomposition_plan"].should_decompose is True
    assert updated["decomposition_plan"].strategy == "comparison"
    assert isinstance(updated["decomposition_plan"].subqueries[0], SubQueryPlan)


def test_maybe_build_decomposition_plan_keeps_plan_unset_for_simple_query() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="What is confidentiality?")
    state["warnings"] = ["carry-forward"]
    state["needs_decomposition"] = False
    state["decomposition_gate_reasons"] = ["simple_single_clause_lookup"]

    updated = nodes.maybe_build_decomposition_plan(state)

    assert updated["decomposition_plan"] is None
    assert updated["warnings"] == ["carry-forward"]


def test_maybe_build_decomposition_plan_output_is_structurally_typed_and_bounded() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="How did the amendment change confidentiality obligations?")
    state["query_classification"] = _decision()
    state["needs_decomposition"] = True
    state["decomposition_gate_reasons"] = ["amendment_vs_base"]

    updated = nodes.maybe_build_decomposition_plan(state)
    plan = updated["decomposition_plan"]

    assert isinstance(plan, DecompositionPlan)
    assert plan is not None
    assert plan.strategy == "amendment_vs_base"
    assert len(plan.subqueries) == 2
    assert [subquery.id for subquery in plan.subqueries] == ["sq-1", "sq-2"]


def test_maybe_build_decomposition_plan_creates_valid_plan_for_cross_clause_query() -> None:
    nodes = _nodes()
    query = "How do the indemnity obligations interact with the limitation-of-liability clause?"
    state = default_retrieval_state(query=query)
    state["query_classification"] = _decision()
    state["resolved_query"] = query
    state["needs_decomposition"] = True
    state["decomposition_gate_reasons"] = ["cross_clause_obligation_condition"]

    updated = nodes.maybe_build_decomposition_plan(state)
    plan = updated["decomposition_plan"]

    assert isinstance(plan, DecompositionPlan)
    assert plan is not None
    assert plan.should_decompose is True
    assert plan.strategy == "cross_clause"
    assert len(plan.subqueries) == 1
    assert isinstance(plan.subqueries[0], SubQueryPlan)


def test_maybe_build_decomposition_plan_preserves_scope_dates_and_negation_markers() -> None:
    nodes = _nodes()
    query = "In the MSA dated January 1, 2024, what obligations apply unless terminated for cause?"
    state = default_retrieval_state(query=query)
    state["query_classification"] = _decision()
    state["resolved_query"] = query
    state["needs_decomposition"] = True
    state["decomposition_gate_reasons"] = ["exception_chain"]

    updated = nodes.maybe_build_decomposition_plan(state)
    plan = updated["decomposition_plan"]

    assert plan is not None
    assert plan.root_question == query
    assert "January 1, 2024" in plan.root_question
    assert "unless" in plan.root_question.lower()
    assert "preserve_negation_and_exceptions" in plan.planner_notes


def test_maybe_build_decomposition_plan_is_inert_when_gate_is_false_even_with_reasons() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="What is the governing law clause?")
    state["needs_decomposition"] = False
    state["decomposition_gate_reasons"] = ["comparison_query"]

    updated = nodes.maybe_build_decomposition_plan(state)

    assert updated["decomposition_plan"] is None
    assert updated["effective_query"] == state["effective_query"]
