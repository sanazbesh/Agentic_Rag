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
class _Deps:
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
        resolved_topic_hints=[],
        resolved_clause_hints=[],
        ambiguity_notes=[],
        routing_notes=["test"],
        warnings=[],
    )


def _nodes() -> RetrievalGraphNodes:
    deps = _Deps(query_decision=_decision()).as_dependencies()
    return RetrievalGraphNodes(dependencies=deps, config=RetrievalGraphConfig())


def test_validate_decomposition_plan_valid_plan_passes_unchanged() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="Compare governing law and dispute resolution in the agreement.")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="Compare governing law and dispute resolution in the agreement.",
        strategy="comparison",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="Locate governing law clauses within the same agreement scope as the root question.",
                purpose="Collect governing law evidence.",
                required=True,
                expected_answer_type="cross_reference",
            ),
            SubQueryPlan(
                id="sq-2",
                question="Locate dispute resolution clauses within the same agreement scope as the root question.",
                purpose="Collect dispute resolution evidence.",
                required=True,
                expected_answer_type="cross_reference",
            ),
        ],
    )

    updated = nodes.validate_decomposition_plan(state)

    assert updated["decomposition_plan"] == state["decomposition_plan"]
    assert updated["decomposition_validation_errors"] == []


def test_validate_decomposition_plan_rejects_duplicate_subqueries() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="Compare termination and notice in the agreement.",
        strategy="comparison",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="Locate termination clauses in the agreement.",
                purpose="one",
                required=True,
                expected_answer_type="cross_reference",
            ),
            SubQueryPlan(
                id="sq-2",
                question="Locate termination clauses in the agreement.",
                purpose="two",
                required=True,
                expected_answer_type="cross_reference",
            ),
        ],
    )

    updated = nodes.validate_decomposition_plan(state)
    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == ["duplicate_subqueries"]


def test_validate_decomposition_plan_rejects_too_many_subqueries() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="Compare clause sets in the agreement.",
        strategy="comparison",
        subqueries=[
            SubQueryPlan(id=f"sq-{index}", question=f"Locate clause {index} in the agreement scope.", purpose="p", required=True, expected_answer_type="cross_reference")
            for index in range(1, 6)
        ],
    )

    updated = nodes.validate_decomposition_plan(state)
    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == ["too_many_subqueries:max_4"]


def test_validate_decomposition_plan_rejects_dropped_key_entity_or_scope() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="In the Acme Corp agreement, compare Delaware governing law and arbitration clauses.",
        strategy="comparison",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="Locate remedies that might apply.",
                purpose="p",
                required=True,
                expected_answer_type="cross_reference",
            )
        ],
    )

    updated = nodes.validate_decomposition_plan(state)
    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == [
        "dropped_key_entity_or_scope:entity=acme corp,scope=agreement,scope=clause,scope=governing law"
    ]


def test_validate_decomposition_plan_rejects_lost_negation_or_exception_logic() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="What indemnity obligations apply unless gross negligence is proven?",
        strategy="exception_chain",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="Locate indemnity obligations in the agreement text.",
                purpose="p",
                required=True,
                expected_answer_type="obligation",
            )
        ],
    )

    updated = nodes.validate_decomposition_plan(state)
    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == ["lost_negation_or_exception_logic"]


def test_validate_decomposition_plan_rejects_vague_or_overly_broad_subquery() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="What are confidentiality carve-outs?",
        strategy="exception_chain",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="General overview please.",
                purpose="p",
                required=True,
                expected_answer_type="exception",
            )
        ],
    )

    updated = nodes.validate_decomposition_plan(state)
    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == ["vague_or_overly_broad_subquery:sq-1"]


def test_validate_decomposition_plan_none_plan_is_safe_noop() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = None
    state["decomposition_validation_errors"] = ["stale_error"]

    updated = nodes.validate_decomposition_plan(state)

    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == []


def test_validate_decomposition_plan_invalid_plan_clears_plan_and_records_errors() -> None:
    nodes = _nodes()
    state = default_retrieval_state(query="q")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=True,
        root_question="In the Acme Corp agreement, what obligations apply except for fraud?",
        strategy="exception_chain",
        subqueries=[
            SubQueryPlan(
                id="sq-1",
                question="Locate obligations in the agreement.",
                purpose="p",
                required=True,
                expected_answer_type="obligation",
            ),
            SubQueryPlan(
                id="sq-2",
                question="Locate obligations in the agreement.",
                purpose="p",
                required=True,
                expected_answer_type="obligation",
            ),
        ],
    )

    updated = nodes.validate_decomposition_plan(state)

    assert updated["decomposition_plan"] is None
    assert updated["decomposition_validation_errors"] == [
        "duplicate_subqueries",
        "dropped_key_entity_or_scope:entity=acme corp",
        "lost_negation_or_exception_logic",
    ]
