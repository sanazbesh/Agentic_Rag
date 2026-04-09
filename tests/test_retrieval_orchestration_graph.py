from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_rag.orchestration.decomposition_gate import decide_decomposition_need
from agentic_rag.orchestration.retrieval_graph import (
    DecompositionPlan,
    QueryRoutingDecision,
    RetrievalDependencies,
    RetrievalGraphConfig,
    classify_decomposition_need,
    build_retrieval_graph,
    default_retrieval_state,
    run_retrieval_stage,
)
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
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

    fail_rewrite: bool = False
    fail_extract: bool = False
    fail_rerank: bool = False
    fail_compress: bool = False

    rewrite_calls: list[dict[str, Any]] = field(default_factory=list)
    extract_calls: list[str] = field(default_factory=list)
    hybrid_calls: list[dict[str, Any]] = field(default_factory=list)

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
        return self.classifier

    def rewrite_query(self, query: str, **kwargs: Any) -> QueryRewriteResult:
        self.rewrite_calls.append({"query": query, **kwargs})
        if self.fail_rewrite:
            raise RuntimeError("rewrite failed")
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=self.rewritten_query or query,
            used_conversation_context=bool(kwargs.get("conversation_summary") or kwargs.get("recent_messages")),
            rewrite_notes="test",
        )

    def extract_legal_entities(self, query: str) -> LegalEntityExtractionResult:
        self.extract_calls.append(query)
        if self.fail_extract:
            raise RuntimeError("extract failed")
        return LegalEntityExtractionResult(
            original_query=query,
            normalized_query=query,
            document_types=["contract"],
            legal_topics=[],
            jurisdictions=["New York"],
            courts=[],
            laws_or_regulations=[],
            legal_citations=[],
            clause_types=["termination clause"],
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
                jurisdiction=["New York"],
                court=[],
                document_type=["contract"],
                date_from=None,
                date_to=None,
                clause_type=["termination clause"],
            ),
            ambiguity_notes=[],
            warnings=[],
            extraction_notes=["deterministic"],
        )

    def hybrid_search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[HybridSearchResult]:
        self.hybrid_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return self.hybrid_results

    def rerank_chunks(self, chunks: list[HybridSearchResult], query: str) -> list[RerankedChunkResult]:
        if self.fail_rerank:
            raise RuntimeError("rerank failed")
        return self.reranked_results

    def retrieve_parent_chunks(self, parent_ids: list[str]) -> list[ParentChunkResult]:
        return self.parent_results

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        if self.fail_compress:
            raise RuntimeError("compress failed")
        return CompressContextResult(items=tuple(self.compressed_items), total_original_chars=10, total_compressed_chars=5)


def _hybrid(child_id: str, parent_id: str, text: str = "termination clause") -> HybridSearchResult:
    return HybridSearchResult(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text=text,
        hybrid_score=0.8,
    )


def _reranked(child_id: str, parent_id: str) -> RerankedChunkResult:
    return RerankedChunkResult(
        child_chunk_id=child_id,
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text="txt",
        rerank_score=0.9,
        original_score=0.8,
    )


def _parent(parent_id: str, text: str = "short legal text") -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id="doc-1",
        text=text,
        source="test",
        source_name="test",
    )


def _decision(*, followup: bool, ambiguous: bool, use_context: bool, rewrite: bool, extract: bool) -> QueryRoutingDecision:
    return QueryRoutingDecision(
        original_query="q",
        normalized_query="q",
        question_type="ambiguous_query" if ambiguous else "other_query",
        is_followup=followup,
        is_context_dependent=followup or ambiguous,
        use_conversation_context=use_context,
        is_document_scoped=False,
        should_rewrite=rewrite,
        should_extract_entities=extract,
        should_retrieve=True,
        may_need_decomposition=False,
        resolved_document_hints=[],
        resolved_topic_hints=[],
        resolved_clause_hints=[],
        answerability_expectation="general_grounded_response",
        refers_to_prior_document_scope=followup,
        refers_to_prior_clause_or_topic=followup,
        ambiguity_notes=[],
        routing_notes=["test"],
        warnings=[],
    )


def test_self_contained_query_path_completes() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    state = run_retrieval_stage(query="Interpret Section 10 under Delaware law", dependencies=services.as_dependencies())
    assert state["effective_query"] == state["original_query"]
    assert state["use_conversation_context"] is False
    assert state["retrieval_stage_complete"] is True


def test_followup_query_uses_context_and_passes_to_rewrite() -> None:
    services = FakeServices(
        classifier=_decision(followup=True, ambiguous=True, use_context=True, rewrite=True, extract=False),
        rewritten_query="How does the termination clause exception apply?",
    )
    run_retrieval_stage(
        query="what about that clause?",
        conversation_summary="Prior turn discussed termination exceptions.",
        recent_messages=[{"role": "assistant", "content": "..."}],
        dependencies=services.as_dependencies(),
    )
    assert services.rewrite_calls
    assert "conversation_summary" in services.rewrite_calls[0]
    assert "recent_messages" in services.rewrite_calls[0]


def test_rewrite_needed_path_updates_effective_query() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=True, use_context=False, rewrite=True, extract=False),
        rewritten_query="Interpret notice obligation in Ontario employment agreement",
    )
    state = run_retrieval_stage(query="notice?", dependencies=services.as_dependencies())
    assert state["effective_query"] == "Interpret notice obligation in Ontario employment agreement"


def test_no_rewrite_path_skips_rewriter() -> None:
    services = FakeServices(classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False))
    state = run_retrieval_stage(query="Interpret GDPR breach notification duties", dependencies=services.as_dependencies())
    assert services.rewrite_calls == []
    assert state["effective_query"] == state["original_query"]


def test_entity_extraction_derives_filters_and_passes_to_hybrid_search() -> None:
    services = FakeServices(classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=True))
    state = run_retrieval_stage(query="New York termination clause in contract", dependencies=services.as_dependencies())
    assert services.extract_calls == [state["effective_query"]]
    assert services.hybrid_calls[0]["filters"] is not None
    assert services.hybrid_calls[0]["filters"]["jurisdiction"] == "New York"


def test_hybrid_rerank_parent_fetch_path() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[_hybrid("c1", "p1"), _hybrid("c2", "p2")],
        reranked_results=[_reranked("c2", "p2"), _reranked("c1", "p1")],
        parent_results=[_parent("p2"), _parent("p1")],
    )
    state = run_retrieval_stage(query="termination clause", dependencies=services.as_dependencies())
    assert state["parent_ids"] == ["p2", "p1"]
    assert [p.parent_chunk_id for p in state["parent_chunks"]] == ["p2", "p1"]


def test_compression_needed_path() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="word " * 200)],
        compressed_items=[
            CompressedParentChunk(
                parent_chunk_id="p1",
                document_id="doc-1",
                source="test",
                source_name="test",
                compressed_text="short",
                original_char_count=100,
                compressed_char_count=20,
            )
        ],
    )
    cfg = RetrievalGraphConfig(compress_if_total_parent_tokens_gte=50)
    state = run_retrieval_stage(query="termination", dependencies=services.as_dependencies(), config=cfg)
    assert state["should_compress"] is True
    assert len(state["compressed_context"]) == 1


def test_no_compression_path() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="small")],
    )
    cfg = RetrievalGraphConfig(compress_if_total_parent_tokens_gte=5000, compress_if_parent_chunks_gte=99)
    state = run_retrieval_stage(query="termination", dependencies=services.as_dependencies(), config=cfg)
    assert state["should_compress"] is False
    assert state["compressed_context"] == []


def test_failure_fallbacks_preserve_completion_and_warnings() -> None:
    services = FakeServices(
        classifier=_decision(followup=True, ambiguous=True, use_context=False, rewrite=True, extract=True),
        fail_rewrite=True,
        fail_extract=True,
        fail_rerank=True,
        fail_compress=True,
        hybrid_results=[_hybrid("c1", "p1")],
        parent_results=[_parent("p1", text="word " * 200)],
    )
    cfg = RetrievalGraphConfig(compress_if_total_parent_tokens_gte=50)
    state = run_retrieval_stage(query="what about that", dependencies=services.as_dependencies(), config=cfg)
    assert state["effective_query"] == state["original_query"]
    assert state["retrieval_stage_complete"] is True
    assert any("rewrite_failed" in warning for warning in state["warnings"])
    assert any("entity_extraction_failed" in warning for warning in state["warnings"])
    assert any("rerank_failed" in warning for warning in state["warnings"])
    assert any("compression_failed" in warning for warning in state["warnings"])


def test_no_results_behavior_completes_cleanly() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[],
        reranked_results=[],
        parent_results=[],
    )
    state = run_retrieval_stage(query="res ipsa loquitur", dependencies=services.as_dependencies())
    assert state["child_results"] == []
    assert state["parent_ids"] == []
    assert state["parent_chunks"] == []
    assert state["compressed_context"] == []
    assert state["retrieval_stage_complete"] is True


def test_deterministic_routing_for_same_inputs() -> None:
    services = FakeServices(classifier=_decision(followup=True, ambiguous=True, use_context=True, rewrite=True, extract=True))
    state_a = run_retrieval_stage(
        query="what about that clause?",
        conversation_summary="summary",
        recent_messages=[{"role": "user", "content": "ctx"}],
        dependencies=services.as_dependencies(),
    )
    state_b = run_retrieval_stage(
        query="what about that clause?",
        conversation_summary="summary",
        recent_messages=[{"role": "user", "content": "ctx"}],
        dependencies=services.as_dependencies(),
    )
    assert state_a["query_classification"] == state_b["query_classification"]
    assert state_a["should_rewrite"] == state_b["should_rewrite"]
    assert state_a["should_extract_entities"] == state_b["should_extract_entities"]


def test_default_retrieval_state_has_safe_decomposition_wiring_defaults() -> None:
    state = default_retrieval_state(query="What is confidentiality?")

    assert state["decomposition_plan"] is None
    assert state["decomposition_validation_errors"] == []
    assert state["decomposition_validation_errors"] is not default_retrieval_state(query="q")[
        "decomposition_validation_errors"
    ]


def test_fallback_graph_carries_decomposition_plan_without_using_it(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    state = default_retrieval_state(query="What is governing law?")
    state["decomposition_plan"] = DecompositionPlan(
        should_decompose=False,
        root_question="What is governing law?",
    )

    result = app.invoke(state)

    assert result["decomposition_plan"] == state["decomposition_plan"]
    assert services.hybrid_calls[0]["query"] == result["effective_query"]


def test_fallback_graph_normalizes_none_decomposition_validation_errors(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    state = default_retrieval_state(query="Define effective date.")
    state["decomposition_plan"] = None
    state["decomposition_validation_errors"] = None  # type: ignore[assignment]

    result = app.invoke(state)

    assert result["decomposition_plan"] is None
    assert result["decomposition_validation_errors"] == []
    assert result["retrieval_stage_complete"] is True




def test_classify_decomposition_need_runs_after_context_resolution(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    call_order: list[str] = []
    original_resolve = retrieval_graph_module.RetrievalGraphNodes.resolve_query_context
    original_gate = retrieval_graph_module.RetrievalGraphNodes.classify_decomposition_need

    def wrapped_resolve(self, state):
        call_order.append("resolve_query_context")
        return original_resolve(self, state)

    def wrapped_gate(self, state):
        call_order.append("classify_decomposition_need")
        return original_gate(self, state)

    monkeypatch.setattr(retrieval_graph_module.RetrievalGraphNodes, "resolve_query_context", wrapped_resolve)
    monkeypatch.setattr(retrieval_graph_module.RetrievalGraphNodes, "classify_decomposition_need", wrapped_gate)

    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    run_retrieval_stage(query="What is confidentiality?", dependencies=services.as_dependencies())

    assert "resolve_query_context" in call_order
    assert "classify_decomposition_need" in call_order
    assert call_order.index("resolve_query_context") < call_order.index("classify_decomposition_need")


def test_invalid_helper_result_defaults_to_safe_gate_output(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    class InvalidDecision:  # pragma: no cover - local test double
        needs_decomposition = "yes"
        reasons = ["comparison_query"]

    monkeypatch.setattr(retrieval_graph_module, "decide_decomposition_need", lambda **_: InvalidDecision())

    needs_decomposition, reasons = retrieval_graph_module.classify_decomposition_need(
        query="Compare governing law versus dispute resolution.",
        query_classification=None,
        context_resolution=None,
    )

    assert needs_decomposition is False
    assert reasons == []


def test_decomposition_true_does_not_change_downstream_search_inputs() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    state = run_retrieval_stage(
        query="Compare governing law versus dispute resolution.",
        dependencies=services.as_dependencies(),
    )

    assert state["needs_decomposition"] is True
    assert services.hybrid_calls[0]["query"] == state["effective_query"]
    assert state["effective_query"] == state["original_query"]


def test_decomposition_gate_simple_definition_does_not_trigger() -> None:
    decision = _decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False)
    needs_decomposition, reasons = classify_decomposition_need(
        query="What is confidentiality?",
        query_classification=decision,
        context_resolution=None,
    )
    assert needs_decomposition is False
    assert reasons == ["simple_single_clause_lookup"]


def test_decomposition_gate_conjunctive_query_triggers() -> None:
    decision = _decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False)
    needs_decomposition, reasons = classify_decomposition_need(
        query="Who are the parties involved in this agreement and what are their obligations?",
        query_classification=decision,
        context_resolution=None,
    )
    helper_decision = decide_decomposition_need(
        query="Who are the parties involved in this agreement and what are their obligations?",
        query_understanding=decision,
        query_context=None,
    )
    assert needs_decomposition is False
    assert reasons == helper_decision.reasons


def test_decomposition_gate_comparison_query_triggers() -> None:
    decision = _decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False)
    needs_decomposition, reasons = classify_decomposition_need(
        query="Compare the governing law and dispute resolution clauses.",
        query_classification=decision,
        context_resolution=None,
    )
    helper_decision = decide_decomposition_need(
        query="Compare the governing law and dispute resolution clauses.",
        query_understanding=decision,
        query_context=None,
    )
    assert needs_decomposition is True
    assert reasons == helper_decision.reasons


def test_decomposition_gate_amendment_temporal_query_triggers() -> None:
    decision = _decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False)
    needs_decomposition, reasons = classify_decomposition_need(
        query="How did the amendment change the confidentiality obligations?",
        query_classification=decision,
        context_resolution=None,
    )
    helper_decision = decide_decomposition_need(
        query="How did the amendment change the confidentiality obligations?",
        query_understanding=decision,
        query_context=None,
    )
    assert needs_decomposition is True
    assert reasons == helper_decision.reasons


def test_decomposition_gate_context_dependent_followup_can_trigger() -> None:
    decision = _decision(followup=True, ambiguous=True, use_context=True, rewrite=True, extract=False)
    state = run_retrieval_stage(
        query="Compare governing law versus dispute resolution in that clause?",
        conversation_summary="Prior turn discussed termination section.",
        recent_messages=[
            {
                "role": "assistant",
                "content": "Governing law is Delaware.",
                "metadata": {"resolved_topic_hints": ["termination"]},
            }
        ],
        dependencies=FakeServices(classifier=decision).as_dependencies(),
    )
    assert state["needs_decomposition"] is True
    helper_decision = decide_decomposition_need(
        query=state["resolved_query"],
        query_context=state["context_resolution"].model_dump() if state["context_resolution"] is not None else None,
        query_understanding=state["query_classification"],
    )
    assert state["decomposition_gate_reasons"] == helper_decision.reasons


def test_decomposition_gate_followup_without_strong_structure_stays_false() -> None:
    decision = _decision(followup=True, ambiguous=True, use_context=True, rewrite=True, extract=False)
    state = run_retrieval_stage(
        query="What about that clause?",
        conversation_summary="Prior turn discussed termination section.",
        recent_messages=[
            {
                "role": "assistant",
                "content": "Termination notice was 30 days.",
                "metadata": {"resolved_clause_hints": ["termination"]},
            }
        ],
        dependencies=FakeServices(classifier=decision).as_dependencies(),
    )
    assert state["needs_decomposition"] is False
    assert state["decomposition_gate_reasons"] == ["simple_single_clause_lookup"]


def test_retrieval_state_contains_decomposition_gate_output() -> None:
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    state = run_retrieval_stage(
        query="What is the governing law?",
        dependencies=services.as_dependencies(),
    )
    assert state["needs_decomposition"] is False
    assert state["decomposition_gate_reasons"] == ["simple_single_clause_lookup"]


def test_fallback_graph_populates_decomposition_gate_output(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    state = app.invoke(default_retrieval_state(query="Define effective date."))
    assert state["needs_decomposition"] is False
    assert state["decomposition_gate_reasons"] == ["simple_single_clause_lookup"]


def test_fallback_graph_records_decomposition_trigger_for_comparison_query(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    state = app.invoke(default_retrieval_state(query="Compare governing law and dispute resolution clauses."))

    assert state["needs_decomposition"] is True
    assert "comparison_query" in state["decomposition_gate_reasons"]


def test_fallback_graph_gate_output_matches_shared_helper(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    query = "Compare governing law and dispute resolution clauses."
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    state = app.invoke(default_retrieval_state(query=query))

    helper_decision = decide_decomposition_need(
        query=state["resolved_query"],
        query_context=state["context_resolution"].model_dump() if state["context_resolution"] is not None else None,
        query_understanding=state["query_classification"],
    )
    assert state["needs_decomposition"] == helper_decision.needs_decomposition
    assert state["decomposition_gate_reasons"] == helper_decision.reasons


def test_fallback_graph_continues_existing_retrieval_flow_after_gate(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1")],
    )
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    state = app.invoke(default_retrieval_state(query="Compare governing law and dispute resolution clauses."))

    assert state["needs_decomposition"] is True
    assert services.hybrid_calls[0]["query"] == state["effective_query"]
    assert state["parent_ids"] == ["p1"]
    assert state["retrieval_stage_complete"] is True


def test_fallback_graph_gate_outputs_are_deterministic_for_identical_inputs(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    services = FakeServices(
        classifier=_decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False),
    )
    app = build_retrieval_graph(dependencies=services.as_dependencies())
    first = app.invoke(default_retrieval_state(query="Compare governing law and dispute resolution clauses."))
    second = app.invoke(default_retrieval_state(query="Compare governing law and dispute resolution clauses."))

    assert first["needs_decomposition"] == second["needs_decomposition"]
    assert first["decomposition_gate_reasons"] == second["decomposition_gate_reasons"]


def test_main_graph_and_fallback_gate_outputs_match_for_equivalent_resolved_inputs(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph_module

    query = "Compare the governing law and dispute resolution clauses."
    classifier = _decision(followup=False, ambiguous=False, use_context=False, rewrite=False, extract=False)

    main_services = FakeServices(classifier=classifier)
    main_state = run_retrieval_stage(query=query, dependencies=main_services.as_dependencies())

    monkeypatch.setattr(retrieval_graph_module, "StateGraph", None)
    fallback_services = FakeServices(classifier=classifier)
    fallback_app = build_retrieval_graph(dependencies=fallback_services.as_dependencies())
    fallback_state = fallback_app.invoke(default_retrieval_state(query=query))

    assert main_state["resolved_query"] == fallback_state["resolved_query"]
    assert main_state["context_resolution"] == fallback_state["context_resolution"]
    assert main_state["query_classification"] == fallback_state["query_classification"]
    assert main_state["needs_decomposition"] == fallback_state["needs_decomposition"]
    assert main_state["decomposition_gate_reasons"] == fallback_state["decomposition_gate_reasons"]
