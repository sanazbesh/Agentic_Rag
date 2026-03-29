"""Deterministic retrieval-stage orchestration graph for legal RAG.

This module intentionally implements retrieval as an explicit graph (node + edge
transitions) rather than a free-form agent loop so production behavior remains
traceable, testable, and bounded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, cast

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.context_processing import CompressContextResult, CompressedParentChunk
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, QueryRewriteResult

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when langgraph is installed
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - deterministic fallback used in tests
    START = "__start__"
    END = "__end__"
    StateGraph = None


class QueryRoutingDecision(BaseModel):
    """Strict routing output for deterministic query-state classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_followup: bool
    is_ambiguous: bool
    is_context_dependent: bool
    use_conversation_context: bool
    should_rewrite: bool
    should_extract_entities: bool
    refers_to_prior_document_scope: bool
    refers_to_prior_clause_or_topic: bool
    routing_notes: list[str] = Field(default_factory=list)


class QueryContextResolution(BaseModel):
    """Typed, inspectable output for conservative follow-up reference resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolved_query: str
    used_conversation_context: bool
    resolved_document_ids: list[str] = Field(default_factory=list)
    resolved_topic_hints: list[str] = Field(default_factory=list)
    resolution_notes: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)


class RetrievalStageState(TypedDict):
    """Strict retrieval-stage state shared by all graph nodes."""

    original_query: str
    conversation_summary: str | None
    recent_messages: list[Mapping[str, Any]]
    use_conversation_context: bool

    rewritten_query: str | None
    resolved_query: str
    effective_query: str
    query_classification: QueryRoutingDecision | None
    context_resolution: QueryContextResolution | None

    extracted_entities: LegalEntityExtractionResult | None
    filters: dict[str, Any] | None

    child_results: list[HybridSearchResult]
    reranked_child_results: list[RerankedChunkResult]
    parent_ids: list[str]
    parent_chunks: list[ParentChunkResult]
    compressed_context: list[CompressedParentChunk]

    should_rewrite: bool
    should_extract_entities: bool
    should_compress: bool

    warnings: list[str]
    retrieval_stage_complete: bool
    last_resolved_document_scope: list[str]
    last_resolved_topic: str | None
    prior_effective_query: str | None
    prior_final_answer: str | None
    prior_citations: list[Mapping[str, Any]]


class QueryClassifier(Protocol):
    """Classification dependency returning strict structured routing decisions."""

    def __call__(
        self,
        query: str,
        *,
        conversation_summary: str | None,
        recent_messages: Sequence[Mapping[str, Any]],
    ) -> QueryRoutingDecision: ...


@dataclass(slots=True, frozen=True)
class RetrievalGraphConfig:
    """Deterministic, configurable thresholds and limits for retrieval flow."""

    hybrid_top_k: int = 12
    compress_if_parent_chunks_gte: int = 6
    compress_if_total_parent_tokens_gte: int = 2400
    token_estimator: Callable[[str], int] = lambda text: len((text or "").split())


@dataclass(slots=True)
class RetrievalDependencies:
    """Callables backing deterministic retrieval-stage graph nodes."""

    rewrite_query: Callable[..., QueryRewriteResult]
    extract_legal_entities: Callable[[str], LegalEntityExtractionResult]
    hybrid_search: Callable[..., list[HybridSearchResult]]
    rerank_chunks: Callable[[Sequence[HybridSearchResult], str], list[RerankedChunkResult]]
    retrieve_parent_chunks: Callable[[Sequence[str]], list[ParentChunkResult]]
    compress_context: Callable[[Sequence[ParentChunkResult]], CompressContextResult]
    classify_query_state: QueryClassifier = field(default_factory=lambda: heuristic_query_classifier)


def default_retrieval_state(
    *,
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
) -> RetrievalStageState:
    """Build strict initial state with explicit defaults for all list fields."""

    normalized_query = (query or "").strip()
    return RetrievalStageState(
        original_query=normalized_query,
        conversation_summary=conversation_summary,
        recent_messages=list(recent_messages or []),
        use_conversation_context=False,
        rewritten_query=None,
        resolved_query=normalized_query,
        effective_query=normalized_query,
        query_classification=None,
        context_resolution=None,
        extracted_entities=None,
        filters=None,
        child_results=[],
        reranked_child_results=[],
        parent_ids=[],
        parent_chunks=[],
        compressed_context=[],
        should_rewrite=False,
        should_extract_entities=False,
        should_compress=False,
        warnings=[],
        retrieval_stage_complete=False,
        last_resolved_document_scope=[],
        last_resolved_topic=None,
        prior_effective_query=None,
        prior_final_answer=None,
        prior_citations=[],
    )


def heuristic_query_classifier(
    query: str,
    *,
    conversation_summary: str | None,
    recent_messages: Sequence[Mapping[str, Any]],
) -> QueryRoutingDecision:
    """Conservative deterministic classifier used when no LLM classifier is injected."""

    lowered = (query or "").strip().lower()
    followup_starters = (
        "what about",
        "how about",
        "does that",
        "is that",
        "compare that",
        "what does it say",
    )
    followup_markers = {"that", "those", "it", "they", "this", "these", "above", "previous"}
    legal_filter_markers = {
        "court",
        "statute",
        "regulation",
        "clause",
        "law",
        "jurisdiction",
        "ontario",
        "california",
        "delaware",
        "new york",
        "federal",
        "section",
        "article",
        "agreement",
        "contract",
    }
    ambiguous_markers = {"exception", "that", "this", "it", "still apply", "what about"}
    pronoun_markers = {
        "it",
        "this",
        "that",
        "the document",
        "this document",
        "that document",
        "the clause",
        "that clause",
        "this agreement",
        "that agreement",
    }
    explicit_scope_markers = {
        " in the nda",
        " in the msa",
        " in the lease",
        " in the agreement",
        " in the contract",
        " in the statute",
    }

    token_set = set(lowered.split())
    has_recent_context = bool((conversation_summary and conversation_summary.strip()) or recent_messages)
    is_followup = lowered.startswith(followup_starters) or bool(token_set.intersection(followup_markers))
    is_context_dependent = any(marker in lowered for marker in pronoun_markers) or lowered.startswith(
        ("what about", "does it", "compare that")
    )
    has_explicit_new_scope = any(marker in f" {lowered}" for marker in explicit_scope_markers) and "that agreement" not in lowered
    is_ambiguous = any(marker in lowered for marker in ambiguous_markers) and len(token_set) <= 10
    use_context = has_recent_context and (is_followup or is_ambiguous or is_context_dependent) and not has_explicit_new_scope
    refers_to_prior_document_scope = is_context_dependent and any(
        marker in lowered
        for marker in ("it", "document", "that agreement", "this agreement")
    )
    refers_to_prior_clause_or_topic = is_context_dependent and any(
        marker in lowered for marker in ("clause", "what about", "compare that", "still apply")
    )

    should_rewrite = is_followup or is_ambiguous or is_context_dependent or len(token_set) <= 3
    should_extract = any(marker in lowered for marker in legal_filter_markers)

    notes: list[str] = []
    if is_followup:
        notes.append("followup_like_query")
    if is_ambiguous:
        notes.append("ambiguous_or_elliptical")
    if is_context_dependent:
        notes.append("context_dependent")
    if use_context:
        notes.append("use_conversation_context")
    if should_rewrite:
        notes.append("rewrite_recommended")
    if should_extract:
        notes.append("entity_extraction_useful")

    return QueryRoutingDecision(
        is_followup=is_followup,
        is_ambiguous=is_ambiguous,
        is_context_dependent=is_context_dependent,
        use_conversation_context=use_context,
        should_rewrite=should_rewrite,
        should_extract_entities=should_extract,
        refers_to_prior_document_scope=refers_to_prior_document_scope,
        refers_to_prior_clause_or_topic=refers_to_prior_clause_or_topic,
        routing_notes=notes,
    )


def _derive_filters(extraction: LegalEntityExtractionResult) -> dict[str, Any] | None:
    filters = extraction.filters
    payload: dict[str, Any] = {}
    if filters.jurisdiction:
        payload["jurisdiction"] = filters.jurisdiction[0]
    if filters.court:
        payload["court"] = filters.court[0]
    if filters.document_type:
        payload["document_type"] = filters.document_type[0]
    if filters.clause_type:
        payload["clause_type"] = filters.clause_type[0]
    if filters.date_from:
        payload["date_from"] = filters.date_from
    if filters.date_to:
        payload["date_to"] = filters.date_to
    return payload or None


class RetrievalGraphNodes:
    """Explicit node implementations for a deterministic retrieval workflow graph."""

    def __init__(self, dependencies: RetrievalDependencies, config: RetrievalGraphConfig) -> None:
        self.dependencies = dependencies
        self.config = config

    def ingest_turn(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=ingest_turn")
        updated = dict(state)
        updated["warnings"] = list(updated.get("warnings", []))
        updated["child_results"] = list(updated.get("child_results", []))
        updated["reranked_child_results"] = list(updated.get("reranked_child_results", []))
        updated["parent_ids"] = list(updated.get("parent_ids", []))
        updated["parent_chunks"] = list(updated.get("parent_chunks", []))
        updated["compressed_context"] = list(updated.get("compressed_context", []))
        self._hydrate_prior_turn_memory(updated)
        logger.info("node_exit name=ingest_turn query_length=%s", len(updated["original_query"]))
        return cast(RetrievalStageState, updated)

    def _hydrate_prior_turn_memory(self, state: dict[str, Any]) -> None:
        messages = list(state.get("recent_messages", []))
        last_assistant = next(
            (msg for msg in reversed(messages) if str(msg.get("role", "")).lower() == "assistant"),
            None,
        )
        if not isinstance(last_assistant, Mapping):
            return
        metadata = last_assistant.get("metadata")
        if not isinstance(metadata, Mapping):
            return
        state["last_resolved_document_scope"] = [str(item) for item in metadata.get("resolved_document_ids", []) if item]
        topic_hints = [str(item) for item in metadata.get("resolved_topic_hints", []) if item]
        state["last_resolved_topic"] = topic_hints[0] if topic_hints else None
        prior_effective_query = metadata.get("effective_query")
        if isinstance(prior_effective_query, str) and prior_effective_query.strip():
            state["prior_effective_query"] = prior_effective_query
        prior_final_answer = metadata.get("answer_text")
        if isinstance(prior_final_answer, str) and prior_final_answer.strip():
            state["prior_final_answer"] = prior_final_answer
        citations = metadata.get("citations")
        if isinstance(citations, list):
            state["prior_citations"] = [item for item in citations if isinstance(item, Mapping)]

    def classify_query_state(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=classify_query_state")
        updated = dict(state)
        decision = self.dependencies.classify_query_state(
            updated["original_query"],
            conversation_summary=updated["conversation_summary"],
            recent_messages=updated["recent_messages"],
        )
        updated["query_classification"] = decision
        updated["use_conversation_context"] = decision.use_conversation_context
        updated["should_rewrite"] = decision.should_rewrite
        updated["should_extract_entities"] = decision.should_extract_entities
        logger.info(
            "node_exit name=classify_query_state followup=%s ambiguous=%s rewrite=%s extract=%s",
            decision.is_followup,
            decision.is_ambiguous,
            decision.should_rewrite,
            decision.should_extract_entities,
        )
        return cast(RetrievalStageState, updated)

    def resolve_query_context(self, state: RetrievalStageState) -> RetrievalStageState:
        """Resolve conversational references before retrieval in a typed, traceable way."""

        logger.info("node_enter name=resolve_query_context")
        updated = dict(state)
        query = str(updated["original_query"]).strip()
        decision = updated.get("query_classification")
        if not isinstance(decision, QueryRoutingDecision):
            resolution = QueryContextResolution(resolved_query=query, used_conversation_context=False)
            updated["context_resolution"] = resolution
            updated["resolved_query"] = query
            return cast(RetrievalStageState, updated)

        candidate_doc_ids: list[str] = list(updated.get("last_resolved_document_scope", []))
        for citation in updated.get("prior_citations", []):
            document_id = citation.get("document_id")
            if isinstance(document_id, str) and document_id and document_id not in candidate_doc_ids:
                candidate_doc_ids.append(document_id)

        topic_hints: list[str] = []
        if updated.get("last_resolved_topic"):
            topic_hints.append(str(updated["last_resolved_topic"]))
        used_context = False
        unresolved_references: list[str] = []
        resolution_notes = list(decision.routing_notes)
        resolved_query = query

        if decision.use_conversation_context and decision.is_context_dependent:
            if decision.refers_to_prior_document_scope and len(candidate_doc_ids) > 1:
                unresolved_references.append("document_reference:ambiguous_multiple_candidates")
                resolution_notes.append("ambiguous_document_reference")
            elif decision.refers_to_prior_document_scope and len(candidate_doc_ids) == 1:
                used_context = True
                resolution_notes.append("resolved_document_scope_from_prior_turn")

            if decision.refers_to_prior_clause_or_topic and topic_hints:
                used_context = True
                resolution_notes.append("resolved_clause_or_topic_from_prior_turn")

            if used_context and topic_hints and query.lower().startswith("what about"):
                resolved_query = f"{query} in relation to {topic_hints[0]}"

        resolution = QueryContextResolution(
            resolved_query=resolved_query,
            used_conversation_context=used_context,
            resolved_document_ids=candidate_doc_ids if used_context else [],
            resolved_topic_hints=topic_hints if used_context else [],
            resolution_notes=resolution_notes,
            unresolved_references=unresolved_references,
        )
        updated["context_resolution"] = resolution
        updated["resolved_query"] = resolution.resolved_query
        if resolution.resolved_document_ids:
            existing_filters = dict(updated.get("filters") or {})
            existing_filters["resolved_document_ids"] = list(resolution.resolved_document_ids)
            updated["filters"] = existing_filters
        if resolution.unresolved_references:
            updated["warnings"] = [*updated["warnings"], *resolution.unresolved_references]
        logger.info(
            "node_exit name=resolve_query_context used_context=%s unresolved_count=%s",
            resolution.used_conversation_context,
            len(resolution.unresolved_references),
        )
        return cast(RetrievalStageState, updated)

    def rewrite_query_if_needed(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=rewrite_query_if_needed should_rewrite=%s", state["should_rewrite"])
        updated = dict(state)
        original_query = str(updated.get("resolved_query") or updated["original_query"])
        updated["effective_query"] = original_query
        if not updated["should_rewrite"]:
            logger.info("node_exit name=rewrite_query_if_needed rewritten=false")
            return cast(RetrievalStageState, updated)
        try:
            kwargs: dict[str, Any] = {}
            if updated["use_conversation_context"]:
                kwargs["conversation_summary"] = updated["conversation_summary"]
                kwargs["recent_messages"] = updated["recent_messages"]
            result = self.dependencies.rewrite_query(original_query, **kwargs)
            updated["rewritten_query"] = result.rewritten_query
            updated["effective_query"] = result.rewritten_query or original_query
        except Exception as exc:  # pragma: no cover - defensive fallback
            updated["warnings"] = [*updated["warnings"], f"rewrite_failed:{type(exc).__name__}"]
            updated["rewritten_query"] = None
            updated["effective_query"] = original_query
        logger.info(
            "node_exit name=rewrite_query_if_needed rewritten=%s effective_query_length=%s",
            bool(updated["rewritten_query"]),
            len(updated["effective_query"]),
        )
        return cast(RetrievalStageState, updated)

    def extract_entities_if_needed(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=extract_entities_if_needed should_extract=%s", state["should_extract_entities"])
        updated = dict(state)
        if not updated["should_extract_entities"]:
            logger.info("node_exit name=extract_entities_if_needed skipped=true")
            return cast(RetrievalStageState, updated)
        try:
            extraction = self.dependencies.extract_legal_entities(updated["effective_query"])
            updated["extracted_entities"] = extraction
            updated["filters"] = _derive_filters(extraction)
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"entity_extraction_failed:{type(exc).__name__}"]
            updated["extracted_entities"] = None
            updated["filters"] = None
        logger.info(
            "node_exit name=extract_entities_if_needed has_filters=%s",
            bool(updated["filters"]),
        )
        return cast(RetrievalStageState, updated)

    def run_hybrid_search(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=run_hybrid_search")
        updated = dict(state)
        try:
            results = self.dependencies.hybrid_search(
                updated["effective_query"],
                filters=updated["filters"],
                top_k=self.config.hybrid_top_k,
            )
            updated["child_results"] = list(results)
            if not results:
                updated["warnings"] = [*updated["warnings"], "no_child_results"]
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"hybrid_search_failed:{type(exc).__name__}"]
            updated["child_results"] = []
        logger.info("node_exit name=run_hybrid_search child_count=%s", len(updated["child_results"]))
        return cast(RetrievalStageState, updated)

    def rerank_results(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=rerank_results child_count=%s", len(state["child_results"]))
        updated = dict(state)
        if not updated["child_results"]:
            updated["reranked_child_results"] = []
            logger.info("node_exit name=rerank_results skipped=true")
            return cast(RetrievalStageState, updated)
        try:
            reranked = self.dependencies.rerank_chunks(updated["child_results"], updated["effective_query"])
            if not reranked:
                updated["warnings"] = [*updated["warnings"], "rerank_empty_fallback_to_child_results"]
                fallback = [
                    RerankedChunkResult(
                        child_chunk_id=item.child_chunk_id,
                        parent_chunk_id=item.parent_chunk_id,
                        document_id=item.document_id,
                        text=item.text,
                        rerank_score=item.hybrid_score,
                        original_score=item.hybrid_score,
                        payload=dict(item.payload),
                    )
                    for item in updated["child_results"]
                ]
                updated["reranked_child_results"] = fallback
            else:
                updated["reranked_child_results"] = list(reranked)
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"rerank_failed:{type(exc).__name__}"]
            updated["reranked_child_results"] = [
                RerankedChunkResult(
                    child_chunk_id=item.child_chunk_id,
                    parent_chunk_id=item.parent_chunk_id,
                    document_id=item.document_id,
                    text=item.text,
                    rerank_score=item.hybrid_score,
                    original_score=item.hybrid_score,
                    payload=dict(item.payload),
                )
                for item in updated["child_results"]
            ]
        logger.info("node_exit name=rerank_results reranked_count=%s", len(updated["reranked_child_results"]))
        return cast(RetrievalStageState, updated)

    def collect_parent_ids(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=collect_parent_ids")
        updated = dict(state)
        seen: set[str] = set()
        parent_ids: list[str] = []
        for item in updated["reranked_child_results"]:
            parent_id = item.parent_chunk_id
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            parent_ids.append(parent_id)
        updated["parent_ids"] = parent_ids
        logger.info("node_exit name=collect_parent_ids parent_count=%s", len(parent_ids))
        return cast(RetrievalStageState, updated)

    def fetch_parent_chunks(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=fetch_parent_chunks parent_id_count=%s", len(state["parent_ids"]))
        updated = dict(state)
        try:
            parents = self.dependencies.retrieve_parent_chunks(updated["parent_ids"])
            updated["parent_chunks"] = list(parents)
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"parent_retrieval_failed:{type(exc).__name__}"]
            updated["parent_chunks"] = []
        logger.info("node_exit name=fetch_parent_chunks parent_chunk_count=%s", len(updated["parent_chunks"]))
        return cast(RetrievalStageState, updated)

    def maybe_compress_context(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=maybe_compress_context")
        updated = dict(state)
        parent_chunks = updated["parent_chunks"]
        total_tokens = sum(self.config.token_estimator(parent.text) for parent in parent_chunks)
        should_compress = (
            len(parent_chunks) >= self.config.compress_if_parent_chunks_gte
            or total_tokens >= self.config.compress_if_total_parent_tokens_gte
        )
        updated["should_compress"] = should_compress
        logger.info(
            "node_exit name=maybe_compress_context should_compress=%s parent_chunks=%s total_tokens=%s",
            should_compress,
            len(parent_chunks),
            total_tokens,
        )
        return cast(RetrievalStageState, updated)

    def compress_context_node(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=compress_context_node")
        updated = dict(state)
        try:
            result = self.dependencies.compress_context(updated["parent_chunks"])
            updated["compressed_context"] = list(result.items)
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"compression_failed:{type(exc).__name__}"]
            updated["compressed_context"] = []
        updated["retrieval_stage_complete"] = True
        logger.info("node_exit name=compress_context_node compressed_count=%s", len(updated["compressed_context"]))
        return cast(RetrievalStageState, updated)

    def mark_complete_without_compression(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=mark_complete_without_compression")
        updated = dict(state)
        updated["retrieval_stage_complete"] = True
        logger.info("node_exit name=mark_complete_without_compression")
        return cast(RetrievalStageState, updated)


DecisionLiteral = Literal[
    "rewrite_query_if_needed",
    "extract_entities_if_needed",
    "compress_context_node",
    "mark_complete_without_compression",
]


def _route_after_classification(state: RetrievalStageState) -> DecisionLiteral:
    return "rewrite_query_if_needed" if state["should_rewrite"] else "extract_entities_if_needed"


def _route_after_compression_check(state: RetrievalStageState) -> DecisionLiteral:
    return "compress_context_node" if state["should_compress"] else "mark_complete_without_compression"


class _FallbackCompiledGraph:
    """Small deterministic executor used only when LangGraph is unavailable."""

    def __init__(self, nodes: RetrievalGraphNodes) -> None:
        self._nodes = nodes

    def invoke(self, state: RetrievalStageState) -> RetrievalStageState:
        current = state
        current = self._nodes.ingest_turn(current)
        current = self._nodes.classify_query_state(current)
        current = self._nodes.resolve_query_context(current)
        if _route_after_classification(current) == "rewrite_query_if_needed":
            current = self._nodes.rewrite_query_if_needed(current)
        current = self._nodes.extract_entities_if_needed(current)
        current = self._nodes.run_hybrid_search(current)
        current = self._nodes.rerank_results(current)
        current = self._nodes.collect_parent_ids(current)
        current = self._nodes.fetch_parent_chunks(current)
        current = self._nodes.maybe_compress_context(current)
        if _route_after_compression_check(current) == "compress_context_node":
            current = self._nodes.compress_context_node(current)
        else:
            current = self._nodes.mark_complete_without_compression(current)
        return current


def build_retrieval_graph(
    dependencies: RetrievalDependencies,
    *,
    config: RetrievalGraphConfig | None = None,
) -> Any:
    """Build retrieval StateGraph with explicit nodes and code-driven edges."""

    resolved_config = config or RetrievalGraphConfig()
    nodes = RetrievalGraphNodes(dependencies=dependencies, config=resolved_config)

    if StateGraph is None:  # pragma: no cover
        return _FallbackCompiledGraph(nodes)

    graph = StateGraph(RetrievalStageState)
    graph.add_node("ingest_turn", nodes.ingest_turn)
    graph.add_node("classify_query_state", nodes.classify_query_state)
    graph.add_node("resolve_query_context", nodes.resolve_query_context)
    graph.add_node("rewrite_query_if_needed", nodes.rewrite_query_if_needed)
    graph.add_node("extract_entities_if_needed", nodes.extract_entities_if_needed)
    graph.add_node("run_hybrid_search", nodes.run_hybrid_search)
    graph.add_node("rerank_results", nodes.rerank_results)
    graph.add_node("collect_parent_ids", nodes.collect_parent_ids)
    graph.add_node("fetch_parent_chunks", nodes.fetch_parent_chunks)
    graph.add_node("maybe_compress_context", nodes.maybe_compress_context)
    graph.add_node("compress_context_node", nodes.compress_context_node)
    graph.add_node("mark_complete_without_compression", nodes.mark_complete_without_compression)

    graph.add_edge(START, "ingest_turn")
    graph.add_edge("ingest_turn", "classify_query_state")
    graph.add_edge("classify_query_state", "resolve_query_context")
    graph.add_conditional_edges(
        "resolve_query_context",
        _route_after_classification,
        {
            "rewrite_query_if_needed": "rewrite_query_if_needed",
            "extract_entities_if_needed": "extract_entities_if_needed",
        },
    )
    graph.add_edge("rewrite_query_if_needed", "extract_entities_if_needed")
    graph.add_edge("extract_entities_if_needed", "run_hybrid_search")
    graph.add_edge("run_hybrid_search", "rerank_results")
    graph.add_edge("rerank_results", "collect_parent_ids")
    graph.add_edge("collect_parent_ids", "fetch_parent_chunks")
    graph.add_edge("fetch_parent_chunks", "maybe_compress_context")
    graph.add_conditional_edges(
        "maybe_compress_context",
        _route_after_compression_check,
        {
            "compress_context_node": "compress_context_node",
            "mark_complete_without_compression": "mark_complete_without_compression",
        },
    )
    graph.add_edge("compress_context_node", END)
    graph.add_edge("mark_complete_without_compression", END)

    return graph.compile()


def run_retrieval_stage(
    *,
    query: str,
    dependencies: RetrievalDependencies,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    config: RetrievalGraphConfig | None = None,
) -> RetrievalStageState:
    """Invoke retrieval graph and return state ready for downstream answer synthesis."""

    initial_state = default_retrieval_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )
    app = build_retrieval_graph(dependencies=dependencies, config=config)
    return cast(RetrievalStageState, app.invoke(initial_state))
