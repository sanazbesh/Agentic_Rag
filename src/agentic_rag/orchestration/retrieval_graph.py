"""Deterministic retrieval-stage orchestration graph for legal RAG.

This module intentionally implements retrieval as an explicit graph (node + edge
transitions) rather than a free-form agent loop so production behavior remains
traceable, testable, and bounded.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, cast

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.orchestration.decomposition_gate import decide_decomposition_need
from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult, understand_query
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


QueryRoutingDecision = QueryUnderstandingResult


class QueryContextResolution(BaseModel):
    """Typed, inspectable output for conservative follow-up reference resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolved_query: str
    used_conversation_context: bool
    resolved_document_ids: list[str] = Field(default_factory=list)
    resolved_topic_hints: list[str] = Field(default_factory=list)
    resolution_notes: list[str] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)


class SubQueryPlan(BaseModel):
    """Typed planner sub-query schema reserved for future decomposition wiring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    purpose: str
    required: bool
    expected_answer_type: Literal[
        "definition",
        "entity",
        "date",
        "obligation",
        "exception",
        "comparison",
        "condition",
        "cross_reference",
    ]
    dependency_ids: list[str] = Field(default_factory=list)


class DecompositionPlan(BaseModel):
    """Typed decomposition plan schema reserved for future planner integration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    should_decompose: bool
    root_question: str
    strategy: Literal[
        "conjunctive",
        "comparison",
        "temporal",
        "exception_chain",
        "cross_clause",
        "definition_plus_application",
        "amendment_vs_base",
    ] | None = None
    subqueries: list[SubQueryPlan] = Field(default_factory=list)
    planner_notes: list[str] = Field(default_factory=list)




class SubqueryRetrievalResult(BaseModel):
    """Stored retrieval output for a validated decomposition subquery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subquery_id: str
    subquery_question: str
    hits: list[HybridSearchResult] = Field(default_factory=list)


class RetrievalCandidateProvenance(BaseModel):
    """Query-path provenance for future merged retrieval candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_root_query: bool = False
    subquery_ids: list[str] = Field(default_factory=list)


class MergedRetrievalCandidate(BaseModel):
    """Stable candidate contract for future root+subquery merge phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit: HybridSearchResult
    provenance: RetrievalCandidateProvenance
    contributing_hits: list[HybridSearchResult] = Field(default_factory=list)


class RetrievalStageState(TypedDict):
    """Strict retrieval-stage state shared by all graph nodes."""

    original_query: str
    conversation_summary: str | None
    recent_messages: list[Mapping[str, Any]]
    active_documents: list[Any]
    selected_documents: list[Any]
    use_conversation_context: bool

    rewritten_query: str | None
    resolved_query: str
    effective_query: str
    query_classification: QueryRoutingDecision | None
    context_resolution: QueryContextResolution | None
    needs_decomposition: bool
    decomposition_plan: DecompositionPlan | None
    decomposition_validation_errors: list[str]
    decomposition_gate_reasons: list[str]

    extracted_entities: LegalEntityExtractionResult | None
    filters: dict[str, Any] | None

    child_results: list[HybridSearchResult]
    subquery_results: dict[str, SubqueryRetrievalResult]
    root_merged_candidates: list[MergedRetrievalCandidate]
    subquery_merged_candidates: dict[str, list[MergedRetrievalCandidate]]
    merged_candidates: list[MergedRetrievalCandidate]
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
        active_documents: Sequence[Any] | None = None,
        selected_documents: Sequence[Any] | None = None,
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
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
) -> RetrievalStageState:
    """Build strict initial state with explicit defaults for all list fields."""

    normalized_query = (query or "").strip()
    return RetrievalStageState(
        original_query=normalized_query,
        conversation_summary=conversation_summary,
        recent_messages=list(recent_messages or []),
        active_documents=list(active_documents or []),
        selected_documents=list(selected_documents or []),
        use_conversation_context=False,
        rewritten_query=None,
        resolved_query=normalized_query,
        effective_query=normalized_query,
        query_classification=None,
        context_resolution=None,
        needs_decomposition=False,
        decomposition_plan=None,
        decomposition_validation_errors=[],
        decomposition_gate_reasons=[],
        extracted_entities=None,
        filters=None,
        child_results=[],
        subquery_results={},
        root_merged_candidates=[],
        subquery_merged_candidates={},
        merged_candidates=[],
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
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
) -> QueryRoutingDecision:
    """Conservative deterministic classifier backed by strict understand_query()."""

    return understand_query(
        query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        active_documents=active_documents,
        selected_documents=selected_documents,
    )


def _build_merged_candidate(
    *,
    hit: HybridSearchResult,
    from_root_query: bool,
    subquery_ids: Sequence[str] | None = None,
) -> MergedRetrievalCandidate:
    ordered_subquery_ids = [subquery_id for subquery_id in (subquery_ids or []) if subquery_id]
    return MergedRetrievalCandidate(
        hit=hit,
        provenance=RetrievalCandidateProvenance(
            from_root_query=from_root_query,
            subquery_ids=ordered_subquery_ids,
        ),
        contributing_hits=[hit],
    )


def _merge_two_candidates(
    *,
    existing: MergedRetrievalCandidate,
    incoming: MergedRetrievalCandidate,
) -> MergedRetrievalCandidate:
    merged_metadata = dict(existing.hit.metadata)
    for key, value in incoming.hit.metadata.items():
        if key not in merged_metadata:
            merged_metadata[key] = value

    merged_hit = HybridSearchResult(
        child_chunk_id=existing.hit.child_chunk_id,
        parent_chunk_id=existing.hit.parent_chunk_id or incoming.hit.parent_chunk_id,
        document_id=existing.hit.document_id or incoming.hit.document_id,
        text=existing.hit.text or incoming.hit.text,
        hybrid_score=existing.hit.hybrid_score,
        metadata=merged_metadata,
        dense_score=existing.hit.dense_score if existing.hit.dense_score is not None else incoming.hit.dense_score,
        sparse_score=existing.hit.sparse_score if existing.hit.sparse_score is not None else incoming.hit.sparse_score,
        dense_rank=existing.hit.dense_rank if existing.hit.dense_rank is not None else incoming.hit.dense_rank,
        sparse_rank=existing.hit.sparse_rank if existing.hit.sparse_rank is not None else incoming.hit.sparse_rank,
        matched_in_dense=existing.hit.matched_in_dense or incoming.hit.matched_in_dense,
        matched_in_sparse=existing.hit.matched_in_sparse or incoming.hit.matched_in_sparse,
    )

    merged_subquery_ids = list(existing.provenance.subquery_ids)
    for subquery_id in incoming.provenance.subquery_ids:
        if subquery_id and subquery_id not in merged_subquery_ids:
            merged_subquery_ids.append(subquery_id)

    return MergedRetrievalCandidate(
        hit=merged_hit,
        provenance=RetrievalCandidateProvenance(
            from_root_query=existing.provenance.from_root_query or incoming.provenance.from_root_query,
            subquery_ids=merged_subquery_ids,
        ),
        contributing_hits=[*existing.contributing_hits, *incoming.contributing_hits],
    )


def _merge_root_and_subquery_candidates(
    *,
    root_candidates: Sequence[MergedRetrievalCandidate],
    subquery_candidates: Mapping[str, Sequence[MergedRetrievalCandidate]],
) -> list[MergedRetrievalCandidate]:
    by_child_chunk_id: dict[str, MergedRetrievalCandidate] = {}

    for candidate in root_candidates:
        child_chunk_id = candidate.hit.child_chunk_id
        if child_chunk_id not in by_child_chunk_id:
            by_child_chunk_id[child_chunk_id] = candidate
        else:
            by_child_chunk_id[child_chunk_id] = _merge_two_candidates(
                existing=by_child_chunk_id[child_chunk_id],
                incoming=candidate,
            )

    for subquery_id in sorted(subquery_candidates):
        for candidate in subquery_candidates[subquery_id]:
            child_chunk_id = candidate.hit.child_chunk_id
            existing = by_child_chunk_id.get(child_chunk_id)
            if existing is None:
                by_child_chunk_id[child_chunk_id] = candidate
            else:
                by_child_chunk_id[child_chunk_id] = _merge_two_candidates(existing=existing, incoming=candidate)

    return list(by_child_chunk_id.values())


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


def classify_decomposition_need(
    *,
    query: str,
    query_classification: QueryRoutingDecision | None,
    context_resolution: QueryContextResolution | None,
) -> tuple[bool, list[str]]:
    """Compatibility wrapper around the centralized decomposition gate helper."""

    context_payload = context_resolution.model_dump() if context_resolution is not None else None
    decision = decide_decomposition_need(
        query=query,
        query_context=context_payload,
        query_understanding=query_classification,
    )
    needs_decomposition = getattr(decision, "needs_decomposition", None)
    reasons = getattr(decision, "reasons", None)
    if not isinstance(needs_decomposition, bool):
        return False, []
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        return False, []
    return needs_decomposition, list(reasons)


def _pick_strategy(reasons: Sequence[str]) -> Literal[
    "conjunctive",
    "comparison",
    "temporal",
    "exception_chain",
    "cross_clause",
    "definition_plus_application",
    "amendment_vs_base",
] | None:
    ordered = list(reasons)
    if "amendment_vs_base" in ordered:
        return "amendment_vs_base"
    if "comparison_query" in ordered:
        return "comparison"
    if "exception_chain" in ordered:
        return "exception_chain"
    if "temporal_relationship" in ordered:
        return "temporal"
    if "cross_clause_obligation_condition" in ordered:
        return "cross_clause"
    if "multi_intent_conjunction" in ordered:
        return "conjunctive"
    if "context_dependent_followup" in ordered:
        return "definition_plus_application"
    return None


def _extract_comparison_terms(query: str) -> tuple[str | None, str | None]:
    cleaned = " ".join((query or "").split())
    patterns = (
        re.compile(r"(?:compare|difference(?:s)?\s+between)\s+(.+?)\s+(?:and|versus|vs\.?|compared\s+to)\s+(.+?)(?:[?.]|$)", re.IGNORECASE),
        re.compile(r"(.+?)\s+(?:versus|vs\.?|compared\s+to)\s+(.+?)(?:[?.]|$)", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        left = match.group(1).strip(" ,")
        right = match.group(2).strip(" ,")
        if left and right:
            return left, right
    return None, None


def build_decomposition_plan(
    *,
    query: str,
    needs_decomposition: bool,
    reasons: Sequence[str],
    query_classification: QueryRoutingDecision | None,
    context_resolution: QueryContextResolution | None,
) -> DecompositionPlan | None:
    """Build a bounded typed decomposition plan only when the gate requires it."""

    if not needs_decomposition:
        return None

    root_question = " ".join((query or "").split())
    strategy = _pick_strategy(reasons)
    plan_notes = [f"gate_reason:{reason}" for reason in reasons]

    if context_resolution is not None:
        if context_resolution.resolved_document_ids:
            plan_notes.append("preserve_document_scope")
        if context_resolution.resolved_topic_hints:
            plan_notes.append("preserve_topic_scope")

    if query_classification is not None and query_classification.is_context_dependent:
        plan_notes.append("context_dependent_query")

    if any(token in root_question.lower() for token in (" not ", " except", " unless", " notwithstanding")):
        plan_notes.append("preserve_negation_and_exceptions")

    subqueries: list[SubQueryPlan]
    if strategy == "comparison":
        left, right = _extract_comparison_terms(root_question)
        if left and right:
            subqueries = [
                SubQueryPlan(
                    id="sq-1",
                    question=f"Locate clauses about {left} within the same agreement scope as the root question.",
                    purpose="Retrieve the first side of the comparison without synthesis.",
                    required=True,
                    expected_answer_type="cross_reference",
                ),
                SubQueryPlan(
                    id="sq-2",
                    question=f"Locate clauses about {right} within the same agreement scope as the root question.",
                    purpose="Retrieve the second side of the comparison without synthesis.",
                    required=True,
                    expected_answer_type="cross_reference",
                ),
            ]
        else:
            subqueries = [
                SubQueryPlan(
                    id="sq-1",
                    question=f"Locate the clause segments needed to compare the requested items in: {root_question}",
                    purpose="Collect comparison evidence only.",
                    required=True,
                    expected_answer_type="comparison",
                )
            ]
    elif strategy == "amendment_vs_base":
        subqueries = [
            SubQueryPlan(
                id="sq-1",
                question=f"Locate the base-agreement clause(s) relevant to: {root_question}",
                purpose="Retrieve baseline clause text before amendment impact analysis.",
                required=True,
                expected_answer_type="cross_reference",
            ),
            SubQueryPlan(
                id="sq-2",
                question=f"Locate the amendment clause(s) relevant to: {root_question}",
                purpose="Retrieve amendment text scoped to the same issue.",
                required=True,
                expected_answer_type="cross_reference",
            ),
        ]
    elif strategy == "exception_chain":
        subqueries = [
            SubQueryPlan(
                id="sq-1",
                question=f"Locate the primary rule clause implicated by: {root_question}",
                purpose="Find base obligation/condition text for exception tracing.",
                required=True,
                expected_answer_type="obligation",
            ),
            SubQueryPlan(
                id="sq-2",
                question=f"Locate exception or carve-out language (for example 'unless', 'except', 'notwithstanding') tied to: {root_question}",
                purpose="Find exception chain text without concluding outcomes.",
                required=True,
                expected_answer_type="exception",
                dependency_ids=["sq-1"],
            ),
        ]
    else:
        subqueries = [
            SubQueryPlan(
                id="sq-1",
                question=f"Locate clause text needed to resolve: {root_question}",
                purpose="Collect retrieval evidence only; no answer generation.",
                required=True,
                expected_answer_type="cross_reference",
            )
        ]

    return DecompositionPlan(
        should_decompose=True,
        root_question=root_question,
        strategy=strategy,
        subqueries=subqueries,
        planner_notes=plan_notes,
    )


_DECOMPOSITION_MAX_SUBQUERIES = 4
_VAGUE_SUBQUERY_PATTERNS = (
    "anything relevant",
    "everything about",
    "general overview",
    "overall summary",
    "all clauses",
    "what does it say",
    "tell me more",
)
_ROOT_SCOPE_MARKERS = (
    "agreement",
    "contract",
    "msa",
    "nda",
    "lease",
    "amendment",
    "clause",
    "section",
    "article",
    "governing law",
    "jurisdiction",
)
_NEGATION_MARKERS = (
    " not ",
    " except",
    " unless",
    " notwithstanding",
    " other than",
)


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _canonical_text(value: str) -> str:
    normalized = _normalize_text(value).lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return " ".join(normalized.split())


def _extract_root_entities(root_question: str) -> list[str]:
    # Capture simple title-cased entity phrases (for example: "Acme Corp").
    matches = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b", root_question)
    entities = sorted({_canonical_text(match) for match in matches if match.strip()})
    return [entity for entity in entities if entity]


def validate_decomposition_plan(plan: DecompositionPlan) -> list[str]:
    """Deterministic conservative checks to keep broken plans out of retrieval."""

    errors: list[str] = []
    subqueries = list(plan.subqueries)
    canonical_subqueries = [_canonical_text(subquery.question) for subquery in subqueries]

    if len(subqueries) > _DECOMPOSITION_MAX_SUBQUERIES:
        errors.append(f"too_many_subqueries:max_{_DECOMPOSITION_MAX_SUBQUERIES}")

    non_empty_questions = [question for question in canonical_subqueries if question]
    if len(non_empty_questions) != len(set(non_empty_questions)):
        errors.append("duplicate_subqueries")

    root_question = _normalize_text(plan.root_question)
    root_canonical = f" {_canonical_text(root_question)} "
    joined_subqueries = " ".join(canonical_subqueries)
    joined_subqueries_padded = f" {joined_subqueries} "

    dropped_entities: list[str] = []
    for entity in _extract_root_entities(root_question):
        if entity not in joined_subqueries:
            dropped_entities.append(entity)

    dropped_scopes: list[str] = []
    for marker in _ROOT_SCOPE_MARKERS:
        marker_canonical = _canonical_text(marker)
        if marker_canonical and marker_canonical in root_canonical and marker_canonical not in joined_subqueries_padded:
            dropped_scopes.append(marker_canonical)

    if dropped_entities or dropped_scopes:
        scope_parts = [f"entity={value}" for value in dropped_entities] + [f"scope={value}" for value in dropped_scopes]
        errors.append("dropped_key_entity_or_scope:" + ",".join(scope_parts))

    root_has_negation = any(marker in root_canonical for marker in _NEGATION_MARKERS)
    subqueries_preserve_negation = any(marker in joined_subqueries_padded for marker in _NEGATION_MARKERS) or any(
        token in joined_subqueries for token in ("exception", "carve out", "carveout", "negation", "unless", "except")
    )
    if root_has_negation and not subqueries_preserve_negation:
        errors.append("lost_negation_or_exception_logic")

    for subquery in subqueries:
        question = _canonical_text(subquery.question)
        word_count = len(question.split())
        if word_count < 5 or any(pattern in question for pattern in _VAGUE_SUBQUERY_PATTERNS):
            errors.append(f"vague_or_overly_broad_subquery:{subquery.id}")

    return errors


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
        raw_subquery_results = updated.get("subquery_results", {})
        if isinstance(raw_subquery_results, Mapping):
            normalized_subquery_results: dict[str, SubqueryRetrievalResult] = {}
            for key, value in raw_subquery_results.items():
                if isinstance(value, SubqueryRetrievalResult):
                    normalized_subquery_results[str(key)] = value
            updated["subquery_results"] = normalized_subquery_results
        else:
            updated["subquery_results"] = {}
        updated["root_merged_candidates"] = list(updated.get("root_merged_candidates", []))
        updated["merged_candidates"] = list(updated.get("merged_candidates", []))
        raw_subquery_candidates = updated.get("subquery_merged_candidates", {})
        if isinstance(raw_subquery_candidates, Mapping):
            normalized_subquery_candidates: dict[str, list[MergedRetrievalCandidate]] = {}
            for key, value in raw_subquery_candidates.items():
                if isinstance(value, list):
                    normalized_subquery_candidates[str(key)] = [
                        item for item in value if isinstance(item, MergedRetrievalCandidate)
                    ]
            updated["subquery_merged_candidates"] = normalized_subquery_candidates
        else:
            updated["subquery_merged_candidates"] = {}
        updated["reranked_child_results"] = list(updated.get("reranked_child_results", []))
        updated["parent_ids"] = list(updated.get("parent_ids", []))
        updated["parent_chunks"] = list(updated.get("parent_chunks", []))
        updated["compressed_context"] = list(updated.get("compressed_context", []))
        updated["decomposition_validation_errors"] = list(updated.get("decomposition_validation_errors", []) or [])
        updated["decomposition_plan"] = updated.get("decomposition_plan")
        self._hydrate_prior_turn_memory(updated)
        logger.info("node_exit name=ingest_turn query_length=%s", len(updated["original_query"]))
        return cast(RetrievalStageState, updated)

    def _hydrate_prior_turn_memory(self, state: dict[str, Any]) -> None:
        messages = list(state.get("recent_messages", []))
        assistant_messages = [msg for msg in reversed(messages) if str(msg.get("role", "")).lower() == "assistant"]
        for assistant in assistant_messages:
            metadata = assistant.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if not state.get("last_resolved_document_scope"):
                state["last_resolved_document_scope"] = [
                    str(item) for item in metadata.get("resolved_document_ids", []) if item
                ]
            if not state.get("last_resolved_topic"):
                topic_hints = [str(item) for item in metadata.get("resolved_topic_hints", []) if item]
                state["last_resolved_topic"] = topic_hints[0] if topic_hints else None
            if not state.get("prior_effective_query"):
                prior_effective_query = metadata.get("effective_query")
                if isinstance(prior_effective_query, str) and prior_effective_query.strip():
                    state["prior_effective_query"] = prior_effective_query
            if not state.get("prior_final_answer"):
                prior_final_answer = metadata.get("answer_text")
                if isinstance(prior_final_answer, str) and prior_final_answer.strip():
                    state["prior_final_answer"] = prior_final_answer
            if not state.get("prior_citations"):
                citations = metadata.get("citations")
                if isinstance(citations, list):
                    state["prior_citations"] = [item for item in citations if isinstance(item, Mapping)]
            if (
                state.get("last_resolved_document_scope")
                and state.get("last_resolved_topic")
                and state.get("prior_effective_query")
                and state.get("prior_final_answer")
                and state.get("prior_citations")
            ):
                return

    def classify_query_state(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=classify_query_state")
        updated = dict(state)
        active_docs = list(updated.get("active_documents") or [])
        selected_docs = list(updated.get("selected_documents") or [])
        selected_ids = [
            str(item.get("id"))
            for item in selected_docs
            if isinstance(item, Mapping) and item.get("id")
        ]
        logger.info(
            "classify_query_state_inputs active_documents_count=%s selected_documents_count=%s selected_document_ids=%s",
            len(active_docs),
            len(selected_docs),
            selected_ids,
        )
        decision = self.dependencies.classify_query_state(
            updated["original_query"],
            conversation_summary=updated["conversation_summary"],
            recent_messages=updated["recent_messages"],
            active_documents=active_docs,
            selected_documents=selected_docs,
        )
        updated["query_classification"] = decision
        updated["use_conversation_context"] = decision.use_conversation_context
        updated["should_rewrite"] = decision.should_rewrite
        updated["should_extract_entities"] = decision.should_extract_entities
        logger.info(
            "node_exit name=classify_query_state followup=%s ambiguous=%s rewrite=%s extract=%s final_question_type=%s final_answerability_expectation=%s",
            decision.is_followup,
            decision.question_type == "ambiguous_query",
            decision.should_rewrite,
            decision.should_extract_entities,
            decision.question_type,
            decision.answerability_expectation,
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

        if decision.is_context_dependent:
            if decision.refers_to_prior_document_scope and len(candidate_doc_ids) > 1:
                unresolved_references.append("document_reference:ambiguous_multiple_candidates")
                resolution_notes.append("ambiguous_document_reference")
            elif decision.refers_to_prior_document_scope and len(candidate_doc_ids) == 1:
                if decision.use_conversation_context:
                    used_context = True
                    resolution_notes.append("resolved_document_scope_from_prior_turn")
            elif decision.refers_to_prior_document_scope and not candidate_doc_ids:
                unresolved_references.append("document_reference:missing_prior_scope")
                resolution_notes.append("unable_to_resolve_document_scope")

            if decision.refers_to_prior_clause_or_topic and topic_hints and decision.use_conversation_context:
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

    def classify_decomposition_need(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=classify_decomposition_need")
        updated = dict(state)
        needs_decomposition, reasons = classify_decomposition_need(
            query=str(updated.get("resolved_query") or updated.get("original_query") or ""),
            query_classification=updated.get("query_classification"),
            context_resolution=updated.get("context_resolution"),
        )
        updated["needs_decomposition"] = needs_decomposition
        updated["decomposition_gate_reasons"] = reasons
        logger.info(
            "node_exit name=classify_decomposition_need needs_decomposition=%s reasons=%s",
            needs_decomposition,
            ",".join(reasons),
        )
        return cast(RetrievalStageState, updated)

    def maybe_build_decomposition_plan(self, state: RetrievalStageState) -> RetrievalStageState:
        logger.info("node_enter name=maybe_build_decomposition_plan")
        updated = dict(state)
        plan = build_decomposition_plan(
            query=str(updated.get("resolved_query") or updated.get("original_query") or ""),
            needs_decomposition=bool(updated.get("needs_decomposition")),
            reasons=list(updated.get("decomposition_gate_reasons") or []),
            query_classification=updated.get("query_classification"),
            context_resolution=updated.get("context_resolution"),
        )
        updated["decomposition_plan"] = plan
        logger.info(
            "node_exit name=maybe_build_decomposition_plan plan_created=%s subquery_count=%s",
            plan is not None,
            len(plan.subqueries) if plan is not None else 0,
        )
        return cast(RetrievalStageState, updated)

    def validate_decomposition_plan(self, state: RetrievalStageState) -> RetrievalStageState:
        """Validate typed decomposition plans and clear invalid plans conservatively."""

        logger.info("node_enter name=validate_decomposition_plan")
        updated = dict(state)
        plan = updated.get("decomposition_plan")
        if plan is None:
            updated["decomposition_validation_errors"] = []
            logger.info("node_exit name=validate_decomposition_plan skipped=true")
            return cast(RetrievalStageState, updated)

        errors = validate_decomposition_plan(plan)
        updated["decomposition_validation_errors"] = list(errors)
        if errors:
            updated["decomposition_plan"] = None
            logger.info(
                "node_exit name=validate_decomposition_plan valid=false error_count=%s",
                len(errors),
            )
            return cast(RetrievalStageState, updated)

        logger.info("node_exit name=validate_decomposition_plan valid=true")
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
            merged_filters = dict(updated.get("filters") or {})
            merged_filters.update(_derive_filters(extraction) or {})
            updated["filters"] = merged_filters or None
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"entity_extraction_failed:{type(exc).__name__}"]
            updated["extracted_entities"] = None
            updated["filters"] = None
        logger.info(
            "node_exit name=extract_entities_if_needed has_filters=%s",
            bool(updated["filters"]),
        )
        return cast(RetrievalStageState, updated)

    def run_subquery_hybrid_search(self, state: RetrievalStageState) -> RetrievalStageState:
        """Run hybrid retrieval for each validated decomposition subquery."""

        logger.info("node_enter name=run_subquery_hybrid_search")
        updated = dict(state)
        plan = updated.get("decomposition_plan")
        if plan is None or not plan.subqueries:
            updated["subquery_results"] = {}
            updated["subquery_merged_candidates"] = {}
            logger.info("node_exit name=run_subquery_hybrid_search skipped=true")
            return cast(RetrievalStageState, updated)

        filters = updated.get("filters")
        subquery_results: dict[str, SubqueryRetrievalResult] = {}
        subquery_merged_candidates: dict[str, list[MergedRetrievalCandidate]] = {}
        for subquery in plan.subqueries:
            try:
                hits = self.dependencies.hybrid_search(
                    subquery.question,
                    filters=filters,
                    top_k=self.config.hybrid_top_k,
                )
                normalized_hits = list(hits)
                if not normalized_hits:
                    updated["warnings"] = [*updated["warnings"], f"no_subquery_child_results:{subquery.id}"]
            except Exception as exc:  # pragma: no cover
                normalized_hits = []
                updated["warnings"] = [*updated["warnings"], f"subquery_hybrid_search_failed:{subquery.id}:{type(exc).__name__}"]

            subquery_results[subquery.id] = SubqueryRetrievalResult(
                subquery_id=subquery.id,
                subquery_question=subquery.question,
                hits=normalized_hits,
            )
            subquery_merged_candidates[subquery.id] = [
                _build_merged_candidate(hit=hit, from_root_query=False, subquery_ids=[subquery.id])
                for hit in normalized_hits
            ]

        updated["subquery_results"] = subquery_results
        updated["subquery_merged_candidates"] = subquery_merged_candidates
        logger.info("node_exit name=run_subquery_hybrid_search subquery_count=%s", len(subquery_results))
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
            updated["root_merged_candidates"] = [
                _build_merged_candidate(hit=item, from_root_query=True) for item in updated["child_results"]
            ]
            if not results:
                updated["warnings"] = [*updated["warnings"], "no_child_results"]
        except Exception as exc:  # pragma: no cover
            updated["warnings"] = [*updated["warnings"], f"hybrid_search_failed:{type(exc).__name__}"]
            updated["child_results"] = []
            updated["root_merged_candidates"] = []
        logger.info("node_exit name=run_hybrid_search child_count=%s", len(updated["child_results"]))
        return cast(RetrievalStageState, updated)

    def merge_retrieval_candidates(self, state: RetrievalStageState) -> RetrievalStageState:
        """Merge root/subquery retrieval candidates into one deduped deterministic pool."""

        logger.info("node_enter name=merge_retrieval_candidates")
        updated = dict(state)
        if updated.get("decomposition_plan") is None:
            updated["merged_candidates"] = []
            logger.info("node_exit name=merge_retrieval_candidates skipped=true")
            return cast(RetrievalStageState, updated)

        updated["merged_candidates"] = _merge_root_and_subquery_candidates(
            root_candidates=list(updated.get("root_merged_candidates", [])),
            subquery_candidates=dict(updated.get("subquery_merged_candidates", {})),
        )
        logger.info(
            "node_exit name=merge_retrieval_candidates merged_count=%s",
            len(updated["merged_candidates"]),
        )
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
    "compress_context_node",
    "mark_complete_without_compression",
]


DecompositionRoutingDecision = Literal[
    "maybe_build_decomposition_plan",
    "rewrite_query_if_needed",
]


def _route_after_decomposition_gate(state: RetrievalStageState) -> DecompositionRoutingDecision:
    return "maybe_build_decomposition_plan" if state["needs_decomposition"] else "rewrite_query_if_needed"


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
        current = self._nodes.classify_decomposition_need(current)
        if _route_after_decomposition_gate(current) == "maybe_build_decomposition_plan":
            current = self._nodes.maybe_build_decomposition_plan(current)
            current = self._nodes.validate_decomposition_plan(current)
        else:
            current = dict(current)
            current["decomposition_plan"] = None
            current["decomposition_validation_errors"] = []
        current = self._nodes.rewrite_query_if_needed(current)
        current = self._nodes.extract_entities_if_needed(current)
        current = self._nodes.run_subquery_hybrid_search(current)
        current = self._nodes.run_hybrid_search(current)
        current = self._nodes.merge_retrieval_candidates(current)
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
    graph.add_node("classify_decomposition_need", nodes.classify_decomposition_need)
    graph.add_node("maybe_build_decomposition_plan", nodes.maybe_build_decomposition_plan)
    graph.add_node("validate_decomposition_plan", nodes.validate_decomposition_plan)
    graph.add_node("rewrite_query_if_needed", nodes.rewrite_query_if_needed)
    graph.add_node("extract_entities_if_needed", nodes.extract_entities_if_needed)
    graph.add_node("run_subquery_hybrid_search", nodes.run_subquery_hybrid_search)
    graph.add_node("run_hybrid_search", nodes.run_hybrid_search)
    graph.add_node("merge_retrieval_candidates", nodes.merge_retrieval_candidates)
    graph.add_node("rerank_results", nodes.rerank_results)
    graph.add_node("collect_parent_ids", nodes.collect_parent_ids)
    graph.add_node("fetch_parent_chunks", nodes.fetch_parent_chunks)
    graph.add_node("maybe_compress_context", nodes.maybe_compress_context)
    graph.add_node("compress_context_node", nodes.compress_context_node)
    graph.add_node("mark_complete_without_compression", nodes.mark_complete_without_compression)

    graph.add_edge(START, "ingest_turn")
    graph.add_edge("ingest_turn", "classify_query_state")
    graph.add_edge("classify_query_state", "resolve_query_context")
    graph.add_edge("resolve_query_context", "classify_decomposition_need")
    graph.add_conditional_edges(
        "classify_decomposition_need",
        _route_after_decomposition_gate,
        {
            "maybe_build_decomposition_plan": "maybe_build_decomposition_plan",
            "rewrite_query_if_needed": "rewrite_query_if_needed",
        },
    )
    graph.add_edge("maybe_build_decomposition_plan", "validate_decomposition_plan")
    graph.add_edge("validate_decomposition_plan", "rewrite_query_if_needed")
    graph.add_edge("rewrite_query_if_needed", "extract_entities_if_needed")
    graph.add_edge("extract_entities_if_needed", "run_subquery_hybrid_search")
    graph.add_edge("run_subquery_hybrid_search", "run_hybrid_search")
    graph.add_edge("run_hybrid_search", "merge_retrieval_candidates")
    graph.add_edge("merge_retrieval_candidates", "rerank_results")
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
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
    config: RetrievalGraphConfig | None = None,
) -> RetrievalStageState:
    """Invoke retrieval graph and return state ready for downstream answer synthesis."""

    initial_state = default_retrieval_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        active_documents=active_documents,
        selected_documents=selected_documents,
    )
    app = build_retrieval_graph(dependencies=dependencies, config=config)
    return cast(RetrievalStageState, app.invoke(initial_state))
