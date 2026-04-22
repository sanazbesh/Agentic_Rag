"""Deterministic end-to-end legal RAG orchestration graph.

This module extends the retrieval-stage graph with explicit answer-stage nodes so
production behavior stays deterministic, debuggable, and non-agentic.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.orchestration.retrieval_graph import (
    DecompositionPlan,
    RetrievalDependencies,
    RetrievalGraphConfig,
    RetrievalStageState,
    SubqueryCoverageRecord,
    build_retrieval_graph,
    default_retrieval_state,
)
from agentic_rag.orchestration.metrics import emit_request_metrics
from agentic_rag.orchestration.tracing import begin_span, create_trace, end_span, finalize_trace
from agentic_rag.orchestration.traffic_sampling import TrafficSamplingConfig, maybe_sample_production_traffic
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult, generate_answer
from agentic_rag.tools.answerability import AnswerabilityAssessment, assess_answerability
from agentic_rag.tools.context_processing import CompressedParentChunk

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when langgraph is installed
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - deterministic fallback used in tests
    START = "__start__"
    END = "__end__"
    StateGraph = None


class FinalAnswerModel(BaseModel):
    """Strict final graph output contract.

    The graph always returns this exact structure so caller code does not need to
    inspect internal graph state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_text: str
    grounded: bool
    sufficient_context: bool
    citations: list[AnswerCitation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SubqueryGroundedSubanswer(BaseModel):
    """Intermediate per-subquery grounded answer artifact for future synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subquery_id: str
    subquery_question: str
    answer_text: str
    grounded: bool
    support_classification: Literal["supported", "weak", "unsupported"]
    citations: list[AnswerCitation] = Field(default_factory=list)
    insufficiency_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)


class LegalRagState(RetrievalStageState, total=False):
    """Retrieval + answer stage state.

    Answer-stage nodes write deterministic, typed fields without introducing any
    free-form agent routing.
    """

    answer_context: list[ParentChunkResult | CompressedParentChunk]
    answerability_result: AnswerabilityAssessment | None
    final_answer: FinalAnswerModel | None
    final_response_ready: bool
    citations: list[AnswerCitation]
    sufficient_context: bool | None
    grounded: bool | None
    answerability_assessment: AnswerabilityAssessment | None
    answerability_assessment_invoked: bool
    should_generate_answer: bool
    should_return_partial_response: bool
    should_return_insufficient_response: bool
    response_route: str
    subquery_subanswers: list[SubqueryGroundedSubanswer]
    trace: dict[str, Any] | None
    metrics: dict[str, Any] | None


class AnswerGenerator(Protocol):
    def __call__(self, context: Sequence[object], query: str) -> GenerateAnswerResult: ...


class AnswerabilityEvaluator(Protocol):
    def __call__(self, query: str, query_understanding: object, retrieved_context: Sequence[object]) -> AnswerabilityAssessment: ...


@dataclass(slots=True)
class LegalRagDependencies:
    """Dependencies needed to run retrieval and answer stages."""

    retrieval: RetrievalDependencies
    generate_grounded_answer: AnswerGenerator = generate_answer
    assess_answerability: AnswerabilityEvaluator = assess_answerability


EMPTY_CONTEXT_MESSAGE = (
    "Direct answer: No relevant information was retrieved from the available context. "
    "The question cannot be answered reliably with the current retrieved material."
)
FAILURE_MESSAGE = (
    "Direct answer: The answer generation step failed safely. "
    "No grounded response could be produced from the provided context."
)


def default_legal_rag_state(
    *,
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
) -> LegalRagState:
    """Build initial end-to-end graph state with explicit defaults."""

    retrieval_state = default_retrieval_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        active_documents=active_documents,
        selected_documents=selected_documents,
    )
    selected_doc_ids: list[str] = []
    for item in selected_documents or []:
        if isinstance(item, Mapping):
            doc_id = item.get("id") or item.get("document_id")
            if isinstance(doc_id, str) and doc_id:
                selected_doc_ids.append(doc_id)
        elif hasattr(item, "id"):
            doc_id = getattr(item, "id")
            if isinstance(doc_id, str) and doc_id:
                selected_doc_ids.append(doc_id)
    merged = dict(retrieval_state)
    merged.update(
        {
            "answer_context": [],
            "final_answer": None,
            "final_response_ready": False,
            "citations": [],
            "sufficient_context": None,
            "grounded": None,
            "answerability_assessment": None,
            "answerability_assessment_invoked": False,
            "answerability_result": None,
            "should_generate_answer": False,
            "should_return_partial_response": False,
            "should_return_insufficient_response": True,
            "response_route": "unresolved",
            "subquery_subanswers": [],
            "trace": create_trace(query=query, selected_document_ids=selected_doc_ids),
            "metrics": None,
        }
    )
    return cast(LegalRagState, merged)


def _safe_fallback(*, warnings: Sequence[str], message: str = FAILURE_MESSAGE) -> FinalAnswerModel:
    return FinalAnswerModel(
        answer_text=message,
        grounded=False,
        sufficient_context=False,
        citations=[],
        warnings=list(warnings),
    )


def _copy_update(model: FinalAnswerModel, **updates: Any) -> FinalAnswerModel:
    payload = model.model_dump()
    payload.update(updates)
    return FinalAnswerModel(**payload)


def _copy_update_answerability(model: AnswerabilityAssessment, **updates: Any) -> AnswerabilityAssessment:
    payload = model.model_dump()
    payload.update(updates)
    return AnswerabilityAssessment(**payload)


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _extract_family_from_query_classification(query_classification: object) -> str | None:
    routing_notes = getattr(query_classification, "routing_notes", [])
    for note in routing_notes or []:
        if isinstance(note, str) and note.startswith("legal_question_family:"):
            family = note.split(":", 1)[1].strip()
            if family:
                return family
    return None


def _decomposition_coverage_gate_assessment(
    *,
    assessment: AnswerabilityAssessment,
    decomposition_plan: DecompositionPlan | None,
    subquery_coverage: Sequence[SubqueryCoverageRecord],
) -> AnswerabilityAssessment:
    """Apply required-subquery support gate before full sufficiency for decomposed queries."""

    if decomposition_plan is None or not decomposition_plan.subqueries:
        return assessment
    if not subquery_coverage:
        return assessment

    required_subquery_ids = [subquery.id for subquery in decomposition_plan.subqueries if subquery.required and subquery.id]
    if not required_subquery_ids:
        return assessment

    coverage_by_id = {item.subquery_id: item for item in subquery_coverage if item.subquery_id}
    below_threshold_required: list[str] = []
    weak_required: list[str] = []
    unsupported_required: list[str] = []
    supported_required_count = 0
    for required_subquery_id in required_subquery_ids:
        record = coverage_by_id.get(required_subquery_id)
        if record is None:
            below_threshold_required.append(required_subquery_id)
            unsupported_required.append(required_subquery_id)
            continue
        if record.support_classification == "supported":
            supported_required_count += 1
            continue
        below_threshold_required.append(required_subquery_id)
        if record.support_classification == "weak":
            weak_required.append(required_subquery_id)
        else:
            unsupported_required.append(required_subquery_id)

    if not below_threshold_required:
        return assessment

    if not assessment.sufficient_context:
        return _copy_update_answerability(
            assessment,
            evidence_notes=_dedupe_preserve_order(
                [
                    *list(assessment.evidence_notes),
                    f"decomposition_required_subqueries_supported:{supported_required_count}/{len(required_subquery_ids)}",
                ]
            ),
            warnings=_dedupe_preserve_order(
                [
                    *list(assessment.warnings),
                    f"decomposition_required_subquery_coverage_below_threshold:{','.join(below_threshold_required)}",
                ]
            ),
        )

    support_level: Literal["weak", "partial"] = "partial" if supported_required_count > 0 else "weak"
    partially_supported = support_level == "partial"
    insufficiency_reason = "partial_evidence_only" if partially_supported else "topic_match_but_not_answer"
    warning_suffix = ",".join(below_threshold_required)
    evidence_notes = [
        *list(assessment.evidence_notes),
        f"decomposition_required_subqueries_supported:{supported_required_count}/{len(required_subquery_ids)}",
    ]
    if weak_required:
        evidence_notes.append(f"decomposition_required_subqueries_weak:{','.join(weak_required)}")
    if unsupported_required:
        evidence_notes.append(f"decomposition_required_subqueries_unsupported:{','.join(unsupported_required)}")

    return _copy_update_answerability(
        assessment,
        sufficient_context=False,
        should_answer=False,
        partially_supported=partially_supported,
        support_level=support_level,
        insufficiency_reason=insufficiency_reason,
        evidence_notes=_dedupe_preserve_order(evidence_notes),
        warnings=_dedupe_preserve_order(
            [
                *list(assessment.warnings),
                f"decomposition_required_subquery_coverage_below_threshold:{warning_suffix}",
            ]
        ),
    )


def _extract_definition_subject(query: str) -> str | None:
    normalized = " ".join(str(query or "").split()).strip()
    if not normalized:
        return None
    match = re.match(r"^what\s+is(?:\s+the)?\s+(.+?)(?:\?)?$", normalized, flags=re.IGNORECASE)
    if not match:
        return None
    subject = match.group(1).strip(" \"'`")
    return subject or None


def _coerce_citation(item: object) -> AnswerCitation:
    if isinstance(item, AnswerCitation):
        return item
    parent_chunk_id = getattr(item, "parent_chunk_id", None)
    if isinstance(item, Mapping):
        parent_chunk_id = item.get("parent_chunk_id")
    if not parent_chunk_id:
        raise ValueError("citation missing parent_chunk_id")

    def _field(name: str) -> str | None:
        if isinstance(item, Mapping):
            value = item.get(name)
        else:
            value = getattr(item, name, None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return AnswerCitation(
        parent_chunk_id=str(parent_chunk_id),
        document_id=_field("document_id"),
        source_name=_field("source_name"),
        heading=_field("heading"),
        supporting_excerpt=_field("supporting_excerpt"),
    )


def _validate_answer_payload(payload: object) -> FinalAnswerModel:
    if isinstance(payload, FinalAnswerModel):
        answer = payload
    elif isinstance(payload, GenerateAnswerResult):
        answer = FinalAnswerModel(
            answer_text=payload.answer_text,
            grounded=payload.grounded,
            sufficient_context=payload.sufficient_context,
            citations=list(payload.citations),
            warnings=list(payload.warnings),
        )
    else:
        if isinstance(payload, Mapping):
            raw = payload
        else:
            raw = {
                "answer_text": getattr(payload, "answer_text", ""),
                "grounded": getattr(payload, "grounded", False),
                "sufficient_context": getattr(payload, "sufficient_context", False),
                "citations": getattr(payload, "citations", []),
                "warnings": getattr(payload, "warnings", []),
            }
        citations = [_coerce_citation(item) for item in list(raw.get("citations", []))]
        answer = FinalAnswerModel(
            answer_text=str(raw.get("answer_text", "")).strip(),
            grounded=bool(raw.get("grounded", False)),
            sufficient_context=bool(raw.get("sufficient_context", False)),
            citations=citations,
            warnings=[str(warning) for warning in list(raw.get("warnings", []))],
        )

    if not answer.citations and answer.grounded:
        answer = _copy_update(
            answer,
            grounded=False,
            warnings=[*answer.warnings, "grounding_adjusted:no_citations"],
        )
    return answer


def _dedupe_citations_preserve_order(citations: Sequence[AnswerCitation]) -> list[AnswerCitation]:
    seen: set[tuple[str, str | None, str | None, str | None, str | None]] = set()
    deduped: list[AnswerCitation] = []
    for citation in citations:
        key = (
            citation.parent_chunk_id,
            citation.document_id,
            citation.source_name,
            citation.heading,
            citation.supporting_excerpt,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def _synthesize_from_grounded_subanswers(
    *,
    decomposition_plan: DecompositionPlan | None,
    subquery_subanswers: Sequence[SubqueryGroundedSubanswer],
    baseline_warnings: Sequence[str],
    sufficient_context: bool,
) -> FinalAnswerModel | None:
    """Compose decomposed final answer strictly from grounded subanswers + explicit gaps."""

    if not isinstance(decomposition_plan, DecompositionPlan) or not decomposition_plan.subqueries:
        return None
    if not subquery_subanswers:
        return None

    by_id = {item.subquery_id: item for item in subquery_subanswers if item.subquery_id}
    supported_sections: list[str] = []
    aggregated_citations: list[AnswerCitation] = []
    required_gap_ids: list[str] = []
    gap_lines: list[str] = []
    subanswer_warnings: list[str] = []

    for subquery in decomposition_plan.subqueries:
        artifact = by_id.get(subquery.id)
        if artifact is None:
            if subquery.required:
                required_gap_ids.append(subquery.id)
                gap_lines.append(f"- Required gap: {subquery.question} (reason: no_subanswer_artifact)")
            else:
                gap_lines.append(f"- Optional gap: {subquery.question} (reason: no_subanswer_artifact)")
            continue

        subanswer_warnings.extend(list(artifact.warnings))
        is_supported = artifact.support_classification == "supported" and artifact.grounded and bool(artifact.citations)
        if is_supported:
            supported_sections.append(f"- {subquery.question}: {artifact.answer_text}")
            aggregated_citations.extend(list(artifact.citations))
            continue

        reason = artifact.insufficiency_reason or f"subquery_not_{artifact.support_classification}"
        if subquery.required:
            required_gap_ids.append(subquery.id)
            gap_lines.append(f"- Required gap: {subquery.question} (reason: {reason})")
        else:
            gap_lines.append(f"- Optional gap: {subquery.question} (reason: {reason})")

    aggregated_citations = _dedupe_citations_preserve_order(aggregated_citations)
    if not supported_sections:
        return None

    sections = ["Direct answer (synthesized from grounded subanswers):", *supported_sections]
    if gap_lines:
        sections.extend(["", "Support gaps:", *gap_lines])
    answer_text = "\n".join(sections)

    merged_warnings = _dedupe_preserve_order(
        [
            *list(baseline_warnings),
            *subanswer_warnings,
            *([f"decomposition_required_support_gaps:{','.join(required_gap_ids)}"] if required_gap_ids else []),
        ]
    )
    return FinalAnswerModel(
        answer_text=answer_text,
        grounded=bool(aggregated_citations),
        sufficient_context=bool(sufficient_context),
        citations=aggregated_citations,
        warnings=merged_warnings,
    )


class AnswerStageNodes:
    """Explicit answer-stage node implementations.

    The graph remains deterministic: context source selection, insufficiency
    handling, and finalization are code-driven without free-form tool loops.
    """

    def __init__(self, answer_generator: AnswerGenerator, answerability_evaluator: AnswerabilityEvaluator) -> None:
        self.answer_generator = answer_generator
        self.answerability_evaluator = answerability_evaluator

    def prepare_answer_context(self, state: LegalRagState) -> LegalRagState:
        """Select answer context deterministically: compressed context first, then parent chunks."""

        logger.info("node_enter name=prepare_answer_context")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        try:
            compressed_context = list(updated.get("compressed_context", []))
            parent_chunks = list(updated.get("parent_chunks", []))
            if compressed_context:
                answer_context: list[ParentChunkResult | CompressedParentChunk] = compressed_context
                source = "compressed_context"
            else:
                answer_context = parent_chunks
                source = "parent_chunks"
            updated["answer_context"] = answer_context
            logger.info(
                "node_exit name=prepare_answer_context source=%s context_count=%s",
                source,
                len(answer_context),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            fallback_context = list(updated.get("parent_chunks", []))
            updated["answer_context"] = fallback_context
            warnings.append(f"prepare_answer_context_failed:{type(exc).__name__}")
            logger.info(
                "node_exit name=prepare_answer_context source=fallback_parent_chunks context_count=%s",
                len(fallback_context),
            )
        updated["warnings"] = warnings
        return cast(LegalRagState, updated)


    def assess_answerability(self, state: LegalRagState) -> LegalRagState:
        """Hard gating node for answer generation.

        Why this gate exists:
        - relevance alone is not enough to answer safely,
        - definition queries must not pass on title-only support,
        - insufficient/partial evidence must route to a safe final response.
        """

        logger.info("node_enter name=assess_answerability")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        updated["answerability_assessment_invoked"] = True
        updated["response_route"] = "answerability_gate"
        query = str(updated.get("effective_query") or updated.get("original_query") or "").strip()
        query_understanding = updated.get("query_classification")
        context = list(updated.get("answer_context", []))
        trace = updated.get("trace")
        if isinstance(trace, dict):
            begin_span(
                trace,
                stage="answerability",
                span_name="Answerability",
                inputs_summary={"query": query, "context_count": len(context)},
            )

        if query_understanding is None:
            warnings.append("answerability_assessment_missing_query_understanding")
            updated["answerability_assessment"] = None
            updated["answerability_result"] = None
            updated["should_generate_answer"] = False
            updated["should_return_partial_response"] = False
            updated["should_return_insufficient_response"] = True
            updated["warnings"] = warnings
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answerability",
                    status="skipped",
                    outputs_summary={
                        "has_relevant_context": bool(context),
                        "sufficient_context": False,
                        "partially_supported": False,
                        "support_level": "none",
                        "insufficiency_reason": "missing_query_understanding",
                        "should_answer": False,
                    },
                    warnings=["answerability_assessment_missing_query_understanding"],
                )
            logger.info("node_exit name=assess_answerability skipped=true reason=missing_query_understanding")
            return cast(LegalRagState, updated)

        try:
            assessment = self.answerability_evaluator(query, query_understanding, context)
            decomposition_plan = updated.get("decomposition_plan")
            subquery_coverage = list(updated.get("subquery_coverage", []))
            assessment = _decomposition_coverage_gate_assessment(
                assessment=assessment,
                decomposition_plan=decomposition_plan if isinstance(decomposition_plan, DecompositionPlan) else None,
                subquery_coverage=[
                    item for item in subquery_coverage if isinstance(item, SubqueryCoverageRecord)
                ],
            )
            updated["answerability_assessment"] = assessment
            updated["answerability_result"] = assessment
            should_generate_answer = bool(assessment.sufficient_context and assessment.should_answer)
            should_return_partial_response = bool((not assessment.sufficient_context) and assessment.partially_supported)
            should_return_insufficient_response = not should_generate_answer
            updated["should_generate_answer"] = should_generate_answer
            updated["should_return_partial_response"] = should_return_partial_response
            updated["should_return_insufficient_response"] = should_return_insufficient_response
            updated["warnings"] = [*warnings, *assessment.warnings]
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answerability",
                    status="success",
                    outputs_summary={
                        "has_relevant_context": bool(assessment.has_relevant_context),
                        "sufficient_context": bool(assessment.sufficient_context),
                        "partially_supported": bool(assessment.partially_supported),
                        "support_level": str(assessment.support_level),
                        "insufficiency_reason": assessment.insufficiency_reason,
                        "should_answer": bool(assessment.should_answer),
                    },
                    warnings=list(assessment.warnings),
                )
            logger.info(
                "node_exit name=assess_answerability support_level=%s sufficient_context=%s partially_supported=%s should_answer=%s",
                assessment.support_level,
                assessment.sufficient_context,
                assessment.partially_supported,
                assessment.should_answer,
            )
        except Exception as exc:
            updated["answerability_assessment"] = None
            updated["answerability_result"] = None
            updated["should_generate_answer"] = False
            updated["should_return_partial_response"] = False
            updated["should_return_insufficient_response"] = True
            updated["warnings"] = [*warnings, f"answerability_assessment_failed:{type(exc).__name__}"]
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answerability",
                    status="failed",
                    outputs_summary={
                        "has_relevant_context": bool(context),
                        "sufficient_context": False,
                        "partially_supported": False,
                        "support_level": "none",
                        "insufficiency_reason": "answerability_assessment_failed",
                        "should_answer": False,
                    },
                    warnings=[f"answerability_assessment_failed:{type(exc).__name__}"],
                    error={"code": "answerability_assessment_failed", "message": str(exc)},
                )
            logger.info("node_exit name=assess_answerability success=false reason=%s", type(exc).__name__)
        return cast(LegalRagState, updated)

    def generate_subquery_subanswers(self, state: LegalRagState) -> LegalRagState:
        """Generate one intermediate grounded subanswer for each covered subquery."""

        logger.info("node_enter name=generate_subquery_subanswers")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        plan = updated.get("decomposition_plan")
        should_generate_answer = bool(updated.get("should_generate_answer", False))
        assessment = updated.get("answerability_result")
        decomposition_gap_override = False
        if isinstance(assessment, AnswerabilityAssessment):
            decomposition_gap_override = any(
                str(warning).startswith("decomposition_required_subquery_coverage_below_threshold:")
                for warning in assessment.warnings
            )
        allow_subanswer_generation = should_generate_answer or decomposition_gap_override
        coverage_records = [
            item for item in list(updated.get("subquery_coverage", [])) if isinstance(item, SubqueryCoverageRecord)
        ]
        answer_context = list(updated.get("answer_context", []))

        if not isinstance(plan, DecompositionPlan) or not plan.subqueries or not coverage_records:
            updated["subquery_subanswers"] = []
            logger.info("node_exit name=generate_subquery_subanswers skipped=true")
            return cast(LegalRagState, updated)

        subquery_by_id = {subquery.id: subquery for subquery in plan.subqueries if subquery.id}
        subanswers: list[SubqueryGroundedSubanswer] = []
        for record in coverage_records:
            subquery = subquery_by_id.get(record.subquery_id)
            subquery_question = subquery.question if subquery is not None else ""

            if record.support_classification != "supported":
                subanswers.append(
                    SubqueryGroundedSubanswer(
                        subquery_id=record.subquery_id,
                        subquery_question=subquery_question,
                        answer_text="No grounded subanswer generated due to insufficient subquery support.",
                        citations=[],
                        grounded=False,
                        support_classification=record.support_classification,
                        insufficiency_reason=record.insufficiency_reason,
                        warnings=[
                            f"subquery_not_answerable:{record.support_classification}",
                            *(
                                [f"subquery_insufficiency_reason:{record.insufficiency_reason}"]
                                if record.insufficiency_reason
                                else []
                            ),
                        ],
                    )
                )
                continue

            if not allow_subanswer_generation:
                subanswers.append(
                    SubqueryGroundedSubanswer(
                        subquery_id=record.subquery_id,
                        subquery_question=subquery_question,
                        answer_text="No grounded subanswer generated because answerability did not approve subquery synthesis.",
                        citations=[],
                        grounded=False,
                        support_classification="weak",
                        insufficiency_reason="answerability_gate_blocked_subquery_synthesis",
                        warnings=["subquery_generation_blocked_by_answerability_gate"],
                    )
                )
                continue

            allowed_parent_ids = {
                parent_chunk_id
                for parent_chunk_id in record.evidence_parent_chunk_ids
                if isinstance(parent_chunk_id, str) and parent_chunk_id
            }
            scoped_context = [
                item
                for item in answer_context
                if str(getattr(item, "parent_chunk_id", "") or (item.get("parent_chunk_id") if isinstance(item, Mapping) else ""))
                in allowed_parent_ids
            ]
            if not scoped_context:
                subanswers.append(
                    SubqueryGroundedSubanswer(
                        subquery_id=record.subquery_id,
                        subquery_question=subquery_question,
                        answer_text="No grounded subanswer generated because no subquery-linked evidence was available.",
                        citations=[],
                        grounded=False,
                        support_classification="unsupported",
                        insufficiency_reason="no_subquery_scoped_context",
                        warnings=["subquery_scoped_context_empty"],
                    )
                )
                continue

            try:
                raw = self.answer_generator(scoped_context, subquery_question)
                validated = _validate_answer_payload(raw)
                filtered_citations = [
                    citation for citation in validated.citations if citation.parent_chunk_id in allowed_parent_ids
                ]
                citation_scope_warnings = []
                if len(filtered_citations) != len(validated.citations):
                    citation_scope_warnings.append("subquery_citation_scope_adjusted")
                subanswers.append(
                    SubqueryGroundedSubanswer(
                        subquery_id=record.subquery_id,
                        subquery_question=subquery_question,
                        answer_text=validated.answer_text,
                        citations=filtered_citations,
                        grounded=bool(validated.grounded and filtered_citations),
                        support_classification="supported",
                        insufficiency_reason=None,
                        warnings=[*validated.warnings, *citation_scope_warnings],
                    )
                )
            except Exception as exc:
                subanswers.append(
                    SubqueryGroundedSubanswer(
                        subquery_id=record.subquery_id,
                        subquery_question=subquery_question,
                        answer_text="No grounded subanswer generated because generation failed safely.",
                        citations=[],
                        grounded=False,
                        support_classification="unsupported",
                        insufficiency_reason=f"generation_failed:{type(exc).__name__}",
                        warnings=[f"subquery_answer_generation_failed:{record.subquery_id}:{type(exc).__name__}"],
                    )
                )

        updated["subquery_subanswers"] = subanswers
        updated["warnings"] = warnings
        logger.info("node_exit name=generate_subquery_subanswers subanswer_count=%s", len(subanswers))
        return cast(LegalRagState, updated)

    def generate_grounded_answer(self, state: LegalRagState) -> LegalRagState:
        """Call `generate_answer(context, effective_query)` and persist structured output."""

        logger.info("node_enter name=generate_grounded_answer")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        answer_context = list(updated.get("answer_context", []))
        should_generate_answer = bool(updated.get("should_generate_answer", False))
        trace = updated.get("trace")
        if isinstance(trace, dict):
            begin_span(
                trace,
                stage="answer_generation",
                span_name="Answer Generation",
                inputs_summary={"context_count": len(answer_context), "should_generate_answer": should_generate_answer},
            )
        synthesized = _synthesize_from_grounded_subanswers(
            decomposition_plan=updated.get("decomposition_plan")
            if isinstance(updated.get("decomposition_plan"), DecompositionPlan)
            else None,
            subquery_subanswers=[
                item
                for item in list(updated.get("subquery_subanswers", []))
                if isinstance(item, SubqueryGroundedSubanswer)
            ],
            baseline_warnings=warnings,
            sufficient_context=True,
        )
        if synthesized is not None:
            updated["final_answer"] = synthesized
            updated["citations"] = list(synthesized.citations)
            updated["grounded"] = synthesized.grounded
            updated["sufficient_context"] = synthesized.sufficient_context
            updated["warnings"] = list(synthesized.warnings)
            updated["response_route"] = "generate_answer"
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answer_generation",
                    status="success",
                    outputs_summary={
                        "path_taken": "synthesized_from_subanswers",
                        "citation_count": len(synthesized.citations),
                        "status": "generated",
                    },
                    warnings=list(synthesized.warnings),
                )
            logger.info(
                "node_exit name=generate_grounded_answer success=true synthesized_from_subanswers=true citations=%s",
                len(synthesized.citations),
            )
            return cast(LegalRagState, updated)

        if not answer_context:
            fallback = FinalAnswerModel(
                answer_text=EMPTY_CONTEXT_MESSAGE,
                grounded=False,
                sufficient_context=False,
                citations=[],
                warnings=[*warnings, "insufficient_context:no_retrieved_context"],
            )
            updated["final_answer"] = fallback
            updated["citations"] = []
            updated["grounded"] = False
            updated["sufficient_context"] = False
            updated["warnings"] = list(fallback.warnings)
            updated["response_route"] = "generate_answer:empty_context"
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answer_generation",
                    status="partial",
                    outputs_summary={"path_taken": "empty_context", "citation_count": 0, "status": "skipped_safe"},
                    warnings=["insufficient_context:no_retrieved_context"],
                )
            logger.info("node_exit name=generate_grounded_answer success=true empty_context=true")
            return cast(LegalRagState, updated)

        if not should_generate_answer:
            fallback = _safe_fallback(
                warnings=[*warnings, "routing_violation:generate_without_answerability_approval"],
            )
            updated["final_answer"] = fallback
            updated["citations"] = []
            updated["grounded"] = False
            updated["sufficient_context"] = False
            updated["warnings"] = list(fallback.warnings)
            updated["response_route"] = "fallback_finalizer:routing_violation"
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answer_generation",
                    status="failed",
                    outputs_summary={"path_taken": "routing_violation", "citation_count": 0, "status": "failed_safe"},
                    warnings=["routing_violation:generate_without_answerability_approval"],
                )
            logger.info("node_exit name=generate_grounded_answer success=false reason=routing_violation")
            return cast(LegalRagState, updated)

        effective_query = str(updated.get("effective_query") or updated.get("original_query") or "").strip()
        try:
            raw = self.answer_generator(answer_context, effective_query)
            validated = _validate_answer_payload(raw)
            merged_warnings = [*warnings, *validated.warnings]
            final = _copy_update(validated, warnings=merged_warnings)
            if should_generate_answer and not final.sufficient_context:
                fallback = _safe_fallback(
                    warnings=[*merged_warnings, "fallback_after_sufficient_gate:generate_answer_returned_insufficient"],
                )
                updated["final_answer"] = fallback
                updated["citations"] = []
                updated["grounded"] = False
                updated["sufficient_context"] = False
                updated["warnings"] = list(fallback.warnings)
                updated["response_route"] = "fallback_finalizer:generate_answer_returned_insufficient"
                if isinstance(trace, dict):
                    end_span(
                        trace,
                        stage="answer_generation",
                        status="partial",
                        outputs_summary={
                            "path_taken": "generate_then_fallback_insufficient",
                            "citation_count": 0,
                            "status": "failed_safe",
                        },
                        warnings=["fallback_after_sufficient_gate:generate_answer_returned_insufficient"],
                    )
                logger.info("node_exit name=generate_grounded_answer success=false reason=inconsistent_sufficient_context")
                return cast(LegalRagState, updated)
            updated["final_answer"] = final
            updated["citations"] = list(final.citations)
            updated["grounded"] = final.grounded
            updated["sufficient_context"] = final.sufficient_context
            updated["warnings"] = list(final.warnings)
            updated["response_route"] = "generate_answer"
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answer_generation",
                    status="success",
                    outputs_summary={
                        "path_taken": "generate_answer",
                        "citation_count": len(final.citations),
                        "status": "generated",
                    },
                    warnings=list(final.warnings),
                )
            logger.info(
                "node_exit name=generate_grounded_answer success=true citations=%s grounded=%s sufficient_context=%s",
                len(final.citations),
                final.grounded,
                final.sufficient_context,
            )
            return cast(LegalRagState, updated)
        except Exception as exc:
            failure_warning = f"answer_generation_failed:{type(exc).__name__}"
            failure = _safe_fallback(warnings=[*warnings, failure_warning])
            updated["final_answer"] = failure
            updated["citations"] = []
            updated["grounded"] = False
            updated["sufficient_context"] = False
            updated["warnings"] = list(failure.warnings)
            updated["response_route"] = "fallback_finalizer:answer_generation_failed"
            if isinstance(trace, dict):
                end_span(
                    trace,
                    stage="answer_generation",
                    status="failed",
                    outputs_summary={"path_taken": "exception_fallback", "citation_count": 0, "status": "failed_safe"},
                    warnings=[failure_warning],
                    error={"code": "answer_generation_failed", "message": str(exc)},
                )
            logger.info("node_exit name=generate_grounded_answer success=false reason=%s", type(exc).__name__)
            return cast(LegalRagState, updated)

    def build_insufficient_response(self, state: LegalRagState) -> LegalRagState:
        """Build a strict final response when answerability blocks normal generation."""

        logger.info("node_enter name=build_insufficient_response")
        updated = dict(state)
        trace = updated.get("trace")
        if isinstance(trace, dict):
            begin_span(
                trace,
                stage="answer_generation",
                span_name="Answer Generation",
                inputs_summary={"context_count": len(list(updated.get("answer_context", []))), "should_generate_answer": False},
            )
            end_span(
                trace,
                stage="answer_generation",
                status="skipped",
                outputs_summary={"path_taken": "answerability_blocked", "citation_count": 0, "status": "skipped"},
                warnings=["answerability_gate_blocked_generation"],
            )
        warnings = list(updated.get("warnings", []))
        assessment = updated.get("answerability_result")
        synthesized = _synthesize_from_grounded_subanswers(
            decomposition_plan=updated.get("decomposition_plan")
            if isinstance(updated.get("decomposition_plan"), DecompositionPlan)
            else None,
            subquery_subanswers=[
                item
                for item in list(updated.get("subquery_subanswers", []))
                if isinstance(item, SubqueryGroundedSubanswer)
            ],
            baseline_warnings=warnings,
            sufficient_context=False,
        )
        if synthesized is not None:
            updated["final_answer"] = synthesized
            updated["citations"] = list(synthesized.citations)
            updated["grounded"] = synthesized.grounded
            updated["sufficient_context"] = synthesized.sufficient_context
            updated["warnings"] = list(synthesized.warnings)
            updated["response_route"] = "build_insufficient_response"
            logger.info(
                "node_exit name=build_insufficient_response route=synthesized_partial_or_insufficient citations=%s",
                len(synthesized.citations),
            )
            return cast(LegalRagState, updated)

        insufficiency_message = (
            "Direct answer: The retrieved context does not contain enough information to answer the question fully."
        )
        if isinstance(assessment, AnswerabilityAssessment):
            is_definition_intent = str(assessment.answerability_expectation) == "definition_required"
            if is_definition_intent and assessment.insufficiency_reason in {
                "definition_not_supported",
                "only_title_or_heading_match",
            }:
                asked_term = _extract_definition_subject(assessment.original_query)
                title_or_label_only = assessment.insufficiency_reason == "only_title_or_heading_match" or any(
                    note == "weakness_signal:title_only_signal_without_body" for note in assessment.evidence_notes
                )
                if asked_term and title_or_label_only:
                    insufficiency_message = (
                        f"Direct answer: I do not see a definition of '{asked_term}' in the retrieved context. "
                        "It appears as a document title or label, not as a defined term or clause."
                    )
                elif asked_term:
                    insufficiency_message = (
                        f"Direct answer: I do not see a definition of '{asked_term}' in the retrieved context."
                    )
                else:
                    insufficiency_message = (
                        "Direct answer: The retrieved context includes related material, but it does not define the term itself."
                    )
            elif assessment.insufficiency_reason == "fact_not_found":
                insufficiency_message = (
                    "Direct answer: The retrieved context does not contain enough information to identify the requested fact."
                )
            elif assessment.partially_supported:
                insufficiency_message = (
                    "Direct answer: The retrieved context only partially supports the question and is not sufficient for a full answer."
                )

            reason = assessment.insufficiency_reason or "other"
            warnings.append(f"answerability_gate:{reason}")
            warnings.extend(list(assessment.evidence_notes))

        final = FinalAnswerModel(
            answer_text=insufficiency_message,
            grounded=False,
            sufficient_context=False,
            citations=[],
            warnings=warnings,
        )
        updated["final_answer"] = final
        updated["citations"] = []
        updated["grounded"] = False
        updated["sufficient_context"] = False
        updated["warnings"] = list(final.warnings)
        updated["response_route"] = "build_insufficient_response"
        logger.info("node_exit name=build_insufficient_response route=insufficient_response")
        return cast(LegalRagState, updated)

    def finalize_response(self, state: LegalRagState) -> LegalRagState:
        """Validate/correct final answer so graph always exits with one typed output object."""

        logger.info("node_enter name=finalize_response")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        trace = updated.get("trace")
        if isinstance(trace, dict):
            begin_span(
                trace,
                stage="final_synthesis",
                span_name="Final Synthesis",
                inputs_summary={"response_route": updated.get("response_route", "unresolved")},
            )
        raw_final = updated.get("final_answer")
        try:
            if raw_final is None:
                final = _safe_fallback(
                    warnings=[*warnings, "finalization_fallback:missing_final_answer"],
                    message=EMPTY_CONTEXT_MESSAGE,
                )
                updated["response_route"] = "fallback_finalizer:missing_final_answer"
            else:
                final = _validate_answer_payload(raw_final)
                merged_warnings = _dedupe_preserve_order([*warnings, *final.warnings])
                final = _copy_update(final, warnings=merged_warnings)
        except (ValueError, TypeError) as exc:
            final = _safe_fallback(
                warnings=[*warnings, f"finalization_fallback:{type(exc).__name__}"],
            )
            updated["response_route"] = f"fallback_finalizer:{type(exc).__name__}"

        updated["final_answer"] = final
        updated["citations"] = list(final.citations)
        updated["grounded"] = final.grounded
        updated["sufficient_context"] = final.sufficient_context
        updated["warnings"] = list(final.warnings)
        updated["final_response_ready"] = True
        if isinstance(trace, dict):
            final_status = "answered" if final.sufficient_context else "insufficient_context"
            if not final.sufficient_context and final.citations:
                final_status = "partial_answer"
            end_span(
                trace,
                stage="final_synthesis",
                status="success",
                outputs_summary={
                    "final_answer_status": final_status,
                    "grounded": final.grounded,
                    "sufficient_context": final.sufficient_context,
                    "citation_count": len(final.citations),
                    "warning_count": len(final.warnings),
                    "synthesis_path": updated.get("response_route", "unresolved"),
                    "final_output_status": "success",
                },
                warnings=list(final.warnings),
            )
        logger.info(
            "node_exit name=finalize_response ready=true citations=%s grounded=%s sufficient_context=%s response_route=%s",
            len(final.citations),
            final.grounded,
            final.sufficient_context,
            updated.get("response_route", "unresolved"),
        )
        return cast(LegalRagState, updated)

    def route_after_answerability(self, state: LegalRagState) -> Literal["generate_grounded_answer", "build_insufficient_response"]:
        """Deterministic routing based on answerability gate output."""

        if bool(state.get("should_generate_answer", False)):
            logger.info("answerability_route selected=generate_answer")
            return "generate_grounded_answer"
        logger.info("answerability_route selected=insufficient_response")
        return "build_insufficient_response"


class _FallbackAnswerGraph:
    def __init__(self, nodes: AnswerStageNodes) -> None:
        self._nodes = nodes

    def invoke(self, state: LegalRagState) -> LegalRagState:
        current = state
        current = self._nodes.prepare_answer_context(current)
        current = self._nodes.assess_answerability(current)
        current = self._nodes.generate_subquery_subanswers(current)
        route = self._nodes.route_after_answerability(current)
        if route == "generate_grounded_answer":
            current = self._nodes.generate_grounded_answer(current)
        else:
            current = self._nodes.build_insufficient_response(current)
        current = self._nodes.finalize_response(current)
        return current


def build_answer_graph(
    *,
    answer_generator: AnswerGenerator = generate_answer,
    answerability_evaluator: AnswerabilityEvaluator = assess_answerability,
) -> Any:
    """Build explicit answer-stage graph without autonomous agent loops."""

    nodes = AnswerStageNodes(answer_generator=answer_generator, answerability_evaluator=answerability_evaluator)

    if StateGraph is None:  # pragma: no cover
        return _FallbackAnswerGraph(nodes)

    graph = StateGraph(LegalRagState)
    graph.add_node("prepare_answer_context", nodes.prepare_answer_context)
    graph.add_node("assess_answerability", nodes.assess_answerability)
    graph.add_node("generate_subquery_subanswers", nodes.generate_subquery_subanswers)
    graph.add_node("generate_grounded_answer", nodes.generate_grounded_answer)
    graph.add_node("build_insufficient_response", nodes.build_insufficient_response)
    graph.add_node("finalize_response", nodes.finalize_response)

    graph.add_edge(START, "prepare_answer_context")
    graph.add_edge("prepare_answer_context", "assess_answerability")
    graph.add_edge("assess_answerability", "generate_subquery_subanswers")
    graph.add_conditional_edges(
        "generate_subquery_subanswers",
        nodes.route_after_answerability,
        {
            "generate_grounded_answer": "generate_grounded_answer",
            "build_insufficient_response": "build_insufficient_response",
        },
    )
    graph.add_edge("generate_grounded_answer", "finalize_response")
    graph.add_edge("build_insufficient_response", "finalize_response")
    graph.add_edge("finalize_response", END)
    return graph.compile()


class _ComposedLegalRagApp:
    """Simple deterministic composition of retrieval graph then answer graph."""

    def __init__(self, retrieval_app: Any, answer_app: Any) -> None:
        self._retrieval_app = retrieval_app
        self._answer_app = answer_app

    def invoke(self, state: LegalRagState) -> LegalRagState:
        retrieval_state = cast(RetrievalStageState, self._retrieval_app.invoke(cast(RetrievalStageState, state)))
        merged = dict(state)
        merged.update(dict(retrieval_state))
        return cast(LegalRagState, self._answer_app.invoke(cast(LegalRagState, merged)))


def build_full_legal_rag_graph(
    dependencies: LegalRagDependencies,
    *,
    retrieval_config: RetrievalGraphConfig | None = None,
) -> Any:
    """Build end-to-end deterministic graph composition for legal RAG turns."""

    retrieval_app = build_retrieval_graph(dependencies.retrieval, config=retrieval_config)
    answer_app = build_answer_graph(
        answer_generator=dependencies.generate_grounded_answer,
        answerability_evaluator=dependencies.assess_answerability,
    )
    return _ComposedLegalRagApp(retrieval_app=retrieval_app, answer_app=answer_app)


def run_legal_rag_turn(
    *,
    query: str,
    dependencies: LegalRagDependencies,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
    retrieval_config: RetrievalGraphConfig | None = None,
    traffic_sampling_config: TrafficSamplingConfig | Mapping[str, Any] | None = None,
) -> FinalAnswerModel:
    """Run one full legal RAG graph turn and return only the final typed answer.

    Internal graph state is intentionally hidden to keep the caller boundary
    stable and production-friendly.
    """

    initial = default_legal_rag_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        active_documents=active_documents,
        selected_documents=selected_documents,
    )
    app = build_full_legal_rag_graph(dependencies=dependencies, retrieval_config=retrieval_config)
    final_state = cast(LegalRagState, app.invoke(initial))
    trace = final_state.get("trace")
    if isinstance(trace, dict):
        active_family = _extract_family_from_query_classification(final_state.get("query_classification"))
        final_state["trace"] = finalize_trace(trace, active_family=active_family)
    final_answer = final_state.get("final_answer")
    if not isinstance(final_answer, FinalAnswerModel):
        return _safe_fallback(warnings=["runner_fallback:missing_or_invalid_final_answer"], message=FAILURE_MESSAGE)
    try:
        final_state["metrics"] = emit_request_metrics(final_answer=final_answer, state=final_state).model_dump()
    except Exception:  # pragma: no cover - metrics must never break core answer path
        final_state["metrics"] = None
    try:
        maybe_sample_production_traffic(state=final_state, final_answer=final_answer, config=traffic_sampling_config)
    except Exception:  # pragma: no cover - sampling must never break core answer path
        pass
    return final_answer


def run_legal_rag_turn_with_state(
    *,
    query: str,
    dependencies: LegalRagDependencies,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    active_documents: Sequence[Any] | None = None,
    selected_documents: Sequence[Any] | None = None,
    retrieval_config: RetrievalGraphConfig | None = None,
    traffic_sampling_config: TrafficSamplingConfig | Mapping[str, Any] | None = None,
) -> tuple[FinalAnswerModel, LegalRagState]:
    """Run one legal RAG turn and return both final answer and full state for debug/session memory."""

    initial = default_legal_rag_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        active_documents=active_documents,
        selected_documents=selected_documents,
    )
    app = build_full_legal_rag_graph(dependencies=dependencies, retrieval_config=retrieval_config)
    final_state = cast(LegalRagState, app.invoke(initial))
    trace = final_state.get("trace")
    if isinstance(trace, dict):
        active_family = _extract_family_from_query_classification(final_state.get("query_classification"))
        final_state["trace"] = finalize_trace(trace, active_family=active_family)
    final_answer = final_state.get("final_answer")
    if not isinstance(final_answer, FinalAnswerModel):
        fallback = _safe_fallback(warnings=["runner_fallback:missing_or_invalid_final_answer"], message=FAILURE_MESSAGE)
        final_state["final_answer"] = fallback
        try:
            final_state["metrics"] = emit_request_metrics(final_answer=fallback, state=final_state).model_dump()
        except Exception:  # pragma: no cover - metrics must never break core answer path
            final_state["metrics"] = None
        try:
            maybe_sample_production_traffic(state=final_state, final_answer=fallback, config=traffic_sampling_config)
        except Exception:  # pragma: no cover - sampling must never break core answer path
            pass
        return fallback, final_state
    try:
        final_state["metrics"] = emit_request_metrics(final_answer=final_answer, state=final_state).model_dump()
    except Exception:  # pragma: no cover - metrics must never break core answer path
        final_state["metrics"] = None
    try:
        maybe_sample_production_traffic(state=final_state, final_answer=final_answer, config=traffic_sampling_config)
    except Exception:  # pragma: no cover - sampling must never break core answer path
        pass
    return final_answer, final_state
