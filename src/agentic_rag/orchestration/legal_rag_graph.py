"""Deterministic end-to-end legal RAG orchestration graph.

This module extends the retrieval-stage graph with explicit answer-stage nodes so
production behavior stays deterministic, debuggable, and non-agentic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.orchestration.retrieval_graph import (
    RetrievalDependencies,
    RetrievalGraphConfig,
    RetrievalStageState,
    build_retrieval_graph,
    default_retrieval_state,
)
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult, generate_answer
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


class LegalRagState(RetrievalStageState, total=False):
    """Retrieval + answer stage state.

    Answer-stage nodes write deterministic, typed fields without introducing any
    free-form agent routing.
    """

    answer_context: list[ParentChunkResult | CompressedParentChunk]
    final_answer: FinalAnswerModel | None
    final_response_ready: bool
    citations: list[AnswerCitation]
    sufficient_context: bool | None
    grounded: bool | None


class AnswerGenerator(Protocol):
    def __call__(self, context: Sequence[object], query: str) -> GenerateAnswerResult: ...


@dataclass(slots=True)
class LegalRagDependencies:
    """Dependencies needed to run retrieval and answer stages."""

    retrieval: RetrievalDependencies
    generate_grounded_answer: AnswerGenerator = generate_answer


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
) -> LegalRagState:
    """Build initial end-to-end graph state with explicit defaults."""

    retrieval_state = default_retrieval_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )
    merged = dict(retrieval_state)
    merged.update(
        {
            "answer_context": [],
            "final_answer": None,
            "final_response_ready": False,
            "citations": [],
            "sufficient_context": None,
            "grounded": None,
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


class AnswerStageNodes:
    """Explicit answer-stage node implementations.

    The graph remains deterministic: context source selection, insufficiency
    handling, and finalization are code-driven without free-form tool loops.
    """

    def __init__(self, answer_generator: AnswerGenerator) -> None:
        self.answer_generator = answer_generator

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

    def generate_grounded_answer(self, state: LegalRagState) -> LegalRagState:
        """Call `generate_answer(context, effective_query)` and persist structured output."""

        logger.info("node_enter name=generate_grounded_answer")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        answer_context = list(updated.get("answer_context", []))

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
            logger.info("node_exit name=generate_grounded_answer success=true empty_context=true")
            return cast(LegalRagState, updated)

        effective_query = str(updated.get("effective_query") or updated.get("original_query") or "").strip()
        try:
            raw = self.answer_generator(answer_context, effective_query)
            validated = _validate_answer_payload(raw)
            merged_warnings = [*warnings, *validated.warnings]
            final = _copy_update(validated, warnings=merged_warnings)
            updated["final_answer"] = final
            updated["citations"] = list(final.citations)
            updated["grounded"] = final.grounded
            updated["sufficient_context"] = final.sufficient_context
            updated["warnings"] = list(final.warnings)
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
            logger.info("node_exit name=generate_grounded_answer success=false reason=%s", type(exc).__name__)
            return cast(LegalRagState, updated)

    def finalize_response(self, state: LegalRagState) -> LegalRagState:
        """Validate/correct final answer so graph always exits with one typed output object."""

        logger.info("node_enter name=finalize_response")
        updated = dict(state)
        warnings = list(updated.get("warnings", []))
        raw_final = updated.get("final_answer")
        try:
            if raw_final is None:
                final = _safe_fallback(
                    warnings=[*warnings, "finalization_fallback:missing_final_answer"],
                    message=EMPTY_CONTEXT_MESSAGE,
                )
            else:
                final = _validate_answer_payload(raw_final)
                merged_warnings = [*warnings, *final.warnings]
                final = _copy_update(final, warnings=merged_warnings)
        except (ValueError, TypeError) as exc:
            final = _safe_fallback(
                warnings=[*warnings, f"finalization_fallback:{type(exc).__name__}"],
            )

        updated["final_answer"] = final
        updated["citations"] = list(final.citations)
        updated["grounded"] = final.grounded
        updated["sufficient_context"] = final.sufficient_context
        updated["warnings"] = list(final.warnings)
        updated["final_response_ready"] = True
        logger.info(
            "node_exit name=finalize_response ready=true citations=%s grounded=%s sufficient_context=%s",
            len(final.citations),
            final.grounded,
            final.sufficient_context,
        )
        return cast(LegalRagState, updated)


class _FallbackAnswerGraph:
    def __init__(self, nodes: AnswerStageNodes) -> None:
        self._nodes = nodes

    def invoke(self, state: LegalRagState) -> LegalRagState:
        current = state
        current = self._nodes.prepare_answer_context(current)
        current = self._nodes.generate_grounded_answer(current)
        current = self._nodes.finalize_response(current)
        return current


def build_answer_graph(*, answer_generator: AnswerGenerator = generate_answer) -> Any:
    """Build explicit answer-stage graph without autonomous agent loops."""

    nodes = AnswerStageNodes(answer_generator=answer_generator)

    if StateGraph is None:  # pragma: no cover
        return _FallbackAnswerGraph(nodes)

    graph = StateGraph(LegalRagState)
    graph.add_node("prepare_answer_context", nodes.prepare_answer_context)
    graph.add_node("generate_grounded_answer", nodes.generate_grounded_answer)
    graph.add_node("finalize_response", nodes.finalize_response)

    graph.add_edge(START, "prepare_answer_context")
    graph.add_edge("prepare_answer_context", "generate_grounded_answer")
    graph.add_edge("generate_grounded_answer", "finalize_response")
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
    answer_app = build_answer_graph(answer_generator=dependencies.generate_grounded_answer)
    return _ComposedLegalRagApp(retrieval_app=retrieval_app, answer_app=answer_app)


def run_legal_rag_turn(
    *,
    query: str,
    dependencies: LegalRagDependencies,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    retrieval_config: RetrievalGraphConfig | None = None,
) -> FinalAnswerModel:
    """Run one full legal RAG graph turn and return only the final typed answer.

    Internal graph state is intentionally hidden to keep the caller boundary
    stable and production-friendly.
    """

    initial = default_legal_rag_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )
    app = build_full_legal_rag_graph(dependencies=dependencies, retrieval_config=retrieval_config)
    final_state = cast(LegalRagState, app.invoke(initial))
    final_answer = final_state.get("final_answer")
    if not isinstance(final_answer, FinalAnswerModel):
        return _safe_fallback(warnings=["runner_fallback:missing_or_invalid_final_answer"], message=FAILURE_MESSAGE)
    return final_answer


def run_legal_rag_turn_with_state(
    *,
    query: str,
    dependencies: LegalRagDependencies,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Mapping[str, Any]] | None = None,
    retrieval_config: RetrievalGraphConfig | None = None,
) -> tuple[FinalAnswerModel, LegalRagState]:
    """Run one legal RAG turn and return both final answer and full state for debug/session memory."""

    initial = default_legal_rag_state(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )
    app = build_full_legal_rag_graph(dependencies=dependencies, retrieval_config=retrieval_config)
    final_state = cast(LegalRagState, app.invoke(initial))
    final_answer = final_state.get("final_answer")
    if not isinstance(final_answer, FinalAnswerModel):
        fallback = _safe_fallback(warnings=["runner_fallback:missing_or_invalid_final_answer"], message=FAILURE_MESSAGE)
        final_state["final_answer"] = fallback
        return fallback, final_state
    return final_answer, final_state
