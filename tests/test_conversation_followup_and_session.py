from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn_with_state
from agentic_rag.orchestration.retrieval_graph import RetrievalDependencies, run_retrieval_stage
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answer_generation import GenerateAnswerResult
from agentic_rag.tools.context_processing import CompressContextResult
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult
from ui.session_memory import append_conversation_turn, clear_conversation, serialize_history_as_messages


@dataclass
class MinimalDeps:
    def retrieval(self) -> RetrievalDependencies:
        return RetrievalDependencies(
            rewrite_query=self.rewrite_query,
            extract_legal_entities=self.extract_legal_entities,
            hybrid_search=self.hybrid_search,
            rerank_chunks=self.rerank_chunks,
            retrieve_parent_chunks=self.retrieve_parent_chunks,
            compress_context=self.compress_context,
        )

    def rewrite_query(self, query: str, **_: Any) -> QueryRewriteResult:
        return QueryRewriteResult(query, query, False)

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

    def hybrid_search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[HybridSearchResult]:
        _ = (query, filters, top_k)
        return [HybridSearchResult(child_chunk_id="c1", parent_chunk_id="p1", document_id="doc-nda", text="text", hybrid_score=0.9)]

    def rerank_chunks(self, chunks: list[HybridSearchResult], query: str) -> list[RerankedChunkResult]:
        _ = query
        return [
            RerankedChunkResult(
                child_chunk_id=chunks[0].child_chunk_id,
                parent_chunk_id=chunks[0].parent_chunk_id,
                document_id=chunks[0].document_id,
                text=chunks[0].text,
                rerank_score=0.9,
                original_score=0.9,
            )
        ]

    def retrieve_parent_chunks(self, parent_ids: list[str]) -> list[ParentChunkResult]:
        _ = parent_ids
        return [ParentChunkResult(parent_chunk_id="p1", document_id="doc-nda", text="confidentiality clause", source="s", source_name="s")]

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        return CompressContextResult(items=tuple(), total_original_chars=10, total_compressed_chars=10)


def test_simple_followup_reference_uses_prior_document_scope() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="What about governing law?",
        conversation_summary="Previous turn: employment agreement confidentiality.",
        recent_messages=[
            {"role": "assistant", "content": "A", "metadata": {"resolved_document_ids": ["doc-employment"], "resolved_topic_hints": ["confidentiality"]}}
        ],
        dependencies=deps,
    )
    assert state["context_resolution"] is not None
    assert state["context_resolution"].used_conversation_context is True
    assert state["context_resolution"].resolved_document_ids == ["doc-employment"]


def test_pronoun_resolution_uses_prior_nda_scope() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="What does it say about confidentiality?",
        conversation_summary="Summarized NDA in prior turn.",
        recent_messages=[{"role": "assistant", "content": "NDA summary", "metadata": {"resolved_document_ids": ["doc-nda"]}}],
        dependencies=deps,
    )
    assert state["context_resolution"] is not None
    assert "doc-nda" in state["context_resolution"].resolved_document_ids


def test_clause_followup_uses_prior_topic_hint() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="Does that clause still apply after termination?",
        conversation_summary="Discussed confidentiality clause.",
        recent_messages=[{"role": "assistant", "content": "A", "metadata": {"resolved_topic_hints": ["confidentiality clause"]}}],
        dependencies=deps,
    )
    assert state["context_resolution"] is not None
    assert state["context_resolution"].resolved_topic_hints == ["confidentiality clause"]


def test_self_contained_query_does_not_force_prior_context() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="Now tell me about arbitration in the MSA.",
        conversation_summary="Prior lease notice discussion.",
        recent_messages=[{"role": "assistant", "content": "A", "metadata": {"resolved_document_ids": ["doc-lease"]}}],
        dependencies=deps,
    )
    assert state["query_classification"] is not None
    assert state["query_classification"].use_conversation_context is False


def test_ambiguity_for_multiple_document_candidates_surfaces_warning() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="What does it say about governing law?",
        conversation_summary="Multiple contracts discussed.",
        recent_messages=[
            {
                "role": "assistant",
                "content": "A",
                "metadata": {"resolved_document_ids": ["doc-a", "doc-b"], "citations": [{"document_id": "doc-c"}]},
            },
        ],
        dependencies=deps,
    )
    assert any("ambiguous_multiple_candidates" in warning for warning in state["warnings"])


def test_missing_prior_scope_for_pronoun_is_safely_flagged() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="Does it mention termination?",
        conversation_summary=None,
        recent_messages=[],
        dependencies=deps,
    )
    assert any("missing_prior_scope" in warning for warning in state["warnings"])


def test_agreement_between_query_succeeds_with_selected_document_even_without_prior_scope() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="Is this agreement between Acme Corp and Jane Smith?",
        conversation_summary=None,
        recent_messages=[],
        selected_documents=[{"id": "doc-employment", "name": "Employment Agreement"}],
        dependencies=deps,
    )
    assert state["context_resolution"] is not None
    assert state["context_resolution"].resolved_document_ids == ["doc-employment"]
    assert not any("missing_prior_scope" in warning for warning in state["warnings"])


def test_session_continuity_helpers_preserve_turns_across_runs() -> None:
    history: list[dict[str, Any]] = []
    history = append_conversation_turn(history=history, query="Q1", answer_text="A1", metadata={"resolved_document_ids": ["doc-1"]})
    history = append_conversation_turn(history=history, query="Q2", answer_text="A2", metadata={"resolved_document_ids": ["doc-1"]})
    messages = serialize_history_as_messages(history)
    assert len(messages) == 4
    assert messages[-1]["role"] == "assistant"


def test_reset_behavior_helper_clears_conversation() -> None:
    assert clear_conversation() == []


def test_debug_visibility_includes_resolution_fields() -> None:
    deps = MinimalDeps()
    legal_deps = LegalRagDependencies(
        retrieval=deps.retrieval(),
        generate_grounded_answer=lambda context, query: GenerateAnswerResult(
            answer_text=f"answer for {query}",
            grounded=True,
            sufficient_context=True,
            citations=[],
            warnings=[],
        ),
    )
    _, state = run_legal_rag_turn_with_state(
        query="What about governing law?",
        dependencies=legal_deps,
        conversation_summary="Prior turn in employment agreement.",
        recent_messages=[{"role": "assistant", "content": "A", "metadata": {"resolved_document_ids": ["doc-employment"]}}],
    )
    assert state["query_classification"] is not None
    assert state["context_resolution"] is not None
    assert state["effective_query"]


def test_resolution_determinism_same_history_same_routing() -> None:
    deps = MinimalDeps().retrieval()
    inputs = dict(
        query="What does it say about confidentiality?",
        conversation_summary="Prior NDA discussion.",
        recent_messages=[{"role": "assistant", "content": "A", "metadata": {"resolved_document_ids": ["doc-nda"]}}],
        dependencies=deps,
    )
    state_1 = run_retrieval_stage(**inputs)
    state_2 = run_retrieval_stage(**inputs)
    assert state_1["query_classification"] == state_2["query_classification"]
    assert state_1["context_resolution"] == state_2["context_resolution"]


def test_resolved_document_filter_survives_entity_extraction_merge() -> None:
    deps = MinimalDeps().retrieval()
    state = run_retrieval_stage(
        query="What does it say about governing law in New York?",
        conversation_summary="Discussed NDA.",
        recent_messages=[{"role": "assistant", "content": "A", "metadata": {"resolved_document_ids": ["doc-nda"]}}],
        dependencies=deps,
    )
    assert state["filters"] is not None
    assert state["filters"].get("resolved_document_ids") == ["doc-nda"]
