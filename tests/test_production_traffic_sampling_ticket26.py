from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn_with_state
from agentic_rag.orchestration.retrieval_graph import QueryRoutingDecision, RetrievalDependencies
from agentic_rag.orchestration.traffic_sampling import (
    TrafficSamplingConfig,
    maybe_sample_production_traffic,
    traffic_sampling_config_from_mapping,
)
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult
from agentic_rag.tools.answerability import AnswerabilityAssessment
from agentic_rag.tools.context_processing import CompressContextResult
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult


@dataclass
class _Services:
    answerable: bool = True
    family: str = "party_role_verification"
    warnings: list[str] = field(default_factory=list)

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
            routing_notes=[f"legal_question_family:{self.family}"],
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
        return [HybridSearchResult(child_chunk_id="c1", parent_chunk_id="p1", document_id="doc-1", text="t", hybrid_score=0.8)]

    def rerank_chunks(self, _: list[HybridSearchResult], __: str) -> list[RerankedChunkResult]:
        return [
            RerankedChunkResult(
                child_chunk_id="c1",
                parent_chunk_id="p1",
                document_id="doc-1",
                text="t",
                rerank_score=0.9,
                original_score=0.8,
            )
        ]

    def retrieve_parent_chunks(self, _: list[str]) -> list[ParentChunkResult]:
        return [
            ParentChunkResult(
                parent_chunk_id="p1",
                document_id="doc-1",
                text="parent",
                source="s",
                source_name="s",
                heading_text="h",
                metadata={"document_type": "contract"},
            )
        ]

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        return CompressContextResult(items=(), total_original_chars=0, total_compressed_chars=0)

    def assess_answerability(self, query: str, query_understanding: Any, retrieved_context: list[object]) -> AnswerabilityAssessment:
        _ = (query, query_understanding, retrieved_context)
        return AnswerabilityAssessment(
            original_query="q",
            question_type="other_query",
            answerability_expectation="general_grounded_response",
            has_relevant_context=self.answerable,
            sufficient_context=self.answerable,
            partially_supported=not self.answerable,
            support_level="sufficient" if self.answerable else "none",
            insufficiency_reason=None if self.answerable else "fact_not_found",
            matched_parent_chunk_ids=["p1"] if self.answerable else [],
            matched_headings=["h"] if self.answerable else [],
            evidence_notes=[],
            warnings=list(self.warnings),
            should_answer=self.answerable,
        )

    def generate_answer(self, _: list[object], __: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text="answer",
            grounded=self.answerable,
            sufficient_context=self.answerable,
            citations=[
                AnswerCitation(
                    parent_chunk_id="p1",
                    document_id="doc-1",
                    source_name="src",
                    heading=None,
                    supporting_excerpt="excerpt",
                )
            ],
            warnings=[],
        )


def _base_state() -> dict[str, Any]:
    return {
        "query": "Who is the employer?",
        "trace": {
            "request_id": "req-1",
            "trace_id": "tr-1",
            "query": "Who is the employer?",
            "selected_document_ids": ["doc-1"],
            "active_family": "party_role_verification",
            "schema_version": "trace.v1",
            "pipeline_version": "legal_rag.v0.20",
        },
        "answerability_result": AnswerabilityAssessment(
            original_query="q",
            question_type="other_query",
            answerability_expectation="general_grounded_response",
            has_relevant_context=False,
            sufficient_context=False,
            partially_supported=True,
            support_level="none",
            insufficiency_reason="fact_not_found",
            matched_parent_chunk_ids=[],
            matched_headings=[],
            evidence_notes=[],
            warnings=["definition_not_supported"],
            should_answer=False,
        ),
        "metrics": {"cost_usd": 0.5, "total_tokens": 2400},
    }


def test_random_sampling_rate_is_configurable_and_deterministic(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(enabled=True, output_path=str(output), random_sample_rate=0.25, low_confidence_enabled=False)

    unsampled = maybe_sample_production_traffic(state=state, final_answer=GenerateAnswerResult(answer_text="a", grounded=False, sufficient_context=False, citations=[], warnings=[]), config=config, rng=lambda: 0.3)
    sampled = maybe_sample_production_traffic(state=state, final_answer=GenerateAnswerResult(answer_text="a", grounded=False, sufficient_context=False, citations=[], warnings=[]), config=config, rng=lambda: 0.2)

    assert unsampled is None
    assert sampled is not None
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_high_risk_family_sampling_uses_family_label(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(
        enabled=True,
        output_path=str(output),
        random_sample_rate=0.0,
        high_risk_family_sample_rates={"party_role_verification": 1.0},
        low_confidence_enabled=False,
    )
    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="a", grounded=True, sufficient_context=True, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.99,
    )
    assert record is not None
    assert "high_risk_family:party_role_verification" in record["sampling_reasons"]


def test_low_confidence_sampling_uses_structural_proxy_signals(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(enabled=True, output_path=str(output), random_sample_rate=0.0)
    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="a", grounded=False, sufficient_context=False, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.99,
    )
    assert record is not None
    reasons = set(record["sampling_reasons"])
    assert "low_confidence:insufficient_context" in reasons
    assert "low_confidence:partially_supported" in reasons
    assert "low_confidence:support_level:none" in reasons
    assert "low_confidence:warning:definition_not_supported" in reasons


def test_high_cost_sampling_uses_cost_or_token_thresholds(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(
        enabled=True,
        output_path=str(output),
        random_sample_rate=0.0,
        low_confidence_enabled=False,
        high_cost_cost_usd_threshold=0.4,
        high_cost_total_tokens_threshold=5000,
    )
    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="a", grounded=True, sufficient_context=True, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.99,
    )
    assert record is not None
    assert "high_cost:cost_usd>=0.4" in record["sampling_reasons"]


def test_multi_strategy_match_is_stored_once_with_multiple_reasons(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(
        enabled=True,
        output_path=str(output),
        random_sample_rate=1.0,
        high_risk_family_sample_rates={"party_role_verification": 1.0},
        high_cost_total_tokens_threshold=1000,
    )

    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="a", grounded=False, sufficient_context=False, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.01,
    )
    assert record is not None
    assert len(record["sampling_reasons"]) > 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_stored_sample_includes_trace_and_final_output(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(enabled=True, output_path=str(output), random_sample_rate=1.0, low_confidence_enabled=False)

    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="final", grounded=True, sufficient_context=True, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.0,
    )
    assert record is not None
    stored = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert stored["trace"]["trace_id"] == "tr-1"
    assert stored["final_result"]["answer_text"] == "final"


def test_sampling_strategy_can_be_loaded_from_mapping_config() -> None:
    config = traffic_sampling_config_from_mapping(
        {
            "random_sample_rate": 0.15,
            "high_risk_family_sample_rates": {"financial_entitlement": 0.9},
            "low_confidence_enabled": False,
            "high_cost_total_tokens_threshold": 1800,
            "output_path": "data/custom/samples.jsonl",
        }
    )
    assert config.random_sample_rate == 0.15
    assert config.high_risk_family_sample_rates["financial_entitlement"] == 0.9
    assert config.low_confidence_enabled is False
    assert config.high_cost_total_tokens_threshold == 1800
    assert config.output_path == "data/custom/samples.jsonl"


def test_malformed_or_partial_metadata_does_not_crash_sampling(tmp_path: Path) -> None:
    state = {"query": "q", "trace": "bad-trace", "answerability_result": "bad"}
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(enabled=True, output_path=str(output), random_sample_rate=1.0, low_confidence_enabled=False)
    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="a", grounded=False, sufficient_context=False, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.0,
    )
    assert record is not None
    assert record["query"] == "q"


def test_unsampled_requests_are_not_stored(tmp_path: Path) -> None:
    state = _base_state()
    output = tmp_path / "samples.jsonl"
    config = TrafficSamplingConfig(
        enabled=True,
        output_path=str(output),
        random_sample_rate=0.0,
        high_risk_family_sample_rates={},
        low_confidence_enabled=False,
        high_cost_cost_usd_threshold=10.0,
    )
    record = maybe_sample_production_traffic(
        state=state,
        final_answer=GenerateAnswerResult(answer_text="a", grounded=True, sufficient_context=True, citations=[], warnings=[]),
        config=config,
        rng=lambda: 0.99,
    )
    assert record is None
    assert not output.exists()


def test_sampling_hook_does_not_change_core_outputs(tmp_path: Path) -> None:
    services = _Services(answerable=True)
    baseline_answer, _ = run_legal_rag_turn_with_state(query="Who is the employer?", dependencies=services.deps())

    config = TrafficSamplingConfig(enabled=True, output_path=str(tmp_path / "samples.jsonl"), random_sample_rate=1.0, low_confidence_enabled=False)
    sampled_answer, _ = run_legal_rag_turn_with_state(
        query="Who is the employer?",
        dependencies=services.deps(),
        traffic_sampling_config=config,
    )

    assert sampled_answer.model_dump() == baseline_answer.model_dump()
