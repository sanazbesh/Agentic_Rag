from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn_with_state
from agentic_rag.orchestration.online_shadow_grading import OnlineShadowGrader, OnlineShadowGradingConfig
from agentic_rag.orchestration.retrieval_graph import QueryRoutingDecision, RetrievalDependencies
from agentic_rag.orchestration.traffic_sampling import TrafficSamplingConfig
from agentic_rag.retrieval.parent_child import HybridSearchResult, ParentChunkResult, RerankedChunkResult
from agentic_rag.tools.answer_generation import AnswerCitation, GenerateAnswerResult
from agentic_rag.tools.answerability import AnswerabilityAssessment
from agentic_rag.tools.context_processing import CompressContextResult
from agentic_rag.tools.query_intelligence import LegalEntityExtractionResult, LegalEntityFilters, QueryRewriteResult


@dataclass
class _Services:
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
            routing_notes=["legal_question_family:party_role_verification"],
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
        return [RerankedChunkResult(child_chunk_id="c1", parent_chunk_id="p1", document_id="doc-1", text="t", rerank_score=0.9, original_score=0.8)]

    def retrieve_parent_chunks(self, _: list[str]) -> list[ParentChunkResult]:
        return [ParentChunkResult(parent_chunk_id="p1", document_id="doc-1", text="parent", source="s", source_name="s", heading_text="h", metadata={"document_type": "contract"})]

    def compress_context(self, _: list[ParentChunkResult]) -> CompressContextResult:
        return CompressContextResult(items=(), total_original_chars=0, total_compressed_chars=0)

    def assess_answerability(self, query: str, query_understanding: Any, retrieved_context: list[object]) -> AnswerabilityAssessment:
        _ = (query, query_understanding, retrieved_context)
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

    def generate_answer(self, _: list[object], __: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text="answer",
            grounded=True,
            sufficient_context=True,
            citations=[AnswerCitation(parent_chunk_id="p1", document_id="doc-1", source_name="src", heading=None, supporting_excerpt="excerpt")],
            warnings=[],
        )


def _sample(trace_id: str = "tr-1") -> dict[str, Any]:
    return {
        "sample_id": "smp-1",
        "trace_id": trace_id,
        "request_id": "req-1",
        "query": "Who is the employer?",
        "family": "party_role_verification",
        "final_result": {
            "answer_text": "The employer is Acme.",
            "grounded": True,
            "sufficient_context": True,
            "citations": [{"parent_chunk_id": "p1", "document_id": "doc-1"}],
            "warnings": [],
        },
        "debug_payload": {
            "query_classification": {"routing_notes": ["legal_question_family:party_role_verification"]},
            "answerability_result": {"sufficient_context": True, "should_answer": True},
        },
        "version_identifiers": {"pipeline_version": "legal_rag.v0.20"},
    }


def test_shadow_grading_runs_deterministic_and_model_checks_and_links_trace(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow.jsonl"
    link_path = tmp_path / "trace_links.json"
    grader = OnlineShadowGrader(
        config=OnlineShadowGradingConfig(
            enabled=True,
            deterministic_evaluators=("contract_checks", "family_routing"),
            model_graders=("groundedness", "safe_failure"),
            output_path=str(output_path),
            trace_link_path=str(link_path),
        ),
        groundedness_judge_callable=lambda _: {"label": "grounded_answer", "confidence_band": "high", "short_reason": "supported", "supporting_notes": []},
        safe_failure_judge_callable=lambda _: {"label": "acceptable_insufficient_response", "confidence_band": "medium", "short_reason": "safe", "supporting_notes": []},
    )

    result = grader.grade_sample_sync(_sample())

    assert result["grading_status"] == "success"
    assert result["deterministic_results"]["contract_checks"]["status"] == "ok"
    assert result["deterministic_results"]["family_routing"]["status"] == "ok"
    assert result["model_results"]["groundedness"]["status"] == "ok"
    assert result["model_results"]["safe_failure"]["status"] == "ok"

    stored = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert stored["trace_id"] == "tr-1"
    links = json.loads(link_path.read_text(encoding="utf-8"))
    assert links["tr-1"]["latest_shadow_eval_id"] == stored["shadow_eval_id"]


def test_shadow_grading_handles_missing_and_failing_graders_without_crashing(tmp_path: Path) -> None:
    output_path = tmp_path / "shadow.jsonl"
    grader = OnlineShadowGrader(
        config=OnlineShadowGradingConfig(
            enabled=True,
            deterministic_evaluators=("missing_eval", "contract_checks", "failing_eval"),
            model_graders=("groundedness", "safe_failure", "missing_model"),
            output_path=str(output_path),
            trace_link_path=str(tmp_path / "trace_links.json"),
        ),
        deterministic_registry={
            "contract_checks": lambda _case, final_result, debug_payload: {"ok": bool(final_result), "debug": bool(debug_payload)},
            "failing_eval": lambda _case, _final_result, _debug_payload: (_ for _ in ()).throw(ValueError("boom")),
        },
    )

    malformed = grader.grade_sample_sync({"sample_id": "s-2"})
    valid = grader.grade_sample_sync(_sample(trace_id="tr-2"))

    assert malformed["grading_status"] == "failed"
    assert any(err["code"] == "missing_trace_id" for err in malformed["errors"])

    assert valid["deterministic_results"]["missing_eval"]["status"] == "unavailable"
    assert valid["deterministic_results"]["failing_eval"]["status"] == "error"
    assert valid["model_results"]["groundedness"]["status"] == "unavailable"
    assert valid["model_results"]["missing_model"]["status"] == "unavailable"
    assert valid["grading_status"] == "partial_failure"


def test_run_path_schedules_shadow_grading_without_waiting_for_completion(tmp_path: Path) -> None:
    started = threading.Event()
    unblock = threading.Event()

    def slow_eval(_case: dict[str, Any], _final_result: dict[str, Any], _debug_payload: dict[str, Any] | None) -> dict[str, Any]:
        started.set()
        unblock.wait(timeout=2)
        return {"status": "done"}

    grader = OnlineShadowGrader(
        config=OnlineShadowGradingConfig(
            enabled=True,
            deterministic_evaluators=("slow_eval",),
            model_graders=(),
            output_path=str(tmp_path / "shadow.jsonl"),
            trace_link_path=str(tmp_path / "trace_links.json"),
        ),
        deterministic_registry={"slow_eval": slow_eval},
    )

    t0 = time.perf_counter()
    answer, state = run_legal_rag_turn_with_state(
        query="Who is the employer?",
        dependencies=_Services().deps(),
        traffic_sampling_config=TrafficSamplingConfig(
            enabled=True,
            output_path=str(tmp_path / "samples.jsonl"),
            random_sample_rate=1.0,
            low_confidence_enabled=False,
        ),
        online_shadow_grader=grader,
    )
    elapsed = time.perf_counter() - t0

    assert answer.answer_text
    assert state.get("trace") is not None
    assert started.wait(timeout=1.0)
    assert elapsed < 0.5

    unblock.set()
    grader.wait_for_all(timeout=2)
    lines = (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
