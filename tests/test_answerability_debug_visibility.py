from __future__ import annotations

from typing import Any

from app import build_real_debug_payload
from agentic_rag.orchestration.legal_rag_graph import run_legal_rag_turn_with_state
from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.tools.answerability import AnswerabilityAssessment
from tests.test_legal_rag_answer_orchestration import FakeServices, _decision, _hybrid, _parent, _reranked
from ui.components import render_debug_panel


def _services_with_context(*, query: str) -> FakeServices:
    classifier = understand_query(query)
    return FakeServices(
        classifier=classifier,
        hybrid_results=[_hybrid("c1", "p1")],
        reranked_results=[_reranked("c1", "p1")],
        parent_results=[_parent("p1", text="Employment Agreement\nTermination Without Cause...")],
    )


def test_real_debug_payload_includes_answerability_result_from_execution_state() -> None:
    services = _services_with_context(query="what is employment agreement?")
    _, state = run_legal_rag_turn_with_state(
        query="what is employment agreement?",
        dependencies=services.as_dependencies(),
    )

    payload = build_real_debug_payload(latest_state=dict(state), selected_documents=[])

    assert "answerability_result" in payload
    assert isinstance(payload["answerability_result"], dict)
    assert payload["answerability_result"]["original_query"] == "what is employment agreement?"
    assert payload["answerability_result"]["sufficient_context"] is False


def test_answerability_result_serializes_without_losing_typed_fields() -> None:
    assessment = AnswerabilityAssessment(
        original_query="who are the parties?",
        question_type="extractive_fact_query",
        answerability_expectation="fact_extraction",
        has_relevant_context=True,
        sufficient_context=False,
        partially_supported=False,
        should_answer=False,
        support_level="weak",
        insufficiency_reason="fact_not_found",
        matched_parent_chunk_ids=["p1"],
        matched_headings=["Parties"],
        evidence_notes=["requested_fact_missing_from_context"],
        warnings=["w1"],
    )

    payload = build_real_debug_payload(
        latest_state={
            "answerability_assessment_invoked": True,
            "answerability_result": assessment,
            "warnings": [],
        },
        selected_documents=[],
    )

    assert payload["answerability_result"] == assessment.model_dump()


def test_missing_invocation_surfaces_explicit_debug_warning() -> None:
    payload = build_real_debug_payload(latest_state={}, selected_documents=[])
    assert payload["answerability_result"] is None
    assert any("assess_answerability not invoked" in warning for warning in payload["warnings"])


def test_regression_definition_query_debug_contains_answerability_result() -> None:
    services = _services_with_context(query="what is employment agreement?")
    _, state = run_legal_rag_turn_with_state(
        query="what is employment agreement?",
        dependencies=services.as_dependencies(),
    )

    payload = build_real_debug_payload(latest_state=dict(state), selected_documents=[])
    assert payload["answerability_result"] is not None
    assert payload["answerability_result"]["question_type"] == "definition_query"


class _NoopExpander:
    def __enter__(self) -> "_NoopExpander":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class _FakeStreamlit:
    def subheader(self, *_: Any, **__: Any) -> None:
        return None

    def expander(self, *_: Any, **__: Any) -> _NoopExpander:
        return _NoopExpander()

    def markdown(self, *_: Any, **__: Any) -> None:
        return None

    def json(self, *_: Any, **__: Any) -> None:
        return None

    def code(self, *_: Any, **__: Any) -> None:
        return None

    def write(self, *_: Any, **__: Any) -> None:
        return None


def test_streamlit_debug_panel_handles_answerability_result_section(monkeypatch: Any) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr("ui.components.st", fake_st)

    render_debug_panel(
        final_result={
            "answer_text": "A",
            "grounded": False,
            "sufficient_context": False,
            "citations": [],
            "warnings": [],
        },
        debug_payload={
            "query_classification": _decision().model_dump(),
            "context_resolution": {"resolved_document_ids": []},
            "answerability_result": AnswerabilityAssessment(
                original_query="q",
                question_type="definition_query",
                answerability_expectation="definition_required",
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=True,
                should_answer=False,
                support_level="weak",
                insufficiency_reason="definition_not_supported",
                matched_parent_chunk_ids=["p1"],
                matched_headings=["Employment Agreement"],
                evidence_notes=["title_only"],
                warnings=[],
            ).model_dump(),
            "warnings": [],
        },
    )
