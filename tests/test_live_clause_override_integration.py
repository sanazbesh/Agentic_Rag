from __future__ import annotations

from pathlib import Path
from typing import Any

import app
from agentic_rag.orchestration.legal_rag_graph import run_legal_rag_turn_with_state
from ui.local_backend import build_local_backend_dependencies


def _descriptor_for(tmp_path: Path) -> dict[str, str]:
    doc_path = tmp_path / "employment_agreement_019.md"
    doc_path.write_text(
        "# Employment Agreement\n\n## Confidentiality\nEmployee must keep confidential information private.\n",
        encoding="utf-8",
    )
    return {
        "id": "uploaded:employment_agreement_019.md",
        "name": "employment_agreement_019.md",
        "path": str(doc_path),
        "type": "md",
        "source": "uploaded",
    }


def test_live_path_classifier_promotes_document_grounded_what_is_clause_lookup(tmp_path: Path) -> None:
    descriptor = _descriptor_for(tmp_path)
    build = build_local_backend_dependencies([descriptor])

    _, state = run_legal_rag_turn_with_state(
        query="what is confidentiality?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )

    classification = state["query_classification"]
    assert classification is not None
    assert classification.question_type == "document_content_query"
    assert classification.answerability_expectation == "clause_lookup"
    assert classification.is_document_scoped is True


def test_selected_documents_from_real_backend_runner_are_plumbed_into_graph_call(tmp_path: Path, monkeypatch: Any) -> None:
    descriptor = _descriptor_for(tmp_path)
    build = build_local_backend_dependencies([descriptor])
    monkeypatch.setattr(app.st, "session_state", {"legal_rag_dependencies": build.dependencies}, raising=False)

    captured: dict[str, Any] = {}

    def _fake_run_legal_rag_turn_with_state(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        captured.update(kwargs)
        return (
            {"answer_text": "ok", "grounded": True, "sufficient_context": True, "citations": [], "warnings": []},
            {"query_classification": None},
        )

    monkeypatch.setattr(
        "agentic_rag.orchestration.legal_rag_graph.run_legal_rag_turn_with_state",
        _fake_run_legal_rag_turn_with_state,
    )

    real_backend_runner, _, error = app.build_real_backend_runners()
    assert error is None
    assert real_backend_runner is not None

    real_backend_runner(
        query="what is confidentiality?",
        conversation_summary=None,
        recent_messages=[],
        selected_documents=[descriptor],
    )

    assert captured["selected_documents"] == [descriptor]


def test_no_override_without_document_scope_regression(tmp_path: Path) -> None:
    descriptor = _descriptor_for(tmp_path)
    build = build_local_backend_dependencies([descriptor])

    _, state = run_legal_rag_turn_with_state(
        query="what is confidentiality?",
        dependencies=build.dependencies,
        selected_documents=[],
    )

    classification = state["query_classification"]
    assert classification is not None
    assert classification.question_type == "definition_query"
    assert classification.answerability_expectation == "definition_required"


def test_no_downstream_overwrite_after_clause_lookup_classification(tmp_path: Path) -> None:
    descriptor = _descriptor_for(tmp_path)
    build = build_local_backend_dependencies([descriptor])

    _, state = run_legal_rag_turn_with_state(
        query="what is confidentiality?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )

    classification = state["query_classification"]
    answerability = state["answerability_result"]
    assert classification is not None
    assert answerability is not None
    assert classification.answerability_expectation == "clause_lookup"
    assert answerability.answerability_expectation == "clause_lookup"
    assert answerability.question_type == "document_content_query"


def test_live_path_clause_lookup_classification_is_deterministic(tmp_path: Path) -> None:
    descriptor = _descriptor_for(tmp_path)
    build = build_local_backend_dependencies([descriptor])

    _, state_a = run_legal_rag_turn_with_state(
        query="what is confidentiality?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    _, state_b = run_legal_rag_turn_with_state(
        query="what is confidentiality?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )

    assert state_a["query_classification"] == state_b["query_classification"]
