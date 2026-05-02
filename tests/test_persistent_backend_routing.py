from __future__ import annotations

from types import SimpleNamespace

import app
from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies
from agentic_rag.orchestration.retrieval_graph import RetrievalDependencies


def test_real_backend_runner_uses_persistent_backend_for_persisted_selection(monkeypatch) -> None:
    selected = [{"id": "doc-1", "source": "persisted", "document_id": "doc-1", "document_version_id": "ver-1"}]

    called = {"persistent": 0, "local": 0}

    def fake_persistent(selected_docs):
        called["persistent"] += 1
        retrieval = RetrievalDependencies(
            rewrite_query=lambda query, **kwargs: query,
            extract_legal_entities=lambda query, **kwargs: {},
            hybrid_search=lambda *args, **kwargs: [],
            rerank_chunks=lambda chunks, **kwargs: chunks,
            retrieve_parent_chunks=lambda ids, **kwargs: [],
            compress_context=lambda parents, **kwargs: [],
            plan_decomposition=lambda query, **kwargs: None,
        )
        deps = LegalRagDependencies(retrieval=retrieval, generate_grounded_answer=lambda **kwargs: None)
        return deps, {"backend": "persistent_postgres_qdrant", "persisted_document_count": 1}, None

    def fake_local(selected_docs, local_llm_settings=None):
        called["local"] += 1
        return SimpleNamespace(dependencies=fake_persistent(selected_docs)[0], scope_meta={"backend": "local_in_memory"})

    def fake_run(**kwargs):
        return (
            {
                "answer_text": "ok",
                "grounded": True,
                "sufficient_context": True,
                "citations": [],
                "warnings": [],
            },
            {"warnings": []},
        )

    monkeypatch.setattr(app, "_build_persistent_backend_dependencies", fake_persistent)
    monkeypatch.setattr(app, "build_local_backend_dependencies", fake_local)
    import agentic_rag.orchestration.legal_rag_graph as legal_graph

    monkeypatch.setattr(legal_graph, "run_legal_rag_turn_with_state", fake_run)

    app.st.session_state = {}
    runner, _, err = app.build_real_backend_runners()
    assert err is None
    assert runner is not None
    result = runner(query="q", selected_documents=selected)

    assert result["answer_text"] == "ok"
    assert called["persistent"] == 1
    assert called["local"] == 0
    assert app.st.session_state["last_real_backend_scope_meta"]["backend"] == "persistent_postgres_qdrant"


def test_real_backend_runner_prefers_persisted_document_ids_for_filters(monkeypatch) -> None:
    selected = [
        {
            "id": "ui-doc-id",
            "path": "/tmp/doc.md",
            "source": "persisted",
            "document_id": "persisted-doc-123",
            "document_version_id": "ver-1",
        }
    ]

    seen_filters: dict[str, object] = {}

    def hybrid_search(_query, *, filters=None, top_k=10):
        seen_filters.update(filters or {})
        return []

    def fake_persistent(selected_docs):
        retrieval = RetrievalDependencies(
            rewrite_query=lambda query, **kwargs: query,
            extract_legal_entities=lambda query, **kwargs: {},
            hybrid_search=hybrid_search,
            rerank_chunks=lambda chunks, **kwargs: chunks,
            retrieve_parent_chunks=lambda ids, **kwargs: [],
            compress_context=lambda parents, **kwargs: [],
            plan_decomposition=lambda query, **kwargs: None,
        )
        deps = LegalRagDependencies(retrieval=retrieval, generate_grounded_answer=lambda **kwargs: None)
        return deps, {"backend": "persistent_postgres_qdrant", "persisted_document_count": 1}, None

    def fake_run(*, dependencies, **kwargs):
        dependencies.retrieval.hybrid_search("q", filters=None, top_k=5)
        return (
            {
                "answer_text": "ok",
                "grounded": True,
                "sufficient_context": True,
                "citations": [],
                "warnings": [],
            },
            {"warnings": []},
        )

    monkeypatch.setattr(app, "_build_persistent_backend_dependencies", fake_persistent)
    monkeypatch.setattr(app, "build_local_backend_dependencies", lambda *_args, **_kwargs: None)
    import agentic_rag.orchestration.legal_rag_graph as legal_graph

    monkeypatch.setattr(legal_graph, "run_legal_rag_turn_with_state", fake_run)

    app.st.session_state = {}
    runner, _, err = app.build_real_backend_runners()
    assert err is None
    assert runner is not None

    result = runner(query="q", selected_documents=selected)

    assert result["answer_text"] == "ok"
    assert seen_filters["selected_document_ids"] == ["persisted-doc-123"]
    assert seen_filters["selected_document_version_ids"] == ["ver-1"]
    assert app.st.session_state["last_real_backend_scope_meta"]["retrieval_source"] == "persistent"
