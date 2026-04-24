from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_rag.orchestration.legal_rag_graph import run_legal_rag_turn, run_legal_rag_turn_with_state
from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult
from ui.debug_payload import build_real_debug_payload
from ui.local_backend import LocalLLMRuntimeSettings, LocalLLMStageToggles, effective_local_llm_settings
from ui.local_backend import build_local_backend_dependencies


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _StubPromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        prompt_text = str(prompt)
        if "Return strict JSON" in prompt_text and "\"rewritten_query\"" in prompt_text:
            return '{"rewritten_query":"How is Section 9 enforced?"}'
        if "\"strategy\"" in prompt_text and "\"subqueries\"" in prompt_text:
            return (
                '{"strategy":"comparison","subqueries":['
                '{"id":"sq-1","question":"Find indemnity clause.","purpose":"Indemnity evidence.","required":true,"expected_answer_type":"cross_reference","dependency_ids":[]}'
                "]}"
            )
        return "Grounded draft from evidence."


class _FailingRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise RuntimeError("rewrite_boom")


class _TimeoutRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        raise TimeoutError("rewrite timeout")


class _MalformedRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return "not-json"


class _FencedJsonRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return '```json\n{"rewritten_query":"who is the hiring company in the agreement?"}\n```'


class _ValidationFailedRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return '{"query":"missing required key"}'


class _EmptyRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return '{"rewritten_query":"   "}'


class _NoOpRewritePromptClient:
    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if "Return strict JSON" in str(prompt) and "\"rewritten_query\"" in str(prompt):
            return '{"rewritten_query":"who is the hiring company here?"}'
        return '{"strategy":"single","subqueries":[]}'


def _forced_rewrite_decision() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        original_query="who is the hiring company here?",
        normalized_query="who is the hiring company here?",
        question_type="ambiguous_query",
        is_followup=False,
        is_context_dependent=False,
        use_conversation_context=False,
        is_document_scoped=False,
        refers_to_prior_document_scope=False,
        refers_to_prior_clause_or_topic=False,
        should_rewrite=True,
        should_extract_entities=False,
        should_retrieve=True,
        may_need_decomposition=False,
        answerability_expectation="fact_extraction",
        resolved_document_hints=[],
        resolved_topic_hints=[],
        resolved_clause_hints=[],
        ambiguity_notes=[],
        routing_notes=["rewrite_recommended"],
        warnings=[],
    )


def test_local_backend_builds_from_uploaded_markdown(tmp_path: Path) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(
        doc_path,
        """# Service Agreement\n\n## Termination\nEither party may terminate with 30 days notice.\n""",
    )

    descriptor = {
        "id": "uploaded:msa.md",
        "name": "msa.md",
        "path": str(doc_path),
        "type": "md",
        "source": "uploaded",
    }

    build = build_local_backend_dependencies([descriptor])
    result = run_legal_rag_turn(
        query="termination notice",
        dependencies=build.dependencies,
    )

    assert build.scope_meta["loaded_document_count"] == 1
    assert build.scope_meta["parent_chunk_count"] >= 1
    assert result.answer_text


def test_local_backend_hybrid_filter_supports_selected_document_ids(tmp_path: Path) -> None:
    doc_a = tmp_path / "a.md"
    doc_b = tmp_path / "b.md"
    _write_text_file(doc_a, "# A\n\n## Rule\nContract A says notice is ten days.\n")
    _write_text_file(doc_b, "# B\n\n## Rule\nContract B says notice is ninety days.\n")

    descriptor_a = {
        "id": "uploaded:a.md",
        "name": "a.md",
        "path": str(doc_a),
        "type": "md",
        "source": "uploaded",
    }
    descriptor_b = {
        "id": "uploaded:b.md",
        "name": "b.md",
        "path": str(doc_b),
        "type": "md",
        "source": "uploaded",
    }

    build = build_local_backend_dependencies([descriptor_a, descriptor_b])
    hits = build.dependencies.retrieval.hybrid_search(
        "notice days",
        filters={"selected_document_ids": ["uploaded:a.md"]},
        top_k=10,
    )

    assert hits
    assert all(hit.document_id == "uploaded:a.md" for hit in hits)


def test_effective_local_llm_settings_respects_mock_mode() -> None:
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.1,
        timeout_seconds=9.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=True,
    )

    assert settings.ui_enabled is True
    assert settings.enabled is False
    assert settings.mock_backend_active is True


def test_local_llm_stage_toggles_only_apply_to_eligible_stages(tmp_path: Path) -> None:
    doc_path = tmp_path / "compare.md"
    _write_text_file(
        doc_path,
        "# Agreement\n\n## Indemnity\nIndemnity applies to third-party claims.\n\n## Liability\nLiability is capped.\n",
    )
    descriptor = {"id": "uploaded:compare.md", "name": "compare.md", "path": str(doc_path), "type": "md", "source": "uploaded"}

    settings = LocalLLMRuntimeSettings(
        ui_enabled=True,
        enabled=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        stages=LocalLLMStageToggles(rewrite=False, decomposition=False, synthesis=False),
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    _, state = run_legal_rag_turn_with_state(
        query="Compare indemnity and liability clauses.",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )

    warnings = [str(item) for item in state.get("warnings", [])]
    planner_notes = [str(item) for item in list(getattr(state.get("decomposition_plan"), "planner_notes", []))]
    final_warnings = [str(item) for item in list(getattr(state.get("final_answer"), "warnings", []))]
    assert not any(item.startswith("rewrite_path:llm:") for item in warnings)
    assert not any(item.startswith("planner_path:llm:") for item in planner_notes)
    assert not any(item.startswith("answer_synthesis_path:llm:") for item in final_warnings)


def test_llama_cpp_unavailability_falls_back_without_crash(tmp_path: Path, monkeypatch) -> None:
    from agentic_rag.llm.local_provider import LlamaCppChatProvider

    doc_path = tmp_path / "msa.md"
    _write_text_file(
        doc_path,
        "# Agreement\n\n## Termination\nEither party may terminate for material breach if the breach remains uncured for 30 days after written notice.\n",
    )
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}

    def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("llama_cpp_down")

    monkeypatch.setattr(LlamaCppChatProvider, "chat", _boom)
    monkeypatch.setattr("pathlib.Path.is_file", lambda _: True)
    settings = LocalLLMRuntimeSettings(
        ui_enabled=True,
        enabled=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        stages=LocalLLMStageToggles(rewrite=True, decomposition=True, synthesis=True),
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    _, state = run_legal_rag_turn_with_state(
        query="When can a party terminate for breach?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )

    result = state["final_answer"]
    assert result is not None
    assert result.answer_text


def test_debug_payload_includes_local_llm_runtime_metadata(tmp_path: Path) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Term\nTerm is one year.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}

    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    _, state = run_legal_rag_turn_with_state(
        query="What is the term?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    payload = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)
    runtime = payload["local_llm_runtime"]
    assert runtime["provider"] == "llama_cpp"
    assert runtime["model_path"] == "/models/llama.gguf"
    assert runtime["stage_toggles"]["rewrite"] is True


def test_invalid_model_path_uses_deterministic_fallback(tmp_path: Path) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Term\nTerm is one year.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}

    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/does/not/exist/model.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    _, state = run_legal_rag_turn_with_state(
        query="What is the term?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )

    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["effective_enabled"] is True
    assert runtime["stages_using_local_llm"] == []
    assert runtime["provider_init_reason"] == "invalid_model_path"
    assert runtime["per_stage_fallback_reason"]["synthesis"] in {"invalid_model_path", "upstream_blocked"}


def test_runtime_metadata_marks_rewrite_as_local_llm_used(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Term\nTerm is one year.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}

    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _StubPromptClient(),
            {
                "local_llm_attempted": True,
                "provider_init_status": "ready",
                "provider_init_error": None,
                "provider_init_reason": None,
            },
        ),
    )
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)

    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    _, state = run_legal_rag_turn_with_state(
        query="How is that clause enforced?",
        conversation_summary="We discussed Section 9.",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["local_llm_used"] is True
    assert runtime["effective_mode"] == "llama_cpp_assisted"
    assert "rewrite" in runtime["stages_using_local_llm"]




def test_stage_toggle_disabled_and_upstream_blocked_reasons_are_explicit(tmp_path: Path) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Term\nTerm is one year.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}

    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/does/not/exist/model.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=False,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    _, state = run_legal_rag_turn_with_state(
        query="What is severability under this contract?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["per_stage_fallback_reason"]["rewrite"] == "disabled_by_toggle"
    assert runtime["per_stage_fallback_reason"]["synthesis"] in {"upstream_blocked", "invalid_model_path"}


def test_rewrite_requested_and_ready_attempts_local_llm_without_stage_not_applicable(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _StubPromptClient(),
            {
                "local_llm_attempted": True,
                "provider_init_status": "ready",
                "provider_init_error": None,
                "provider_init_reason": None,
            },
        ),
    )
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)

    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["rewrite_requested"] is True
    assert runtime["rewrite_applicable"] is True
    assert runtime["rewrite_attempted"] is True
    assert runtime["rewrite_used_local_llm"] is True
    assert runtime["per_stage_local_llm_status"]["rewrite"] == "used"
    assert runtime["per_stage_fallback_reason"].get("rewrite") != "stage_not_applicable"
    assert "rewrite" in runtime["stages_attempted_local_llm"]
    assert "rewrite" in runtime["stages_using_local_llm"]
    assert runtime["local_llm_used"] is True


def test_rewrite_failure_reports_explicit_reason(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _FailingRewritePromptClient(),
            {
                "local_llm_attempted": True,
                "provider_init_status": "ready",
                "provider_init_error": None,
                "provider_init_reason": None,
            },
        ),
    )
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["rewrite_attempted"] is True
    assert runtime["per_stage_local_llm_status"]["rewrite"] == "fallback"
    assert runtime["per_stage_fallback_reason"]["rewrite"] == "provider_runtime_error"
    assert runtime["rewrite_fallback_reason"] == "provider_runtime_error"
    assert runtime["rewrite_result_type"] == "failed"
    assert runtime["rewrite_provider_error"] is not None


def test_rewrite_invocation_exception_surfaces_specific_failure_diagnostics(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _StubPromptClient(),
            {
                "local_llm_attempted": True,
                "provider_init_status": "ready",
                "provider_init_error": None,
                "provider_init_reason": None,
            },
        ),
    )
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()

    def _raise_from_rewrite(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("provider_call_failed")

    build.dependencies.retrieval.rewrite_query = _raise_from_rewrite
    _, state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]

    assert runtime["rewrite_attempted"] is True
    assert runtime["rewrite_used_local_llm"] is False
    assert runtime["local_llm_used"] is False
    assert runtime["rewrite_result_type"] == "failed"
    assert runtime["rewrite_fallback_reason"] == "provider_runtime_error"
    assert runtime["rewrite_failure_stage"] == "provider_call"
    assert runtime["per_stage_local_llm_status"]["rewrite"] == "fallback"
    assert runtime["per_stage_fallback_reason"]["rewrite"] == "provider_runtime_error"
    assert runtime["rewrite_provider_error"] == "RuntimeError:provider_call_failed"
    assert runtime["provider_call_attempted"] is True
    assert runtime["provider_call_succeeded"] is False
    assert runtime["raw_response_present"] is False
    assert runtime["normalized_rewrite_present"] is False
    assert "rewrite" not in runtime["stages_using_local_llm"]


def test_rewrite_failure_reasons_distinguish_timeout_and_malformed(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)

    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )

    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _TimeoutRewritePromptClient(),
            {"local_llm_attempted": True, "provider_init_status": "ready", "provider_init_error": None, "provider_init_reason": None},
        ),
    )
    timeout_build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    timeout_build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, timeout_state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=timeout_build.dependencies,
        selected_documents=[descriptor],
    )
    timeout_runtime = build_real_debug_payload(
        latest_state=timeout_state, selected_documents=[descriptor], scope_meta=timeout_build.scope_meta
    )["local_llm_runtime"]
    assert timeout_runtime["rewrite_fallback_reason"] == "timeout"
    assert timeout_runtime["rewrite_failure_stage"] == "provider_call"

    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _MalformedRewritePromptClient(),
            {"local_llm_attempted": True, "provider_init_status": "ready", "provider_init_error": None, "provider_init_reason": None},
        ),
    )
    malformed_build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    malformed_build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, malformed_state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=malformed_build.dependencies,
        selected_documents=[descriptor],
    )
    malformed_runtime = build_real_debug_payload(
        latest_state=malformed_state, selected_documents=[descriptor], scope_meta=malformed_build.scope_meta
    )["local_llm_runtime"]
    assert malformed_runtime["rewrite_fallback_reason"] == "malformed_response"
    assert malformed_runtime["rewrite_failure_stage"] == "response_parse"


def test_rewrite_noop_is_not_reported_as_fallback(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _NoOpRewritePromptClient(),
            {"local_llm_attempted": True, "provider_init_status": "ready", "provider_init_error": None, "provider_init_reason": None},
        ),
    )
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["rewrite_used_local_llm"] is True
    assert runtime["rewrite_result_type"] == "no_change"
    assert runtime["rewrite_failure_stage"] is None
    assert runtime["per_stage_local_llm_status"]["rewrite"] == "used"
    assert runtime["provider_call_succeeded"] is True
    assert runtime["normalized_rewrite_present"] is True
    assert "rewrite" in runtime["stages_using_local_llm"]


def test_rewrite_accepts_fenced_json_and_marks_local_llm_used(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)
    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _FencedJsonRewritePromptClient(),
            {"local_llm_attempted": True, "provider_init_status": "ready", "provider_init_error": None, "provider_init_reason": None},
        ),
    )
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["rewrite_used_local_llm"] is True
    assert runtime["rewrite_result_type"] == "rewritten"
    assert runtime["rewrite_failure_stage"] is None
    assert runtime["rewrite_provider_error"] is None
    assert runtime["per_stage_local_llm_status"]["rewrite"] == "used"
    assert runtime["per_stage_fallback_reason"].get("rewrite") != "fallback_used"
    assert runtime["local_llm_used"] is True
    assert "rewrite" in runtime["stages_using_local_llm"]


def test_rewrite_validation_and_empty_failures_surface_specific_reasons(tmp_path: Path, monkeypatch: Any) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    monkeypatch.setattr("pathlib.Path.is_file", lambda _path: True)
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/models/llama.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _ValidationFailedRewritePromptClient(),
            {"local_llm_attempted": True, "provider_init_status": "ready", "provider_init_error": None, "provider_init_reason": None},
        ),
    )
    validation_build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    validation_build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, validation_state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=validation_build.dependencies,
        selected_documents=[descriptor],
    )
    validation_runtime = build_real_debug_payload(
        latest_state=validation_state, selected_documents=[descriptor], scope_meta=validation_build.scope_meta
    )["local_llm_runtime"]
    assert validation_runtime["rewrite_result_type"] == "failed"
    assert validation_runtime["rewrite_fallback_reason"] == "validation_failed"
    assert validation_runtime["rewrite_failure_stage"] == "response_validation"
    assert validation_runtime["per_stage_fallback_reason"]["rewrite"] == "validation_failed"
    assert validation_runtime["rewrite_provider_error"] == "validation_failed:missing_rewritten_query"
    assert validation_runtime["local_llm_used"] is False

    monkeypatch.setattr(
        "ui.local_backend.build_local_prompt_llm_with_diagnostics",
        lambda *_args, **_kwargs: (
            _EmptyRewritePromptClient(),
            {"local_llm_attempted": True, "provider_init_status": "ready", "provider_init_error": None, "provider_init_reason": None},
        ),
    )
    empty_build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    empty_build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, empty_state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=empty_build.dependencies,
        selected_documents=[descriptor],
    )
    empty_runtime = build_real_debug_payload(
        latest_state=empty_state, selected_documents=[descriptor], scope_meta=empty_build.scope_meta
    )["local_llm_runtime"]
    assert empty_runtime["rewrite_result_type"] == "failed"
    assert empty_runtime["rewrite_fallback_reason"] == "empty_response"
    assert empty_runtime["rewrite_failure_stage"] == "normalization"
    assert empty_runtime["per_stage_fallback_reason"]["rewrite"] == "empty_response"
    assert empty_runtime["rewrite_provider_error"] == "empty_response:rewritten_query_blank"
    assert empty_runtime["local_llm_used"] is False
    assert empty_runtime["per_stage_local_llm_status"]["decomposition"] in {"skipped", "fallback"}
    assert empty_runtime["per_stage_local_llm_status"]["synthesis"] in {"blocked", "skipped", "fallback"}


def test_rewrite_provider_not_ready_reports_explicit_reason(tmp_path: Path) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(doc_path, "# MSA\n\n## Parties\nAcme Corp employs Jane Roe.\n")
    descriptor = {"id": "uploaded:msa.md", "name": "msa.md", "path": str(doc_path), "type": "md", "source": "uploaded"}
    settings = effective_local_llm_settings(
        enable_local_llm=True,
        provider="llama_cpp",
        model_path="/does/not/exist/model.gguf",
        temperature=0.0,
        timeout_seconds=8.0,
        n_ctx=4096,
        max_tokens=512,
        n_gpu_layers=0,
        threads=None,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    build = build_local_backend_dependencies([descriptor], local_llm_settings=settings)
    build.dependencies.retrieval.classify_query_state = lambda *_args, **_kwargs: _forced_rewrite_decision()
    _, state = run_legal_rag_turn_with_state(
        query="who is the hiring company here?",
        dependencies=build.dependencies,
        selected_documents=[descriptor],
    )
    runtime = build_real_debug_payload(latest_state=state, selected_documents=[descriptor], scope_meta=build.scope_meta)[
        "local_llm_runtime"
    ]
    assert runtime["rewrite_attempted"] is False
    assert runtime["per_stage_fallback_reason"]["rewrite"] == "provider_not_ready"
    assert runtime["rewrite_fallback_reason"] == "provider_not_ready"


def test_rewrite_mock_backend_reports_explicit_reason() -> None:
    runtime = build_real_debug_payload(
        latest_state={"warnings": [], "should_rewrite": True},
        scope_meta={
            "local_llm": {
                "ui_enabled": True,
                "effective_enabled": False,
                "mock_backend_active": True,
                "provider_init_status": "disabled",
                "stage_toggles": {"rewrite": True, "decomposition": True, "synthesis": True},
            }
        },
    )["local_llm_runtime"]
    assert runtime["rewrite_attempted"] is False
    assert runtime["per_stage_fallback_reason"]["rewrite"] == "mock_backend_active"
    assert runtime["rewrite_fallback_reason"] == "mock_backend_active"
