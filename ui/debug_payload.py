from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def _to_debug_jsonable(value: Any) -> Any:
    """Convert nested runtime objects into JSON-safe values for debug rendering."""

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_debug_jsonable(value.model_dump())
    if is_dataclass(value):
        return _to_debug_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_debug_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_debug_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _to_debug_jsonable(vars(value))
    return value


def build_real_debug_payload(
    *,
    latest_state: dict[str, Any],
    selected_documents: list[dict[str, Any]] | None = None,
    scope_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build real-backend debug payload from full orchestration state.

    `assess_answerability(...)` runs in the answer-stage graph and writes
    `answerability_result` to state; this helper surfaces that typed object in
    a JSON-safe structure for the UI debug panel.
    """

    selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
    selected_ids = [str(doc.get("id")) for doc in selected_docs if doc.get("id")]
    selected_paths = [str(doc.get("path")) for doc in selected_docs if doc.get("path")]
    resolution = latest_state.get("context_resolution")
    if isinstance(resolution, dict):
        resolved_topics = list(resolution.get("resolved_topic_hints", []))
    else:
        resolved_topics = list(getattr(resolution, "resolved_topic_hints", []))

    def _read_decomposition_gate_state(state: dict[str, Any]) -> dict[str, Any]:
        needs = state.get("needs_decomposition")
        reasons = state.get("decomposition_gate_reasons")
        stable_needs = needs if isinstance(needs, bool) else False
        stable_reasons = reasons if isinstance(reasons, list) and all(isinstance(item, str) for item in reasons) else []
        return {
            "needs_decomposition": stable_needs,
            "decomposition_gate_reasons": list(stable_reasons),
        }

    answerability_result = _to_debug_jsonable(latest_state.get("answerability_result"))
    decomposition = _read_decomposition_gate_state(latest_state)
    warnings = list(latest_state.get("warnings", []))
    invoked = bool(latest_state.get("answerability_assessment_invoked", False))
    if not invoked:
        warnings.append("answerability_result unavailable: assess_answerability not invoked in this execution path")

    scope = dict(scope_meta) if isinstance(scope_meta, dict) else {}
    local_llm_scope = scope.get("local_llm") if isinstance(scope, dict) else None
    runtime = _derive_local_llm_runtime(latest_state=latest_state, local_llm_scope=local_llm_scope)

    return {
        "meta": {
            "mode": "real",
            "selected_document_ids": selected_ids,
            "selected_document_paths": selected_paths,
            "uses_uploaded_documents": any(doc.get("source") == "uploaded" for doc in selected_docs),
        },
        "scope": scope,
        "local_llm_runtime": runtime,
        "query_classification": _to_debug_jsonable(latest_state.get("query_classification")),
        "context_resolution": _to_debug_jsonable(latest_state.get("context_resolution")),
        "decomposition": decomposition,
        "answerability_result": answerability_result,
        "resolved_query": latest_state.get("resolved_query"),
        "effective_query": latest_state.get("effective_query"),
        "resolved_document_scope": _to_debug_jsonable(latest_state.get("last_resolved_document_scope", [])),
        "resolved_topic_hints": resolved_topics,
        "trace": _to_debug_jsonable(latest_state.get("trace")),
        "metrics": _to_debug_jsonable(latest_state.get("metrics")),
        "warnings": warnings,
        "recent_messages_used": _to_debug_jsonable(latest_state.get("recent_messages", [])),
    }


def _derive_local_llm_runtime(*, latest_state: dict[str, Any], local_llm_scope: Any) -> dict[str, Any]:
    scope = local_llm_scope if isinstance(local_llm_scope, dict) else {}
    stage_toggles = scope.get("stage_toggles") if isinstance(scope.get("stage_toggles"), dict) else {}
    warnings = [str(item) for item in list(latest_state.get("warnings", []))]
    final_answer = latest_state.get("final_answer")
    final_warnings = []
    if final_answer is not None:
        final_warnings = [str(item) for item in list(getattr(final_answer, "warnings", []))]

    used_stages: list[str] = []
    fallback_stages: list[str] = []
    if any(item.startswith("rewrite_path:llm:") for item in warnings):
        used_stages.append("rewrite")
    if any(item.startswith("rewrite_path:deterministic_fallback:") for item in warnings):
        fallback_stages.append("rewrite")

    decomposition_plan = latest_state.get("decomposition_plan")
    planner_notes = [str(item) for item in list(getattr(decomposition_plan, "planner_notes", []))]
    if any(item.startswith("planner_path:llm:") for item in planner_notes):
        used_stages.append("decomposition")
    if any(item.startswith("planner_path:deterministic_fallback:") for item in planner_notes):
        fallback_stages.append("decomposition")

    if any(item.startswith("answer_synthesis_path:llm:") for item in final_warnings):
        used_stages.append("synthesis")
    if any(item.startswith("answer_synthesis_path:deterministic_fallback:") for item in final_warnings):
        fallback_stages.append("synthesis")

    enabled = bool(scope.get("effective_enabled", False))
    provider_init_reason = scope.get("provider_init_reason")
    provider_init_status = scope.get("provider_init_status", "not_attempted")

    def _stage_status(stage: str) -> tuple[str, str | None, bool]:
        toggle_enabled = bool(stage_toggles.get(stage, False))
        used = stage in used_stages
        if used:
            return "used", None, True
        if not toggle_enabled:
            return "skipped", "disabled_by_toggle", False

        if stage == "rewrite":
            reached = bool(latest_state.get("should_rewrite", False))
            if not reached:
                return "skipped", "stage_not_reached", False
            if not enabled:
                return "fallback", "fallback_used", True
            if provider_init_status != "ready":
                return "fallback", str(provider_init_reason or "provider_init_failed"), True
            if stage in fallback_stages:
                return "fallback", "inference_failed", True
            return "skipped", "stage_not_applicable", False

        if stage == "decomposition":
            reached = bool(latest_state.get("needs_decomposition", False))
            if not reached:
                return "skipped", "stage_not_applicable", False
            if not enabled:
                return "fallback", "fallback_used", True
            if provider_init_status != "ready":
                return "fallback", str(provider_init_reason or "provider_init_failed"), True
            if stage in fallback_stages:
                return "fallback", "inference_failed", True
            return "skipped", "stage_not_reached", False

        reached = bool(latest_state.get("should_generate_answer", False))
        if not reached:
            if bool(latest_state.get("should_return_insufficient_response", False)):
                return "blocked", "upstream_blocked", False
            return "skipped", "stage_not_reached", False
        if not enabled:
            return "fallback", "fallback_used", True
        if provider_init_status != "ready":
            return "fallback", str(provider_init_reason or "provider_init_failed"), True
        if stage in fallback_stages:
            return "fallback", "inference_failed", True
        return "skipped", "stage_not_applicable", False

    per_stage_status: dict[str, str] = {}
    per_stage_reason: dict[str, str] = {}
    attempted_stages: list[str] = []
    for stage_name in ("rewrite", "decomposition", "synthesis"):
        status, reason, attempted = _stage_status(stage_name)
        per_stage_status[stage_name] = status
        if reason is not None:
            per_stage_reason[stage_name] = reason
        if attempted:
            attempted_stages.append(stage_name)

    effective_mode = "llama_cpp_assisted" if used_stages else "deterministic"
    return {
        "ui_enabled": bool(scope.get("ui_enabled", False)),
        "effective_enabled": enabled,
        "provider": scope.get("provider", "llama_cpp"),
        "model_path": scope.get("model_path", ""),
        "n_ctx": scope.get("n_ctx"),
        "temperature": scope.get("temperature"),
        "timeout_seconds": scope.get("timeout_seconds"),
        "max_tokens": scope.get("max_tokens"),
        "n_gpu_layers": scope.get("n_gpu_layers"),
        "threads": scope.get("threads"),
        "stage_toggles": {
            "rewrite": bool(stage_toggles.get("rewrite", False)),
            "decomposition": bool(stage_toggles.get("decomposition", False)),
            "synthesis": bool(stage_toggles.get("synthesis", False)),
        },
        "local_llm_attempted": bool(scope.get("local_llm_attempted", False)),
        "provider_init_status": provider_init_status,
        "provider_init_error": scope.get("provider_init_error"),
        "provider_init_reason": provider_init_reason,
        "stages_using_local_llm": sorted(set(used_stages)),
        "fallback_stages": sorted(set(fallback_stages)),
        "stages_attempted_local_llm": sorted(set(attempted_stages)),
        "per_stage_local_llm_status": per_stage_status,
        "per_stage_fallback_reason": per_stage_reason,
        "local_llm_used": bool(used_stages),
        "effective_mode": effective_mode if enabled else "deterministic",
    }
