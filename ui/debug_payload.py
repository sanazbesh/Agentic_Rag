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

    return {
        "meta": {
            "mode": "real",
            "selected_document_ids": selected_ids,
            "selected_document_paths": selected_paths,
            "uses_uploaded_documents": any(doc.get("source") == "uploaded" for doc in selected_docs),
        },
        "scope": dict(scope_meta) if isinstance(scope_meta, dict) else {},
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
