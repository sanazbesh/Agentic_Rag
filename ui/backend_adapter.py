"""Backend adapter boundary for the Streamlit legal RAG test UI.

The UI should call `run_backend_query(...)` only. This adapter is responsible for
switching between mock and real backends and for returning:
- final_result: strict final answer model as a dictionary
- debug_payload: optional structured debug object
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ui.mock_backend import get_mock_documents, run_mock_backend_query
from ui.upload_manager import register_uploaded_documents

REQUIRED_FINAL_FIELDS = {"answer_text", "grounded", "sufficient_context", "citations", "warnings"}
logger = logging.getLogger(__name__)


class BackendAdapterError(RuntimeError):
    """Raised when backend invocation or response validation fails."""


@dataclass(slots=True)
class BackendQueryResponse:
    final_result: dict[str, Any]
    debug_payload: dict[str, Any] | None


def _adapter_debug_meta(
    *,
    mode: str,
    selected_documents: list[dict[str, Any]] | None,
    has_real_debug_payload: bool,
) -> dict[str, Any]:
    selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
    selected_ids = [str(doc.get("id")) for doc in selected_docs if doc.get("id")]
    selected_paths = [str(doc.get("path")) for doc in selected_docs if doc.get("path")]
    uses_uploaded = any(doc.get("source") == "uploaded" for doc in selected_docs)
    return {
        "backend_mode": mode,
        "selected_document_ids": selected_ids,
        "selected_document_paths": selected_paths,
        "uses_uploaded_documents": uses_uploaded,
        "has_real_debug_payload": has_real_debug_payload,
    }


def _coerce_final_result(raw: Any) -> dict[str, Any]:
    """Convert supported result objects to a dictionary for strict validation."""

    if isinstance(raw, Mapping):
        return dict(raw)

    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    if hasattr(raw, "__dict__"):
        return dict(raw.__dict__)

    raise BackendAdapterError("Unsupported backend final result type; expected mapping or model-like object.")


def validate_final_result(raw_result: Any) -> dict[str, Any]:
    """Validate final answer payload against strict required keys and shape."""

    result = _coerce_final_result(raw_result)
    missing = REQUIRED_FINAL_FIELDS.difference(result.keys())
    if missing:
        raise BackendAdapterError(
            "Backend result is missing required final fields: " + ", ".join(sorted(missing))
        )

    if not isinstance(result.get("answer_text"), str):
        raise BackendAdapterError("`answer_text` must be a string.")
    if not isinstance(result.get("grounded"), bool):
        raise BackendAdapterError("`grounded` must be a boolean.")
    if not isinstance(result.get("sufficient_context"), bool):
        raise BackendAdapterError("`sufficient_context` must be a boolean.")
    if not isinstance(result.get("citations"), list):
        raise BackendAdapterError("`citations` must be a list.")
    if not isinstance(result.get("warnings"), list):
        raise BackendAdapterError("`warnings` must be a list.")

    return result


def parse_recent_messages(raw_text: str) -> list[dict[str, Any]] | None:
    """Parse JSON recent messages input safely.

    Expected shape (v1): list[dict], for example:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """

    stripped = raw_text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise BackendAdapterError(f"Invalid JSON for recent_messages: {exc.msg}") from exc

    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise BackendAdapterError("recent_messages must be a JSON list of objects.")

    return parsed


def get_available_documents(
    *,
    use_mock_backend: bool | None = None,
    use_mock: bool | None = None,
    uploaded_documents: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return documents for sidebar selection.

    Document descriptor contract (v1):
    ``{"id", "name", "path", "type", "source", ...}``

    Current sources are:
    - uploaded local documents
    - mock backend documents (when mock mode is enabled)

    Real backend document listing can be merged here later.
    """
    if use_mock_backend is None and use_mock is None:
        raise BackendAdapterError("Expected `use_mock_backend` (or legacy `use_mock`) argument.")

    mock_mode = use_mock_backend if use_mock_backend is not None else use_mock

    documents: list[dict[str, Any]] = []
    documents.extend(uploaded_documents or [])

    if mock_mode:
        documents.extend(get_mock_documents())

    return documents


def run_backend_query(
    *,
    query: str,
    conversation_summary: str | None,
    recent_messages: list[dict[str, Any]] | None,
    selected_documents: list[dict[str, Any]] | None,
    use_mock_backend: bool,
    real_backend_runner: Callable[..., Any] | None = None,
    real_debug_runner: Callable[..., Any] | None = None,
) -> BackendQueryResponse:
    """Run query against mock or real backend.

    `selected_documents` is a list of stable document descriptors from the UI,
    including uploaded local files. Real backend wiring can consume descriptor
    metadata directly (e.g., ``path``) for ingestion/retrieval handoff.

    Real mode supports two integration options:
    1) `real_backend_runner`: callable returning final result model/object.
    2) `real_debug_runner`: optional callable returning full debug/state object.
       If present, this adapter stores that output in `debug_payload`.
    """

    register_uploaded_documents(selected_documents or [])
    selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
    logger.info(
        "run_backend_query use_mock_backend=%s selected_document_ids=%s selected_document_paths=%s",
        use_mock_backend,
        [doc.get("id") for doc in selected_docs],
        [doc.get("path") for doc in selected_docs],
    )

    if use_mock_backend:
        final_result, debug_payload = run_mock_backend_query(
            query=query,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            selected_documents=selected_documents,
        )
        payload = dict(debug_payload or {})
        payload["adapter_meta"] = _adapter_debug_meta(
            mode="mock",
            selected_documents=selected_documents,
            has_real_debug_payload=False,
        )
        return BackendQueryResponse(
            final_result=validate_final_result(final_result),
            debug_payload=payload,
        )

    if real_backend_runner is None:
        raise BackendAdapterError(
            "Real backend mode is enabled, but no runner is configured. "
            "Wire a callable that invokes run_legal_rag_turn(...)."
        )

    raw_final_result = real_backend_runner(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        selected_documents=selected_documents,
    )
    debug_payload = None
    if real_debug_runner is not None:
        debug_raw = real_debug_runner(
            query=query,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            selected_documents=selected_documents,
        )
        if isinstance(debug_raw, Mapping):
            debug_payload = dict(debug_raw)
    payload = dict(debug_payload or {})
    payload["adapter_meta"] = _adapter_debug_meta(
        mode="real",
        selected_documents=selected_documents,
        has_real_debug_payload=debug_payload is not None,
    )

    return BackendQueryResponse(
        final_result=validate_final_result(raw_final_result),
        debug_payload=payload,
    )
