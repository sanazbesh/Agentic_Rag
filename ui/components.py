"""Reusable Streamlit UI components for legal RAG inspection."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

import streamlit as st

from ui.backend_adapter import BackendAdapterError, get_available_documents, parse_recent_messages
from ui.local_backend import effective_local_llm_settings
from ui.upload_manager import ALLOWED_EXTENSIONS, remove_uploaded_document, save_uploaded_files

DEBUG_SECTIONS = [
    ("adapter_meta", "Adapter Meta"),
    ("local_llm_runtime", "Local LLM Runtime"),
    ("query_classification", "Query Classification"),
    ("context_resolution", "Conversation Resolution"),
    ("decomposition", "Decomposition Gate"),
    ("answerability_result", "Answerability Assessment"),
    ("resolved_query", "Resolved Query"),
    ("effective_query", "Effective Query"),
    ("resolved_document_scope", "Resolved Document Scope"),
    ("resolved_topic_hints", "Resolved Topic Hints"),
    ("recent_messages_used", "Recent Messages Used"),
    ("rewritten_query", "Rewritten Query"),
    ("extracted_entities", "Extracted Entities"),
    ("filters", "Filters"),
    ("hybrid_search_results", "Hybrid Search Results"),
    ("reranked_child_results", "Reranked Child Results"),
    ("parent_chunks", "Parent Chunks"),
    ("compressed_context", "Compressed Context"),
    ("warnings", "Warnings"),
]


def _to_jsonable(value: Any) -> Any:
    """Convert nested objects (including pydantic models) into JSON-safe values."""

    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _to_jsonable(value.model_dump())
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return value


def initialize_session_state() -> None:
    """Initialize session defaults for local debugging workflow."""

    defaults: dict[str, Any] = {
        "current_query_input": "",
        "conversation_summary_input": "",
        "recent_messages_override": "[]",
        "pending_query_input_clear": False,
        "pending_full_reset": False,
        "conversation_history": [],
        "selected_document_ids": [],
        "selected_documents": [],
        "selected_persisted_document_ids": [],
        "selected_persisted_document_version_ids": [],
        "uploaded_documents": [],
        "use_mock_backend": True,
        "show_debug": True,
        "latest_result": None,
        "latest_debug_payload": None,
        "last_run": None,
        "local_llm_enabled": False,
        "local_llm_provider": "llama_cpp",
        "local_llm_model_path": "",
        "local_llm_n_ctx": 4096,
        "local_llm_temperature": 0.0,
        "local_llm_timeout_seconds": 8.0,
        "local_llm_max_tokens": 512,
        "local_llm_n_gpu_layers": 0,
        "local_llm_threads": 0,
        "local_llm_stage_rewrite": True,
        "local_llm_stage_decomposition": True,
        "local_llm_stage_synthesis": True,
        "latest_ingestion_result": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)




def _safe_session_list(value: Any) -> list[Any]:
    """Normalize session-state list-like values and guard against type objects."""

    if isinstance(value, type):
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return []


def _render_upload_controls() -> None:
    """Render local upload controls and update uploaded-document session registry."""

    st.sidebar.subheader("Persistent ingestion")
    build_runtime, ingest_uploaded, ingestion_dependency_error = _load_persistent_ingestion_dependencies()
    persistent_upload = st.sidebar.file_uploader(
        "Upload for persistent ingestion (.pdf, .md, .txt)",
        type=_file_uploader_types(),
        accept_multiple_files=False,
        key="persistent_ingestion_upload",
        help="This stores files in the persistent document store via IngestionOrchestrator.",
    )
    if ingestion_dependency_error:
        st.sidebar.info(ingestion_dependency_error)
    if st.sidebar.button(
        "Ingest uploaded document",
        use_container_width=True,
        disabled=persistent_upload is None or ingestion_dependency_error is not None,
    ):
        try:
            runtime = build_runtime()
            ingestion_result = ingest_uploaded(persistent_upload, runtime=runtime)
            st.session_state.latest_ingestion_result = ingestion_result
            if ingestion_result.error_message:
                st.sidebar.error(f"Ingestion failed: {ingestion_result.error_message}")
            elif ingestion_result.duplicate_document:
                st.sidebar.warning("Duplicate document detected. Existing document version was reused.")
            else:
                st.sidebar.success("Document ingested successfully.")
            st.rerun()
        except Exception:  # pragma: no cover - defensive UI error handling
            import traceback

            st.sidebar.error("Persistent ingestion failed")
            st.sidebar.code(traceback.format_exc())

    latest_ingestion_result = st.session_state.get("latest_ingestion_result")
    if latest_ingestion_result is not None:
        st.sidebar.markdown("**Latest ingestion result**")
        st.sidebar.json(
            {
                "document_id": latest_ingestion_result.document_id,
                "document_version_id": latest_ingestion_result.document_version_id,
                "ingestion_job_id": latest_ingestion_result.ingestion_job_id,
                "status": latest_ingestion_result.status,
                "error_message": latest_ingestion_result.error_message,
                "status_from_database": latest_ingestion_result.status_from_database,
            },
            expanded=False,
        )

    st.sidebar.divider()
    st.sidebar.subheader("Upload legal documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .pdf, .md, or .txt files",
        type=_file_uploader_types(),
        accept_multiple_files=True,
        help="Files are stored locally in data/uploads for this test UI.",
    )

    if st.sidebar.button("Save uploaded files", use_container_width=True):
        saved_documents, errors = save_uploaded_files(uploaded_files)

        if saved_documents:
            st.session_state.uploaded_documents.extend(saved_documents)
            st.sidebar.success(f"Saved {len(saved_documents)} file(s) to local upload storage.")

        for error in errors:
            st.sidebar.error(error)

        if not saved_documents and not errors:
            st.sidebar.warning("No files were provided for upload.")

        if saved_documents:
            st.rerun()

    uploaded_documents: list[dict[str, Any]] = st.session_state.uploaded_documents
    if not uploaded_documents:
        st.sidebar.caption("No uploaded documents yet.")
        return

    st.sidebar.markdown("**Uploaded documents**")
    for doc in uploaded_documents:
        st.sidebar.caption(f"• {doc['name']} ({doc.get('type', 'unknown')}, {doc.get('size_bytes', 0)} bytes)")

    removable_ids = [doc["id"] for doc in uploaded_documents]
    remove_id = st.sidebar.selectbox(
        "Remove uploaded document",
        options=[""] + removable_ids,
        format_func=lambda doc_id: "Select a document" if not doc_id else doc_id,
    )
    if st.sidebar.button("Remove selected upload", use_container_width=True, disabled=not remove_id):
        document = next((doc for doc in uploaded_documents if doc["id"] == remove_id), None)
        if document is not None:
            remove_uploaded_document(document)
            st.session_state.uploaded_documents = [doc for doc in uploaded_documents if doc["id"] != remove_id]
            st.session_state.selected_document_ids = [
                doc_id for doc_id in st.session_state.selected_document_ids if doc_id != remove_id
            ]
            st.sidebar.success(f"Removed {remove_id}.")
            st.rerun()

    if st.sidebar.button("Clear all uploaded documents", use_container_width=True):
        for document in uploaded_documents:
            remove_uploaded_document(document)
        st.session_state.uploaded_documents = []
        st.session_state.selected_document_ids = [
            doc_id for doc_id in st.session_state.selected_document_ids if not doc_id.startswith("uploaded:")
        ]
        st.sidebar.success("Cleared uploaded document list.")
        st.rerun()


def _load_persistent_ingestion_dependencies() -> tuple[Any | None, Any | None, str | None]:
    """Load persistent ingestion dependencies lazily for optional SQLAlchemy environments."""

    try:
        from ui.persistent_ingestion import build_ingestion_runtime_from_env, ingest_uploaded_document
    except ModuleNotFoundError as exc:
        dependency_name = exc.name or "optional dependency"
        return None, None, (
            "Persistent ingestion is unavailable in this environment. "
            f"Missing dependency: {dependency_name}."
        )
    except Exception as exc:  # pragma: no cover - defensive import guardrail
        return None, None, f"Persistent ingestion is unavailable: {type(exc).__name__}: {exc}"

    return build_ingestion_runtime_from_env, ingest_uploaded_document, None


def _load_persisted_document_dependencies() -> tuple[Any | None, Any | None, Any | None, str | None]:
    """Load persisted document listing dependencies lazily for optional SQLAlchemy environments."""

    try:
        from ui.persisted_documents import (
            PersistedDocumentServiceError,
            list_persisted_documents,
            ready_persisted_documents,
            to_document_descriptor,
        )
    except ModuleNotFoundError as exc:
        dependency_name = exc.name or "optional dependency"
        return None, None, None, (
            "Persisted document selection is unavailable in this environment. "
            f"Missing dependency: {dependency_name}."
        )
    except Exception as exc:  # pragma: no cover - defensive import guardrail
        return None, None, None, f"Persisted document selection is unavailable: {type(exc).__name__}: {exc}"

    return (
        list_persisted_documents,
        ready_persisted_documents,
        to_document_descriptor,
        None,
    )


def render_sidebar(
    *,
    real_backend_available: bool,
    real_backend_wiring_error: str | None,
) -> dict[str, Any]:
    """Render sidebar controls and return selected options."""

    st.sidebar.header("RAG Controls")
    use_mock_backend = st.sidebar.toggle(
        "Use mock backend",
        value=st.session_state.use_mock_backend,
        disabled=not real_backend_available,
        help="Turn off to use your real run_legal_rag_turn wiring.",
    )
    if not real_backend_available:
        use_mock_backend = True
        st.sidebar.caption(
            "Real backend unavailable in this session. "
            f"Reason: {real_backend_wiring_error or 'not configured'}."
        )

    st.session_state.use_mock_backend = use_mock_backend

    st.sidebar.subheader("Local LLM Controls")
    llm_controls_disabled = use_mock_backend
    if llm_controls_disabled:
        st.sidebar.caption("Mock backend is active; local LLM settings are ignored for this run.")

    enable_local_llm = st.sidebar.toggle(
        "Enable local LLM",
        value=st.session_state.local_llm_enabled,
        disabled=llm_controls_disabled,
    )
    st.session_state.local_llm_enabled = enable_local_llm

    st.sidebar.text_input("Provider", value="llama.cpp (llama-cpp-python)", disabled=True)
    local_llm_model_path = st.sidebar.text_input(
        "GGUF model path",
        value=st.session_state.local_llm_model_path,
        disabled=llm_controls_disabled,
    )
    st.session_state.local_llm_model_path = local_llm_model_path

    with st.sidebar.expander("Advanced local LLM settings", expanded=False):
        local_llm_n_ctx = st.number_input(
            "Context window (n_ctx)",
            min_value=128,
            max_value=32768,
            value=int(st.session_state.local_llm_n_ctx),
            step=128,
            disabled=llm_controls_disabled,
        )
        local_llm_temperature = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.local_llm_temperature),
            step=0.1,
            disabled=llm_controls_disabled,
        )
        local_llm_timeout_seconds = st.number_input(
            "Timeout (seconds)",
            min_value=0.5,
            max_value=120.0,
            value=float(st.session_state.local_llm_timeout_seconds),
            step=0.5,
            disabled=llm_controls_disabled,
        )
        local_llm_max_tokens = st.number_input(
            "Max tokens",
            min_value=32,
            max_value=4096,
            value=int(st.session_state.local_llm_max_tokens),
            step=32,
            disabled=llm_controls_disabled,
        )
        local_llm_n_gpu_layers = st.number_input(
            "n_gpu_layers",
            min_value=0,
            max_value=200,
            value=int(st.session_state.local_llm_n_gpu_layers),
            step=1,
            disabled=llm_controls_disabled,
        )
        local_llm_threads = st.number_input(
            "Threads (0 = auto)",
            min_value=0,
            max_value=128,
            value=int(st.session_state.local_llm_threads),
            step=1,
            disabled=llm_controls_disabled,
        )
        use_llm_rewrite = st.toggle(
            "Use local LLM for rewrite",
            value=st.session_state.local_llm_stage_rewrite,
            disabled=llm_controls_disabled,
        )
        use_llm_decomposition = st.toggle(
            "Use local LLM for decomposition",
            value=st.session_state.local_llm_stage_decomposition,
            disabled=llm_controls_disabled,
        )
        use_llm_synthesis = st.toggle(
            "Use local LLM for final synthesis",
            value=st.session_state.local_llm_stage_synthesis,
            disabled=llm_controls_disabled,
        )
    st.session_state.local_llm_temperature = float(local_llm_temperature)
    st.session_state.local_llm_timeout_seconds = float(local_llm_timeout_seconds)
    st.session_state.local_llm_n_ctx = int(local_llm_n_ctx)
    st.session_state.local_llm_max_tokens = int(local_llm_max_tokens)
    st.session_state.local_llm_n_gpu_layers = int(local_llm_n_gpu_layers)
    st.session_state.local_llm_threads = int(local_llm_threads)
    st.session_state.local_llm_stage_rewrite = use_llm_rewrite
    st.session_state.local_llm_stage_decomposition = use_llm_decomposition
    st.session_state.local_llm_stage_synthesis = use_llm_synthesis

    local_llm_settings = effective_local_llm_settings(
        enable_local_llm=enable_local_llm,
        provider="llama_cpp",
        model_path=local_llm_model_path,
        temperature=float(local_llm_temperature),
        timeout_seconds=float(local_llm_timeout_seconds),
        n_ctx=int(local_llm_n_ctx),
        max_tokens=int(local_llm_max_tokens),
        n_gpu_layers=int(local_llm_n_gpu_layers),
        threads=None if int(local_llm_threads) <= 0 else int(local_llm_threads),
        use_rewrite=use_llm_rewrite,
        use_decomposition=use_llm_decomposition,
        use_synthesis=use_llm_synthesis,
        mock_backend_active=use_mock_backend,
    )
    st.session_state.local_llm_effective_settings = local_llm_settings

    _render_upload_controls()

    persisted_document_descriptors: list[dict[str, Any]] = []
    list_persisted_documents, ready_persisted_documents, to_document_descriptor, persisted_dependency_error = (
        _load_persisted_document_dependencies()
    )
    if persisted_dependency_error:
        st.sidebar.info(persisted_dependency_error)
    else:
        try:
            persisted_documents = list_persisted_documents()
            if not persisted_documents:
                st.sidebar.caption("No persisted documents found.")
            else:
                ready_rows = ready_persisted_documents(persisted_documents)
                if not ready_rows:
                    st.sidebar.caption("No READY documents available.")
                else:
                    ready_options = [row.document_id for row in ready_rows]
                    ready_map = {row.document_id: row for row in ready_rows}
                    selected_persisted_ids = _safe_session_list(st.session_state.get("selected_persisted_document_ids", []))
                    st.session_state.selected_persisted_document_ids = selected_persisted_ids
                    default_ready_ids = [doc_id for doc_id in selected_persisted_ids if doc_id in ready_map]
                    selected_ready_ids = st.sidebar.multiselect(
                        "Persisted READY documents",
                        options=ready_options,
                        default=default_ready_ids,
                        format_func=lambda doc_id: (
                            f"{ready_map[doc_id].source_name} "
                            f"[{ready_map[doc_id].source_type}:{ready_map[doc_id].status}] "
                            f"v={ready_map[doc_id].current_version_id or 'none'}"
                        ),
                        help="Select READY persisted documents loaded from Postgres.",
                    )
                    st.session_state.selected_persisted_document_ids = selected_ready_ids
                    st.session_state.selected_persisted_document_version_ids = [
                        ready_map[doc_id].current_version_id
                        for doc_id in selected_ready_ids
                        if ready_map[doc_id].current_version_id
                    ]
                    persisted_document_descriptors = [
                        to_document_descriptor(ready_map[doc_id]) for doc_id in selected_ready_ids
                    ]
        except Exception:
            import traceback

            st.sidebar.error("Failed to load persisted documents")
            st.sidebar.code(traceback.format_exc())

    available_documents = get_available_documents(
        use_mock_backend=use_mock_backend,
        uploaded_documents=st.session_state.uploaded_documents,
    )
    document_options = {doc["id"]: doc for doc in available_documents}

    selected_document_ids_state = _safe_session_list(st.session_state.get("selected_document_ids", []))
    st.session_state.selected_document_ids = selected_document_ids_state
    valid_defaults = [doc_id for doc_id in selected_document_ids_state if doc_id in document_options]
    if not valid_defaults and st.session_state.uploaded_documents:
        uploaded_ids = [str(doc.get("id")) for doc in st.session_state.uploaded_documents if doc.get("id")]
        valid_defaults = [doc_id for doc_id in uploaded_ids if doc_id in document_options]
    if len(valid_defaults) != len(selected_document_ids_state):
        st.session_state.selected_document_ids = valid_defaults

    selected_document_ids = st.sidebar.multiselect(
        "Selected legal documents",
        options=list(document_options.keys()),
        default=valid_defaults,
        format_func=lambda doc_id: (
            f"{document_options[doc_id]['name']} "
            f"[{document_options[doc_id].get('source', 'unknown')}:{document_options[doc_id].get('type', 'n/a')}]"
        ),
        help=(
            "Selected documents are passed to backend_adapter.run_backend_query(...) as "
            "structured descriptors, including local uploaded file paths."
        ),
    )

    selected_documents = [document_options[doc_id] for doc_id in selected_document_ids] + persisted_document_descriptors
    st.session_state.selected_document_ids = selected_document_ids
    st.session_state.selected_documents = selected_documents

    st.sidebar.subheader("Pipeline settings")
    show_debug = st.sidebar.toggle(
        "Show debug details",
        value=st.session_state.show_debug,
        help="Show/hide retrieval and debug inspection panels.",
    )
    st.session_state.show_debug = show_debug

    if st.sidebar.button("Clear latest result", use_container_width=True):
        st.session_state.last_run = None

    if show_debug and st.session_state.uploaded_documents:
        with st.sidebar.expander("Debug: uploaded metadata", expanded=False):
            compact_debug = [
                {
                    "id": doc.get("id"),
                    "path": doc.get("path"),
                    "type": doc.get("type"),
                    "source": doc.get("source"),
                }
                for doc in st.session_state.uploaded_documents
            ]
            st.json(compact_debug, expanded=False)

    return {
        "use_mock_backend": use_mock_backend,
        "selected_documents": selected_documents,
        "show_debug": show_debug,
        "local_llm_settings": local_llm_settings,
    }


def render_runtime_mode_status(*, use_mock_backend: bool, debug_payload: dict[str, Any] | None) -> None:
    payload = debug_payload or {}
    llm_runtime = payload.get("local_llm_runtime") if isinstance(payload, dict) else None
    if use_mock_backend:
        st.info("Runtime mode: Mock backend active — local LLM ignored.")
        return
    if not isinstance(llm_runtime, dict):
        st.caption("Runtime mode: Deterministic mode.")
        return
    mode = str(llm_runtime.get("effective_mode") or "deterministic").strip().lower()
    stages = [stage for stage in list(llm_runtime.get("stages_using_local_llm", [])) if isinstance(stage, str)]
    if mode == "llama_cpp_assisted":
        stage_label = ", ".join(stages) if stages else "(none)"
        st.success(f"Runtime mode: llama.cpp-assisted mode. Stages using llama.cpp: {stage_label}.")
    else:
        st.caption("Runtime mode: Deterministic mode.")


def render_query_input() -> dict[str, Any]:
    """Render query form with independent state for input vs conversation memory.

    `current_query_input` is intentionally separate from `conversation_history` and
    latest result payloads so normal multi-turn runs do not require manual reset.
    """

    st.subheader("Run Legal RAG Query")
    with st.form("rag_query_form", clear_on_submit=False):
        st.text_area(
            "Query",
            key="current_query_input",
            height=100,
            placeholder="Ask a legal question to test retrieval and grounding...",
        )
        st.text_area(
            "Conversation summary (optional)",
            key="conversation_summary_input",
            height=100,
        )
        st.text_area(
            "recent_messages JSON (optional)",
            key="recent_messages_override",
            height=130,
            help='Expected: [{"role": "user", "content": "..."}, ...]',
        )

        col_run, col_reset = st.columns([1, 1])
        run_clicked = col_run.form_submit_button("Run", use_container_width=True)
        reset_clicked = col_reset.form_submit_button("Start New Conversation", use_container_width=True)

    if reset_clicked:
        # Defer widget-backed state mutation until next rerun to avoid
        # Streamlit's "cannot be modified after widget is instantiated" error.
        st.session_state.pending_full_reset = True
        st.rerun()

    parse_error = None
    recent_messages = None
    try:
        recent_messages = parse_recent_messages(st.session_state.recent_messages_override)
    except BackendAdapterError as exc:
        parse_error = str(exc)

    return {
        "run_clicked": run_clicked,
        "clear_clicked": reset_clicked,
        "query": st.session_state.current_query_input,
        "conversation_summary": st.session_state.conversation_summary_input or None,
        "recent_messages": recent_messages,
        "recent_messages_override_used": bool(
            st.session_state.recent_messages_override.strip() and st.session_state.recent_messages_override.strip() != "[]"
        ),
        "recent_messages_parse_error": parse_error,
    }


def render_answer_panel(final_result: dict[str, Any]) -> None:
    """Render strict final result model values and status indicators."""

    st.subheader("Final Answer")

    grounded = final_result["grounded"]
    sufficient_context = final_result["sufficient_context"]

    status_col_1, status_col_2 = st.columns(2)
    if grounded:
        status_col_1.success("Grounded: True")
    else:
        status_col_1.error("Grounded: False")

    if sufficient_context:
        status_col_2.success("Sufficient context: True")
    else:
        status_col_2.warning("Sufficient context: False")

    st.markdown("#### answer_text")
    st.write(final_result["answer_text"])

    warnings = final_result.get("warnings", [])
    if warnings:
        st.markdown("#### warnings")
        for warning in warnings:
            st.warning(str(warning))


def _citation_value(citation: Any, key: str, default: Any = None) -> Any:
    """Read citation fields from dict-like or object-like citation payloads."""

    if isinstance(citation, dict):
        return citation.get(key, default)
    return getattr(citation, key, default)


def render_citations(citations: list[Any]) -> None:
    """Render citations in structured evidence cards."""

    st.subheader("Citations")
    if not citations:
        st.info("No citations returned.")
        return

    for index, citation in enumerate(citations, start=1):
        source_name = _citation_value(citation, "source_name") or "Unknown source"
        heading = _citation_value(citation, "heading") or "(No heading)"
        with st.expander(f"Citation {index}: {source_name} — {heading}", expanded=False):
            st.markdown(f"- **source_name:** {source_name}")
            st.markdown(f"- **heading:** {heading}")
            st.markdown(f"- **parent_chunk_id:** {_citation_value(citation, 'parent_chunk_id', '')}")
            document_id = _citation_value(citation, "document_id")
            if document_id:
                st.markdown(f"- **document_id:** {document_id}")
            st.markdown("- **supporting_excerpt:**")
            st.code(_citation_value(citation, "supporting_excerpt") or "", language="text")


def render_debug_panel(final_result: dict[str, Any], debug_payload: dict[str, Any] | None) -> None:
    """Render debug payload sections plus raw payload expansion."""

    st.subheader("Debug / Inspection")
    payload = debug_payload or {}

    for key, title in DEBUG_SECTIONS:
        if key not in payload:
            continue
        with st.expander(title, expanded=False):
            value = payload[key]
            if isinstance(value, (dict, list)):
                st.json(value, expanded=False)
            elif isinstance(value, str):
                st.code(value, language="text")
            else:
                st.write(value)

    with st.expander("Raw final result + debug payload", expanded=False):
        st.markdown("**Raw final_result**")
        st.json(final_result, expanded=False)
        st.markdown("**Raw debug_payload**")
        st.json(payload, expanded=False)


def render_download_button(final_result: dict[str, Any], debug_payload: dict[str, Any] | None) -> None:
    """Provide download action for latest response/debug bundle."""

    blob = {"final_result": _to_jsonable(final_result), "debug_payload": _to_jsonable(debug_payload)}
    st.download_button(
        "Download latest result JSON",
        data=json.dumps(blob, indent=2),
        file_name="legal_rag_result.json",
        mime="application/json",
        use_container_width=False,
    )
def _file_uploader_types() -> list[str]:
    """Return Streamlit-safe upload extension list."""

    fallback = ["md", "pdf", "txt"]
    configured = ALLOWED_EXTENSIONS
    if isinstance(configured, type):
        return fallback

    try:
        candidates = list(configured)
    except TypeError:
        return fallback

    cleaned: list[str] = []
    for ext in candidates:
        if not isinstance(ext, str):
            return fallback
        normalized = ext.lstrip(".").strip().lower()
        if not normalized:
            return fallback
        cleaned.append(normalized)

    return sorted(set(cleaned)) or fallback
