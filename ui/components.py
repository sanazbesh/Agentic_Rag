"""Reusable Streamlit UI components for legal RAG inspection."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from ui.backend_adapter import BackendAdapterError, get_available_documents, parse_recent_messages
from ui.upload_manager import ALLOWED_EXTENSIONS, remove_uploaded_document, save_uploaded_files

DEBUG_SECTIONS = [
    ("adapter_meta", "Adapter Meta"),
    ("rewritten_query", "Rewritten Query"),
    ("extracted_entities", "Extracted Entities"),
    ("filters", "Filters"),
    ("hybrid_search_results", "Hybrid Search Results"),
    ("reranked_child_results", "Reranked Child Results"),
    ("parent_chunks", "Parent Chunks"),
    ("compressed_context", "Compressed Context"),
    ("warnings", "Warnings"),
]


def initialize_session_state() -> None:
    """Initialize session defaults for local debugging workflow."""

    defaults: dict[str, Any] = {
        "query": "",
        "conversation_summary": "",
        "recent_messages_json": "[]",
        "selected_document_ids": [],
        "selected_documents": [],
        "uploaded_documents": [],
        "use_mock_backend": True,
        "show_debug": True,
        "last_run": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_upload_controls() -> None:
    """Render local upload controls and update uploaded-document session registry."""

    st.sidebar.subheader("Upload legal documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .pdf, .md, or .txt files",
        type=sorted(ALLOWED_EXTENSIONS),
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

    _render_upload_controls()

    available_documents = get_available_documents(
        use_mock_backend=use_mock_backend,
        uploaded_documents=st.session_state.uploaded_documents,
    )
    document_options = {doc["id"]: doc for doc in available_documents}

    valid_defaults = [doc_id for doc_id in st.session_state.selected_document_ids if doc_id in document_options]
    if len(valid_defaults) != len(st.session_state.selected_document_ids):
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

    selected_documents = [document_options[doc_id] for doc_id in selected_document_ids]
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
    }


def render_query_input() -> dict[str, Any]:
    """Render query and optional context input form."""

    st.subheader("Run Legal RAG Query")
    with st.form("rag_query_form", clear_on_submit=False):
        query = st.text_area(
            "Query",
            value=st.session_state.query,
            height=100,
            placeholder="Ask a legal question to test retrieval and grounding...",
        )
        conversation_summary = st.text_area(
            "Conversation summary (optional)",
            value=st.session_state.conversation_summary,
            height=100,
        )
        recent_messages_json = st.text_area(
            "recent_messages JSON (optional)",
            value=st.session_state.recent_messages_json,
            height=130,
            help='Expected: [{"role": "user", "content": "..."}, ...]',
        )

        col_run, col_reset = st.columns([1, 1])
        run_clicked = col_run.form_submit_button("Run", use_container_width=True)
        reset_clicked = col_reset.form_submit_button("Reset Inputs", use_container_width=True)

    if reset_clicked:
        st.session_state.query = ""
        st.session_state.conversation_summary = ""
        st.session_state.recent_messages_json = "[]"
        st.rerun()

    st.session_state.query = query
    st.session_state.conversation_summary = conversation_summary
    st.session_state.recent_messages_json = recent_messages_json

    parse_error = None
    recent_messages = None
    try:
        recent_messages = parse_recent_messages(recent_messages_json)
    except BackendAdapterError as exc:
        parse_error = str(exc)

    return {
        "run_clicked": run_clicked,
        "query": query,
        "conversation_summary": conversation_summary or None,
        "recent_messages": recent_messages,
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


def render_citations(citations: list[dict[str, Any]]) -> None:
    """Render citations in structured evidence cards."""

    st.subheader("Citations")
    if not citations:
        st.info("No citations returned.")
        return

    for index, citation in enumerate(citations, start=1):
        source_name = citation.get("source_name") or "Unknown source"
        heading = citation.get("heading") or "(No heading)"
        with st.expander(f"Citation {index}: {source_name} — {heading}", expanded=False):
            st.markdown(f"- **source_name:** {source_name}")
            st.markdown(f"- **heading:** {heading}")
            st.markdown(f"- **parent_chunk_id:** {citation.get('parent_chunk_id', '')}")
            if citation.get("document_id"):
                st.markdown(f"- **document_id:** {citation.get('document_id')}")
            st.markdown("- **supporting_excerpt:**")
            st.code(citation.get("supporting_excerpt") or "", language="text")


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

    blob = {"final_result": final_result, "debug_payload": debug_payload}
    st.download_button(
        "Download latest result JSON",
        data=json.dumps(blob, indent=2),
        file_name="legal_rag_result.json",
        mime="application/json",
        use_container_width=False,
    )
