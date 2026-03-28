"""Reusable Streamlit UI components for legal RAG inspection."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from ui.backend_adapter import BackendAdapterError, get_available_documents, parse_recent_messages

DEBUG_SECTIONS = [
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
        "selected_documents": [],
        "use_mock_backend": True,
        "show_debug": True,
        "last_run": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return selected options."""

    st.sidebar.header("RAG Controls")
    use_mock_backend = st.sidebar.toggle(
        "Use mock backend",
        value=st.session_state.use_mock_backend,
        help="Turn off to use your real run_legal_rag_turn wiring.",
    )
    st.session_state.use_mock_backend = use_mock_backend

    available_documents = get_available_documents(use_mock_backend=use_mock_backend)
    document_options = {doc["id"]: doc["name"] for doc in available_documents}
    selected_documents = st.sidebar.multiselect(
        "Selected legal documents",
        options=list(document_options.keys()),
        default=st.session_state.selected_documents,
        format_func=lambda doc_id: f"{document_options.get(doc_id, doc_id)} ({doc_id})",
        help="Mock list now; replace in backend_adapter.get_available_documents for real index data.",
    )
    st.session_state.selected_documents = selected_documents

    st.sidebar.subheader("Upload (placeholder)")
    st.sidebar.file_uploader(
        "Optional legal document upload (not wired in v1)",
        type=["pdf", "md", "txt", "docx"],
        disabled=True,
    )

    st.sidebar.subheader("Pipeline settings")
    show_debug = st.sidebar.toggle(
        "Show debug details",
        value=st.session_state.show_debug,
        help="Show/hide retrieval and debug inspection panels.",
    )
    st.session_state.show_debug = show_debug

    if st.sidebar.button("Clear latest result", use_container_width=True):
        st.session_state.last_run = None

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
            st.json(payload[key], expanded=False)

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
