"""Streamlit local-first test UI for legal RAG pipeline inspection.

Run locally:
    streamlit run app.py

Integration note:
- Keep `use_mock_backend=True` for immediate testing.
- To wire real backend, provide `real_backend_runner` and optionally
  `real_debug_runner` in `build_real_backend_runners()`.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import streamlit as st

from ui.backend_adapter import BackendAdapterError, run_backend_query
from ui.components import (
    initialize_session_state,
    render_answer_panel,
    render_citations,
    render_debug_panel,
    render_download_button,
    render_query_input,
    render_sidebar,
)


st.set_page_config(page_title="Legal RAG Test UI", layout="wide")
logger = logging.getLogger(__name__)


def _ensure_src_path_for_local_runs() -> None:
    """Ensure `src/` is importable when running `streamlit run app.py` locally."""

    if importlib.util.find_spec("agentic_rag") is not None:
        return

    src_path = Path(__file__).resolve().parent / "src"
    if src_path.is_dir():
        sys.path.insert(0, str(src_path))
        logger.info("Added src path for local imports: %s", src_path)


_ensure_src_path_for_local_runs()


def build_real_backend_runners() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None, str | None]:
    """Return configured real backend runners.

    Replace this with your project-specific wiring to invoke:
        run_legal_rag_turn(query=..., conversation_summary=..., recent_messages=..., selected_documents=...)

    Why this adapter exists:
    - keeps UI independent from orchestration wiring
    - supports strict final-result contract
    - allows optional debug payload runner without changing UI code
    """

    try:
        from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn
    except Exception as exc:
        return None, None, f"Unable to import legal RAG runner: {type(exc).__name__}: {exc}"

    dependencies = st.session_state.get("legal_rag_dependencies")
    if dependencies is None:
        return (
            None,
            None,
            "Real backend not wired. Set st.session_state['legal_rag_dependencies'] with a LegalRagDependencies "
            "instance before running with mock mode disabled.",
        )
    if not isinstance(dependencies, LegalRagDependencies):
        return (
            None,
            None,
            "Real backend wiring invalid: st.session_state['legal_rag_dependencies'] must be a "
            "LegalRagDependencies instance.",
        )

    retrieval_config = st.session_state.get("legal_rag_retrieval_config")

    def real_backend_runner(
        *,
        query: str,
        conversation_summary: str | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        selected_documents: list[dict[str, Any]] | None = None,
    ) -> Any:
        selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
        selected_ids = [str(doc.get("id")) for doc in selected_docs if doc.get("id")]
        selected_paths = [str(doc.get("path")) for doc in selected_docs if doc.get("path")]

        base_hybrid_search = dependencies.retrieval.hybrid_search

        def hybrid_search_with_selected_docs(user_query: str, *, filters: dict[str, Any] | None, top_k: int) -> Any:
            merged_filters = dict(filters or {})
            if selected_ids:
                merged_filters["selected_document_ids"] = selected_ids
            if selected_paths:
                merged_filters["selected_document_paths"] = selected_paths
            logger.info(
                "real_backend_hybrid_search selected_document_ids=%s selected_document_paths=%s",
                selected_ids,
                selected_paths,
            )
            return base_hybrid_search(user_query, filters=merged_filters or None, top_k=top_k)

        wrapped_dependencies = replace(
            dependencies,
            retrieval=replace(dependencies.retrieval, hybrid_search=hybrid_search_with_selected_docs),
        )

        return run_legal_rag_turn(
            query=query,
            dependencies=wrapped_dependencies,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            retrieval_config=retrieval_config,
        )

    def real_debug_runner(
        *,
        selected_documents: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
        selected_ids = [str(doc.get("id")) for doc in selected_docs if doc.get("id")]
        selected_paths = [str(doc.get("path")) for doc in selected_docs if doc.get("path")]
        return {
            "meta": {
                "mode": "real",
                "selected_document_ids": selected_ids,
                "selected_document_paths": selected_paths,
                "uses_uploaded_documents": any(doc.get("source") == "uploaded" for doc in selected_docs),
            }
        }

    return real_backend_runner, real_debug_runner, None


def main() -> None:
    st.title("Legal RAG Inspection Dashboard")
    st.caption("Local-first Streamlit UI for testing retrieval, grounding, citations, and debug state.")

    initialize_session_state()

    sidebar_state = render_sidebar()
    input_state = render_query_input()

    if input_state["recent_messages_parse_error"]:
        st.error(input_state["recent_messages_parse_error"])

    if input_state["run_clicked"]:
        if not input_state["query"].strip():
            st.warning("Please enter a query before running.")
        elif input_state["recent_messages_parse_error"]:
            st.warning("Please fix recent_messages JSON before running.")
        else:
            real_backend_runner, real_debug_runner, real_backend_wiring_error = build_real_backend_runners()
            if not sidebar_state["use_mock_backend"] and real_backend_runner is None:
                st.warning(real_backend_wiring_error or "Real backend mode is enabled, but no backend is wired.")
                logger.warning("real_backend_not_wired reason=%s", real_backend_wiring_error)
                return
            with st.spinner("Running legal RAG pipeline..."):
                try:
                    response = run_backend_query(
                        query=input_state["query"],
                        conversation_summary=input_state["conversation_summary"],
                        recent_messages=input_state["recent_messages"],
                        selected_documents=sidebar_state["selected_documents"],
                        use_mock_backend=sidebar_state["use_mock_backend"],
                        real_backend_runner=real_backend_runner,
                        real_debug_runner=real_debug_runner,
                    )
                    st.session_state.last_run = {
                        "final_result": response.final_result,
                        "debug_payload": response.debug_payload,
                    }
                except BackendAdapterError as exc:
                    st.warning(f"Backend adapter warning: {exc}")
                    st.session_state.last_run = {
                        "error": str(exc),
                        "query": input_state["query"],
                        "selected_documents": sidebar_state["selected_documents"],
                    }
                except Exception as exc:  # pragma: no cover - defensive UI error handling
                    st.error(f"Backend call failed: {type(exc).__name__}: {exc}")
                    st.session_state.last_run = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "query": input_state["query"],
                    }

    last_run = st.session_state.last_run
    if not last_run:
        st.info("Submit a query to see answer, citations, and debug details.")
        return

    if "error" in last_run:
        st.error("Latest run failed. See diagnostic below.")
        st.json(last_run, expanded=False)
        return

    final_result = last_run["final_result"]
    debug_payload = last_run.get("debug_payload")

    render_answer_panel(final_result)
    render_citations(final_result.get("citations", []))
    adapter_meta = (debug_payload or {}).get("adapter_meta") if isinstance(debug_payload, dict) else None
    if isinstance(adapter_meta, dict):
        st.caption(
            "Backend mode: "
            f"{adapter_meta.get('backend_mode', 'unknown')} | "
            f"Selected docs: {len(adapter_meta.get('selected_document_ids', []))} | "
            f"Uses uploaded documents: {adapter_meta.get('uses_uploaded_documents', False)}"
        )

    if sidebar_state["show_debug"]:
        render_debug_panel(final_result, debug_payload)

    render_download_button(final_result, debug_payload)


if __name__ == "__main__":
    main()
