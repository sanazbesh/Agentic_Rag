"""Streamlit local-first test UI for legal RAG pipeline inspection.

Run locally:
    streamlit run app.py

Integration note:
- Keep `use_mock_backend=True` for immediate testing.
- To wire real backend, provide `real_backend_runner` and optionally
  `real_debug_runner` in `build_real_backend_runners()`.
"""

from __future__ import annotations

from collections.abc import Callable
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


def build_real_backend_runners() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    """Return configured real backend runners.

    Replace this with your project-specific wiring to invoke:
        run_legal_rag_turn(query=..., conversation_summary=..., recent_messages=..., selected_documents=...)

    Why this adapter exists:
    - keeps UI independent from orchestration wiring
    - supports strict final-result contract
    - allows optional debug payload runner without changing UI code
    """

    # Example sketch (uncomment and adapt in your environment):
    # from agentic_rag.orchestration.legal_rag_graph import run_legal_rag_turn
    #
    # def real_backend_runner(query: str, conversation_summary=None, recent_messages=None, selected_documents=None):
    #     return run_legal_rag_turn(
    #         query=query,
    #         conversation_summary=conversation_summary,
    #         recent_messages=recent_messages,
    #         selected_documents=selected_documents,
    #         dependencies=...,  # your LegalRagDependencies
    #     )
    #
    # return real_backend_runner, None
    return None, None


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
            real_backend_runner, real_debug_runner = build_real_backend_runners()
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
                    st.error(f"Backend adapter error: {exc}")
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

    if sidebar_state["show_debug"]:
        render_debug_panel(final_result, debug_payload)

    render_download_button(final_result, debug_payload)


if __name__ == "__main__":
    main()
