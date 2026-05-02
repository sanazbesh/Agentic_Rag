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


def _ensure_src_path_for_local_runs() -> None:
    """Ensure `src/` is importable when running `streamlit run app.py` locally."""

    if importlib.util.find_spec("agentic_rag") is not None:
        return

    src_path = Path(__file__).resolve().parent / "src"
    if src_path.is_dir():
        sys.path.insert(0, str(src_path))
        logger.info("Added src path for local imports: %s", src_path)


logger = logging.getLogger(__name__)
_ensure_src_path_for_local_runs()

from ui.backend_adapter import BackendAdapterError, run_backend_query
from ui.components import (
    initialize_session_state,
    render_answer_panel,
    render_citations,
    render_debug_panel,
    render_download_button,
    render_query_input,
    render_runtime_mode_status,
    render_sidebar,
)
from ui.local_backend import LocalLLMRuntimeSettings, build_local_backend_dependencies
from ui.session_memory import append_conversation_turn, build_backend_context
from ui.quality_dashboard import render_quality_dashboard
from ui.trace_dashboard import render_trace_dashboard
from ui.triage_dashboard import render_triage_dashboard
from ui.review_queue_dashboard import render_review_queue_dashboard
from ui.debug_payload import build_real_debug_payload


def _has_persisted_selection(selected_docs: list[dict[str, Any]]) -> bool:
    return any(str(doc.get("source", "")) == "persisted" and doc.get("document_id") for doc in selected_docs)


def _build_persistent_backend_dependencies(selected_docs: list[dict[str, Any]]) -> tuple[Any | None, dict[str, Any], str | None]:
    """Build retrieval dependencies for persisted Postgres+Qdrant docs only."""

    persisted_docs = [doc for doc in selected_docs if str(doc.get("source", "")) == "persisted" and doc.get("document_id")]
    if not persisted_docs:
        return None, {}, None
    try:
        from sqlalchemy import func, select
        from qdrant_client import QdrantClient

        from agentic_rag.ingestion_pipeline.vector_indexing import qdrant_config_from_env
        from agentic_rag.indexing.dense_child_chunks import DEFAULT_COLLECTION_NAME
        from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies
        from agentic_rag.orchestration.retrieval_graph import RetrievalDependencies, llm_assisted_decomposition_plan
        from agentic_rag.retrieval import (
            ChildChunkSearcher,
            ChunkReranker,
            KeywordSearchService,
            ParentChildRetrievalTools,
            ParentChunkStore,
            PostgresChunkRepository,
            PostgresResolvedQdrantChildRepository,
            QdrantResultResolver,
        )
        from agentic_rag.storage import get_postgres_engine, get_postgres_session_factory, postgres_config_from_env
        from agentic_rag.storage.models import Chunk
        from agentic_rag.tools import LegalAnswerSynthesizer, compress_context, extract_legal_entities
        from agentic_rag.tools.query_intelligence import QueryTransformationService
    except Exception as exc:
        return None, {}, f"Persistent backend unavailable: {type(exc).__name__}: {exc}"

    class _QdrantDenseSearch:
        def __init__(self, client: Any, collection_name: str) -> None:
            self.client = client
            self.collection_name = collection_name

        def search(self, query: str, *, filters: dict[str, Any] | None = None, limit: int = 10) -> list[dict[str, Any]]:
            points = self.client.query_points(collection_name=self.collection_name, query=query, limit=limit).points
            return [{"id": str(point.id), "score": point.score, "payload": dict(point.payload or {})} for point in points]

    class _EmptyKeywordRepository:
        def search_keyword(self, query: str, *, filters: dict[str, Any] | None = None, limit: int = 10) -> list[Any]:
            del query, filters, limit
            return []

    postgres_config = postgres_config_from_env()
    engine = get_postgres_engine(postgres_config)
    session_factory = get_postgres_session_factory(engine)
    session = session_factory()
    chunk_repo = PostgresChunkRepository(session=session)
    qdrant = QdrantClient(url=qdrant_config_from_env().url)
    child_repo = PostgresResolvedQdrantChildRepository(
        qdrant_backend=_QdrantDenseSearch(qdrant, DEFAULT_COLLECTION_NAME),
        resolver=QdrantResultResolver(chunk_repository=chunk_repo),
    )
    retrieval_tools = ParentChildRetrievalTools(
        child_searcher=ChildChunkSearcher(repository=child_repo, default_limit=20),
        parent_store=ParentChunkStore(repository=chunk_repo),
        keyword_search_service=KeywordSearchService(repository=_EmptyKeywordRepository(), default_limit=20),
        chunk_reranker=ChunkReranker(),
    )
    rewrite_service = QueryTransformationService()
    retrieval_dependencies = RetrievalDependencies(
        rewrite_query=rewrite_service.rewrite_query,
        extract_legal_entities=extract_legal_entities,
        hybrid_search=retrieval_tools.hybrid_search,
        rerank_chunks=retrieval_tools.rerank_chunks,
        retrieve_parent_chunks=retrieval_tools.retrieve_parent_chunks,
        compress_context=compress_context,
        plan_decomposition=llm_assisted_decomposition_plan,
    )
    persisted_doc_ids = [str(doc["document_id"]) for doc in persisted_docs]
    persisted_ver_ids = [str(doc.get("document_version_id")) for doc in persisted_docs if doc.get("document_version_id")]
    child_count = session.execute(select(func.count(Chunk.id)).where(Chunk.document_id.in_(persisted_doc_ids), Chunk.chunk_type == "child")).scalar_one()
    deps = LegalRagDependencies(retrieval=retrieval_dependencies, generate_grounded_answer=LegalAnswerSynthesizer().generate)
    return deps, {
        "backend": "persistent_postgres_qdrant",
        "persisted_document_count": len(persisted_docs),
        "persisted_child_chunk_count": int(child_count),
        "selected_document_ids": persisted_doc_ids,
        "selected_document_version_ids": persisted_ver_ids,
    }, None


st.set_page_config(page_title="Legal RAG Test UI", layout="wide")


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
        from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn_with_state
    except Exception as exc:
        return None, None, f"Unable to import legal RAG runner: {type(exc).__name__}: {exc}"

    dependencies = st.session_state.get("legal_rag_dependencies")
    if dependencies is not None and not isinstance(dependencies, LegalRagDependencies):
        return (
            None,
            None,
            "Real backend wiring invalid: st.session_state['legal_rag_dependencies'] must be a "
            "LegalRagDependencies instance.",
        )

    retrieval_config = st.session_state.get("legal_rag_retrieval_config")

    latest_state: dict[str, Any] = {}

    def real_backend_runner(
        *,
        query: str,
        conversation_summary: str | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        selected_documents: list[dict[str, Any]] | None = None,
        local_llm_settings: LocalLLMRuntimeSettings | None = None,
    ) -> Any:
        selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
        selected_ids = [str(doc.get("id")) for doc in selected_docs if doc.get("id")]
        selected_paths = [str(doc.get("path")) for doc in selected_docs if doc.get("path")]

        active_dependencies = dependencies
        local_scope_meta: dict[str, Any] | None = None

        if active_dependencies is None:
            persistent_dependencies, persistent_scope_meta, persistent_error = _build_persistent_backend_dependencies(selected_docs)
            if persistent_dependencies is not None:
                active_dependencies = persistent_dependencies
                local_scope_meta = dict(persistent_scope_meta)
            else:
                if persistent_error and _has_persisted_selection(selected_docs):
                    logger.warning("persistent_backend_fallback_to_local reason=%s", persistent_error)
                local_backend = build_local_backend_dependencies(selected_docs, local_llm_settings=local_llm_settings)
                active_dependencies = local_backend.dependencies
                local_scope_meta = dict(local_backend.scope_meta)

        base_hybrid_search = active_dependencies.retrieval.hybrid_search

        def hybrid_search_with_selected_docs(user_query: str, *, filters: dict[str, Any] | None, top_k: int) -> Any:
            merged_filters = dict(filters or {})
            if selected_ids:
                merged_filters["selected_document_ids"] = selected_ids
            selected_version_ids = [str(doc.get("document_version_id")) for doc in selected_docs if doc.get("document_version_id")]
            if selected_version_ids:
                merged_filters["selected_document_version_ids"] = selected_version_ids
            if selected_paths:
                merged_filters["selected_document_paths"] = selected_paths
            logger.info(
                "real_backend_hybrid_search selected_document_ids=%s selected_document_paths=%s",
                selected_ids,
                selected_paths,
            )
            return base_hybrid_search(user_query, filters=merged_filters or None, top_k=top_k)

        wrapped_dependencies = replace(
            active_dependencies,
            retrieval=replace(active_dependencies.retrieval, hybrid_search=hybrid_search_with_selected_docs),
        )

        final_answer, final_state = run_legal_rag_turn_with_state(
            query=query,
            dependencies=wrapped_dependencies,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            selected_documents=selected_docs,
            retrieval_config=retrieval_config,
        )
        latest_state.clear()
        latest_state.update(dict(final_state))

        scope_meta = {
            "selected_document_count": len(selected_docs),
            "selected_document_ids": selected_ids,
            "selected_document_paths": selected_paths,
            "local_llm_ui_enabled": bool(local_llm_settings and local_llm_settings.ui_enabled),
        }
        if local_scope_meta:
            scope_meta.update(local_scope_meta)
        st.session_state["last_real_backend_scope_meta"] = scope_meta

        return final_answer

    def real_debug_runner(
        *,
        selected_documents: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return build_real_debug_payload(
            latest_state=latest_state,
            selected_documents=selected_documents,
            scope_meta=st.session_state.get("last_real_backend_scope_meta"),
        )

    return real_backend_runner, real_debug_runner, None


def main() -> None:
    page = st.sidebar.radio("Dashboard", options=["Inspection", "Quality", "Trace Debug", "Failure Triage", "Human Review Queue"], index=0)
    if page == "Quality":
        render_quality_dashboard()
        return
    if page == "Trace Debug":
        render_trace_dashboard()
        return
    if page == "Failure Triage":
        render_triage_dashboard()
        return
    if page == "Human Review Queue":
        render_review_queue_dashboard()
        return

    st.title("Legal RAG Inspection Dashboard")
    st.caption("Local-first Streamlit UI for testing retrieval, grounding, citations, and debug state.")

    initialize_session_state()
    if st.session_state.get("pending_full_reset"):
        st.session_state.current_query_input = ""
        st.session_state.conversation_summary_input = ""
        st.session_state.recent_messages_override = "[]"
        st.session_state.conversation_history = []
        st.session_state.latest_result = None
        st.session_state.latest_debug_payload = None
        st.session_state.last_run = None
        st.session_state.pending_full_reset = False
    if st.session_state.get("pending_query_input_clear"):
        st.session_state.current_query_input = ""
        st.session_state.pending_query_input_clear = False

    real_backend_runner, real_debug_runner, real_backend_wiring_error = build_real_backend_runners()

    sidebar_state = render_sidebar(
        real_backend_available=real_backend_runner is not None,
        real_backend_wiring_error=real_backend_wiring_error,
    )
    input_state = render_query_input()

    if input_state["recent_messages_parse_error"]:
        st.error(input_state["recent_messages_parse_error"])

    if input_state["run_clicked"]:
        if not input_state["query"].strip():
            st.warning("Please enter a query before running.")
        elif input_state["recent_messages_parse_error"]:
            st.warning("Please fix recent_messages JSON before running.")
        else:
            if not sidebar_state["use_mock_backend"] and real_backend_runner is None:
                st.warning(real_backend_wiring_error or "Real backend mode is enabled, but no backend is wired.")
                logger.warning("real_backend_not_wired reason=%s", real_backend_wiring_error)
                return
            with st.spinner("Running legal RAG pipeline..."):
                try:
                    conversation_history = st.session_state.get("conversation_history", [])
                    conversation_summary, recent_messages, _ = build_backend_context(
                        history=conversation_history,
                        conversation_summary_input=input_state["conversation_summary"],
                        recent_messages_override=(
                            input_state["recent_messages"] if input_state["recent_messages_override_used"] else None
                        ),
                    )

                    response = run_backend_query(
                        query=input_state["query"],
                        conversation_summary=conversation_summary,
                        recent_messages=recent_messages,
                        selected_documents=sidebar_state["selected_documents"],
                        use_mock_backend=sidebar_state["use_mock_backend"],
                        local_llm_settings=sidebar_state["local_llm_settings"],
                        real_backend_runner=real_backend_runner,
                        real_debug_runner=real_debug_runner,
                    )
                    st.session_state.last_run = {
                        "final_result": response.final_result,
                        "debug_payload": response.debug_payload,
                    }
                    st.session_state.latest_result = response.final_result
                    st.session_state.latest_debug_payload = response.debug_payload
                    debug_payload = response.debug_payload or {}
                    context_resolution = debug_payload.get("context_resolution") if isinstance(debug_payload, dict) else None
                    resolution_dict = (
                        context_resolution.model_dump()
                        if hasattr(context_resolution, "model_dump")
                        else (dict(context_resolution) if isinstance(context_resolution, dict) else {})
                    )
                    turn_metadata = {
                        "effective_query": debug_payload.get("effective_query"),
                        "resolved_document_ids": resolution_dict.get("resolved_document_ids", []),
                        "resolved_topic_hints": resolution_dict.get("resolved_topic_hints", []),
                        "answer_text": response.final_result.get("answer_text", ""),
                        "citations": response.final_result.get("citations", []),
                    }
                    st.session_state.conversation_history = append_conversation_turn(
                        history=conversation_history,
                        query=input_state["query"],
                        answer_text=response.final_result.get("answer_text", ""),
                        metadata=turn_metadata,
                    )
                    st.session_state.pending_query_input_clear = True
                    st.rerun()
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

    render_runtime_mode_status(
        use_mock_backend=bool(sidebar_state.get("use_mock_backend", True)),
        debug_payload=debug_payload,
    )
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
