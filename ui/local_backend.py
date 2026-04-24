"""Local real-backend wiring for Streamlit legal RAG UI.

Builds an in-memory end-to-end legal RAG dependency set from uploaded/local files,
so users can run non-mock retrieval + grounded answering without external services.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from contextlib import contextmanager

from agentic_rag.llm import LocalLLMConfig, build_local_prompt_llm_with_diagnostics
from agentic_rag.chunking import MarkdownParentChildChunker
from agentic_rag.ingestion import MarkdownDocumentIngestor, PDFDocumentIngestor
from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies
from agentic_rag.orchestration.retrieval_graph import RetrievalDependencies, llm_assisted_decomposition_plan
from agentic_rag.retrieval import (
    ChildChunkSearcher,
    ChunkReranker,
    InMemoryChildChunkRepository,
    InMemoryKeywordChunkRepository,
    InMemoryParentChunkRepository,
    KeywordSearchService,
    ParentChildRetrievalTools,
    ParentChunkStore,
)
from agentic_rag.tools import LegalAnswerSynthesizer, compress_context, extract_legal_entities
from agentic_rag.tools.query_intelligence import QueryTransformationService


@dataclass(slots=True)
class LocalBackendBuildResult:
    dependencies: LegalRagDependencies
    scope_meta: dict[str, Any]


@dataclass(slots=True, frozen=True)
class LocalLLMStageToggles:
    rewrite: bool = True
    decomposition: bool = True
    synthesis: bool = True


@dataclass(slots=True, frozen=True)
class LocalLLMRuntimeSettings:
    ui_enabled: bool = False
    enabled: bool = False
    provider: str = "llama_cpp"
    model_path: str = ""
    n_ctx: int = 4096
    temperature: float = 0.0
    timeout_seconds: float = 8.0
    max_tokens: int = 512
    n_gpu_layers: int = 0
    threads: int | None = None
    stages: LocalLLMStageToggles = LocalLLMStageToggles()
    mock_backend_active: bool = False

    def is_stage_enabled(self, stage: str) -> bool:
        if not self.enabled:
            return False
        return bool(getattr(self.stages, stage, False))

    def as_local_llm_config(self) -> LocalLLMConfig:
        return LocalLLMConfig(
            enabled=self.enabled,
            provider=self.provider,
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            temperature=self.temperature,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            n_gpu_layers=self.n_gpu_layers,
            threads=self.threads,
        )


def effective_local_llm_settings(
    *,
    enable_local_llm: bool,
    provider: str,
    model_path: str,
    temperature: float,
    timeout_seconds: float,
    n_ctx: int,
    max_tokens: int,
    n_gpu_layers: int,
    threads: int | None,
    use_rewrite: bool,
    use_decomposition: bool,
    use_synthesis: bool,
    mock_backend_active: bool,
) -> LocalLLMRuntimeSettings:
    provider_name = (provider or "llama_cpp").strip().lower() or "llama_cpp"
    safe_model_path = (model_path or "").strip()
    return LocalLLMRuntimeSettings(
        ui_enabled=bool(enable_local_llm),
        enabled=bool(enable_local_llm) and not mock_backend_active,
        provider=provider_name,
        model_path=safe_model_path,
        n_ctx=max(128, int(n_ctx)),
        temperature=float(temperature),
        timeout_seconds=max(0.5, float(timeout_seconds)),
        max_tokens=max(32, int(max_tokens)),
        n_gpu_layers=max(0, int(n_gpu_layers)),
        threads=None if threads is None else max(1, int(threads)),
        stages=LocalLLMStageToggles(
            rewrite=bool(use_rewrite),
            decomposition=bool(use_decomposition),
            synthesis=bool(use_synthesis),
        ),
        mock_backend_active=mock_backend_active,
    )


def build_local_backend_dependencies(
    selected_documents: list[dict[str, Any]] | None,
    *,
    local_llm_settings: LocalLLMRuntimeSettings | None = None,
) -> LocalBackendBuildResult:
    """Build LegalRagDependencies from selected local documents.

    Supported file types: `.md`, `.txt`, `.pdf`.
    """

    selected_docs = [doc for doc in (selected_documents or []) if isinstance(doc, dict)]
    loaded_documents, warnings = _load_documents(selected_docs)

    chunker = MarkdownParentChildChunker()
    child_records: list[dict[str, Any]] = []
    parent_lookup: dict[str, dict[str, Any]] = {}

    for doc in loaded_documents:
        chunking_result = chunker.chunk(doc)
        child_records.extend(chunking_result.child_qdrant_records())
        parent_lookup.update(chunking_result.parent_lookup())

    supported_filter_keys: set[str] = set()
    for record in child_records:
        payload = record.get("payload")
        if isinstance(payload, dict):
            supported_filter_keys.update(str(key) for key in payload.keys())

    retrieval_tools = ParentChildRetrievalTools(
        child_searcher=ChildChunkSearcher(
            repository=InMemoryChildChunkRepository(child_records),
            default_limit=20,
        ),
        parent_store=ParentChunkStore(
            repository=InMemoryParentChunkRepository(parent_lookup),
        ),
        keyword_search_service=KeywordSearchService(
            repository=InMemoryKeywordChunkRepository(child_records),
            default_limit=20,
        ),
        chunk_reranker=ChunkReranker(),
    )

    def _hybrid_search(query: str, *, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[Any]:
        scoped_filters = dict(filters or {})
        selected_ids = _as_string_set(scoped_filters.pop("selected_document_ids", []))
        selected_paths = _as_string_set(scoped_filters.pop("selected_document_paths", []))

        base_filters = {
            key: value for key, value in scoped_filters.items() if key in supported_filter_keys
        }
        results = retrieval_tools.hybrid_search(query=query, filters=base_filters or None, top_k=top_k)
        if not selected_ids and not selected_paths:
            return results

        filtered: list[Any] = []
        for result in results:
            document_id = str(result.document_id)
            metadata = dict(result.metadata or {})
            source_path = str(metadata.get("source", ""))
            if selected_ids and document_id in selected_ids:
                filtered.append(result)
                continue
            if selected_paths and source_path in selected_paths:
                filtered.append(result)
        return filtered

    llm_settings = local_llm_settings or LocalLLMRuntimeSettings()
    provider_label = f"{llm_settings.provider}:{llm_settings.model_path or 'unset_model_path'}"
    if llm_settings.enabled:
        llm_client, llm_diagnostics = build_local_prompt_llm_with_diagnostics(llm_settings.as_local_llm_config())
    else:
        llm_client, llm_diagnostics = None, {
            "local_llm_attempted": False,
            "provider_init_status": "disabled",
            "provider_init_error": None,
            "provider_init_reason": None,
        }

    rewrite_service = QueryTransformationService(
        llm_client=llm_client if llm_settings.is_stage_enabled("rewrite") else None,
        llm_provider_label=provider_label,
    )
    if llm_settings.is_stage_enabled("decomposition"):
        decomposition_planner = _build_stage_scoped_decomposition_planner(llm_settings)
    else:
        decomposition_planner = _deterministic_decomposition_plan

    answer_synthesizer = LegalAnswerSynthesizer(
        llm_client=llm_client if llm_settings.is_stage_enabled("synthesis") else None,
        llm_provider_label=provider_label,
    )

    retrieval_dependencies = RetrievalDependencies(
        rewrite_query=rewrite_service.rewrite_query,
        extract_legal_entities=extract_legal_entities,
        hybrid_search=_hybrid_search,
        rerank_chunks=retrieval_tools.rerank_chunks,
        retrieve_parent_chunks=retrieval_tools.retrieve_parent_chunks,
        compress_context=compress_context,
        plan_decomposition=decomposition_planner,
    )

    scope_meta = {
        "backend": "local_in_memory",
        "selected_document_count": len(selected_docs),
        "loaded_document_count": len(loaded_documents),
        "parent_chunk_count": len(parent_lookup),
        "child_chunk_count": len(child_records),
        "warnings": warnings,
        "local_llm": {
            "ui_enabled": llm_settings.ui_enabled,
            "effective_enabled": llm_settings.enabled,
            "mock_backend_active": llm_settings.mock_backend_active,
            "provider": llm_settings.provider,
            "model_path": llm_settings.model_path,
            "n_ctx": llm_settings.n_ctx,
            "temperature": llm_settings.temperature,
            "timeout_seconds": llm_settings.timeout_seconds,
            "max_tokens": llm_settings.max_tokens,
            "n_gpu_layers": llm_settings.n_gpu_layers,
            "threads": llm_settings.threads,
            "stage_toggles": {
                "rewrite": llm_settings.stages.rewrite,
                "decomposition": llm_settings.stages.decomposition,
                "synthesis": llm_settings.stages.synthesis,
            },
            "local_llm_attempted": bool(llm_diagnostics.get("local_llm_attempted", False)),
            "provider_init_status": str(llm_diagnostics.get("provider_init_status", "not_attempted")),
            "provider_init_error": llm_diagnostics.get("provider_init_error"),
            "provider_init_reason": llm_diagnostics.get("provider_init_reason"),
        },
    }
    return LocalBackendBuildResult(
        dependencies=LegalRagDependencies(
            retrieval=retrieval_dependencies,
            generate_grounded_answer=answer_synthesizer.generate,
        ),
        scope_meta=scope_meta,
    )


def _deterministic_decomposition_plan(**kwargs: Any) -> Any:
    import agentic_rag.orchestration.retrieval_graph as retrieval_graph

    return retrieval_graph.build_decomposition_plan(**kwargs)


def _build_stage_scoped_decomposition_planner(settings: LocalLLMRuntimeSettings):
    def _planner(**kwargs: Any) -> Any:
        with _temporary_local_llm_env(settings):
            return llm_assisted_decomposition_plan(**kwargs)

    return _planner


@contextmanager
def _temporary_local_llm_env(settings: LocalLLMRuntimeSettings):
    env_updates = {
        "AGENTIC_RAG_LOCAL_LLM_ENABLED": "true" if settings.enabled else "false",
        "AGENTIC_RAG_LOCAL_LLM_PROVIDER": settings.provider,
        "AGENTIC_RAG_LOCAL_LLM_MODEL_PATH": settings.model_path,
        "AGENTIC_RAG_LOCAL_LLM_N_CTX": str(settings.n_ctx),
        "AGENTIC_RAG_LOCAL_LLM_TEMPERATURE": str(settings.temperature),
        "AGENTIC_RAG_LOCAL_LLM_TIMEOUT_SECONDS": str(settings.timeout_seconds),
        "AGENTIC_RAG_LOCAL_LLM_MAX_TOKENS": str(settings.max_tokens),
        "AGENTIC_RAG_LOCAL_LLM_N_GPU_LAYERS": str(settings.n_gpu_layers),
        "AGENTIC_RAG_LOCAL_LLM_THREADS": "" if settings.threads is None else str(settings.threads),
    }
    prior = {key: os.environ.get(key) for key in env_updates}
    os.environ.update(env_updates)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_documents(selected_documents: list[dict[str, Any]]) -> tuple[list[Any], list[str]]:
    markdown_records: list[dict[str, Any]] = []
    pdf_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for descriptor in selected_documents:
        path_value = descriptor.get("path")
        if not path_value:
            continue
        path = Path(str(path_value))
        if not path.exists() or not path.is_file():
            warnings.append(f"missing_file:{path}")
            continue

        suffix = path.suffix.lower()
        source_name = str(descriptor.get("name") or path.name)
        base_record = {
            "id": str(descriptor.get("id") or path.name),
            "source": str(path),
            "source_name": source_name,
            "source_type": suffix.lstrip("."),
        }

        if suffix in {".md", ".txt"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:  # pragma: no cover - defensive I/O guard
                warnings.append(f"read_failed:{path}:{type(exc).__name__}")
                continue
            markdown_records.append({**base_record, "text": text})
            continue

        if suffix == ".pdf":
            try:
                payload = path.read_bytes()
            except Exception as exc:  # pragma: no cover - defensive I/O guard
                warnings.append(f"read_failed:{path}:{type(exc).__name__}")
                continue
            pdf_records.append({**base_record, "content": payload})
            continue

        warnings.append(f"unsupported_extension:{path.name}")

    docs = []
    if markdown_records:
        docs.extend(MarkdownDocumentIngestor().ingest(markdown_records))

    if pdf_records:
        try:
            docs.extend(PDFDocumentIngestor().ingest(pdf_records))
        except Exception as exc:
            warnings.append(f"pdf_ingestion_failed:{type(exc).__name__}:{exc}")

    return docs, warnings


def _as_string_set(values: Any) -> set[str]:
    if isinstance(values, list):
        return {str(v) for v in values if v}
    return set()
