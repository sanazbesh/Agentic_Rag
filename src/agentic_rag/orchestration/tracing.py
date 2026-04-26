"""Small in-repo structured tracing helpers for legal RAG orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from agentic_rag.versioning import get_version_attribution

SpanStatus = Literal["success", "partial", "failed", "skipped"]
TraceStatus = Literal["success", "partial", "failed"]

SCHEMA_VERSION = "trace.v1"
PIPELINE_VERSION = "legal_rag.v0.20"
REQUIRED_STAGES: tuple[str, ...] = (
    "query_understanding",
    "decomposition",
    "retrieval",
    "rerank",
    "parent_expansion",
    "answerability",
    "final_synthesis",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _warning_objects(warnings: Sequence[str]) -> list[dict[str, str]]:
    stable: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in warnings:
        code = str(raw).strip()
        if not code or code in seen:
            continue
        seen.add(code)
        stable.append({"code": code, "message": code, "severity": "low"})
    return stable


def _get_span(trace: dict[str, Any], stage: str) -> dict[str, Any] | None:
    for span in trace.get("spans", []):
        if span.get("stage") == stage:
            return span
    return None


def create_trace(*, query: str, selected_document_ids: Sequence[str], model_version: str | None = None) -> dict[str, Any]:
    versions = get_version_attribution(model_version=model_version)
    return {
        "trace_id": f"tr_{uuid4().hex}",
        "request_id": f"req_{uuid4().hex[:8]}",
        "timestamp_utc": _utc_now_iso(),
        "query": str(query),
        "selected_document_ids": [str(item) for item in selected_document_ids if item],
        "active_family": None,
        "overall_status": "success",
        "total_latency_ms": None,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "retrieval_version": versions["retrieval_version"],
        "answerability_version": versions["answerability_version"],
        "generation_version": versions["generation_version"],
        "prompt_bundle_version": versions["prompt_bundle_version"],
        "model_version": versions["model_version"],
        "spans": [],
        "_start_perf": perf_counter(),
    }


def begin_span(
    trace: dict[str, Any],
    *,
    stage: str,
    span_name: str,
    parent_stage: str | None = None,
    inputs_summary: Mapping[str, Any] | None = None,
) -> None:
    if _get_span(trace, stage) is not None:
        return
    parent = _get_span(trace, parent_stage) if parent_stage else None
    trace["spans"].append(
        {
            "span_id": f"sp_{stage}_{len(trace['spans']) + 1}",
            "parent_span_id": parent.get("span_id") if parent else None,
            "span_name": span_name,
            "stage": stage,
            "start_time_utc": _utc_now_iso(),
            "end_time_utc": None,
            "duration_ms": None,
            "status": "partial",
            "inputs_summary": dict(inputs_summary or {}),
            "outputs_summary": {},
            "warnings": [],
            "error": None,
            "_start_perf": perf_counter(),
        }
    )


def end_span(
    trace: dict[str, Any],
    *,
    stage: str,
    status: SpanStatus,
    outputs_summary: Mapping[str, Any] | None = None,
    warnings: Sequence[str] | None = None,
    error: Mapping[str, str] | None = None,
) -> None:
    span = _get_span(trace, stage)
    if span is None:
        return
    if span.get("end_time_utc") is not None:
        return
    start_perf = span.get("_start_perf")
    end_perf = perf_counter()
    duration_ms = int(round((end_perf - float(start_perf)) * 1000)) if isinstance(start_perf, (int, float)) else None
    span["end_time_utc"] = _utc_now_iso()
    span["duration_ms"] = duration_ms
    span["status"] = status
    span["outputs_summary"] = dict(outputs_summary or {})
    span["warnings"] = _warning_objects(list(warnings or []))
    span["error"] = dict(error) if error else None


def finalize_trace(trace: dict[str, Any], *, active_family: str | None = None) -> dict[str, Any]:
    trace["active_family"] = active_family
    now_iso = _utc_now_iso()
    for stage in REQUIRED_STAGES:
        if _get_span(trace, stage) is None:
            trace["spans"].append(
                {
                    "span_id": f"sp_{stage}_{len(trace['spans']) + 1}",
                    "parent_span_id": None,
                    "span_name": stage.replace("_", " ").title(),
                    "stage": stage,
                    "start_time_utc": now_iso,
                    "end_time_utc": now_iso,
                    "duration_ms": 0,
                    "status": "skipped",
                    "inputs_summary": {},
                    "outputs_summary": {},
                    "warnings": [],
                    "error": None,
                }
            )
    statuses = [span.get("status") for span in trace.get("spans", [])]
    overall: TraceStatus = "success"
    if any(status == "failed" for status in statuses):
        overall = "failed"
    elif any(status in {"partial", "skipped"} for status in statuses):
        overall = "partial"
    trace["overall_status"] = overall
    started = trace.get("_start_perf")
    if isinstance(started, (int, float)):
        trace["total_latency_ms"] = int(round((perf_counter() - started) * 1000))
    for span in trace.get("spans", []):
        span.pop("_start_perf", None)
    trace.pop("_start_perf", None)
    return trace
