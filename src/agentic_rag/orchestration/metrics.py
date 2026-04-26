"""Structured per-request metrics for local-first legal RAG observability.

Ticket 21 scope:
- emit one structured metrics record per request,
- keep deterministic, in-repo aggregation helpers,
- no external metrics vendor/backend dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field


class RequestMetricsRecord(BaseModel):
    """One deterministic metrics record emitted for one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str | None = None
    trace_id: str | None = None

    legal_family: str | None = None
    document_type: str | None = None

    grounded_answer: int = 0
    insufficient_answer: int = 0
    false_confident_proxy: int = 0
    family_route_present: int = 0

    latency_ms: int | None = None

    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    model_version: str | None = None
    pipeline_mode: str | None = None
    selected_document_count: int | None = None
    decomposition_used: bool | None = None


class MetricsAggregate(BaseModel):
    """Rollup summary derived from request metrics records."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int = 0

    grounded_answer_rate: float = 0.0
    insufficient_answer_rate: float = 0.0
    false_confident_proxy_rate: float = 0.0
    family_routing_rate: float = 0.0

    p95_latency_ms: int | None = None
    avg_cost_usd: float | None = None

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0


def _extract_family(state: Mapping[str, Any]) -> str | None:
    trace = state.get("trace")
    if isinstance(trace, Mapping):
        active_family = trace.get("active_family")
        if isinstance(active_family, str) and active_family:
            return active_family

    query_classification = state.get("query_classification")
    routing_notes = getattr(query_classification, "routing_notes", [])
    if isinstance(routing_notes, Sequence):
        for note in routing_notes:
            if isinstance(note, str) and note.startswith("legal_question_family:"):
                family = note.split(":", 1)[1].strip()
                if family:
                    return family
    return None


def _extract_document_type(state: Mapping[str, Any]) -> str | None:
    for parent in state.get("parent_chunks", []) or []:
        metadata = getattr(parent, "metadata", None)
        if isinstance(parent, Mapping):
            metadata = parent.get("metadata", metadata)
        if isinstance(metadata, Mapping):
            doc_type = metadata.get("document_type")
            if isinstance(doc_type, str) and doc_type.strip():
                return doc_type.strip()

    for selected in state.get("selected_documents", []) or []:
        doc_type = selected.get("type") if isinstance(selected, Mapping) else getattr(selected, "type", None)
        if isinstance(doc_type, str) and doc_type.strip():
            return doc_type.strip()
    return None


def _extract_cost_usd(state: Mapping[str, Any]) -> float | None:
    for key in ("cost_usd", "request_cost_usd", "total_cost_usd"):
        value = state.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_token_usage(state: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = state.get("token_usage")
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        input_tokens = state.get("input_tokens")
        output_tokens = state.get("output_tokens")
        total_tokens = state.get("total_tokens")

    in_tok = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    out_tok = int(output_tokens) if isinstance(output_tokens, (int, float)) else None
    total_tok = int(total_tokens) if isinstance(total_tokens, (int, float)) else None
    if total_tok is None and (in_tok is not None or out_tok is not None):
        total_tok = (in_tok or 0) + (out_tok or 0)
    return in_tok, out_tok, total_tok


def emit_request_metrics(*, final_answer: Any, state: Mapping[str, Any]) -> RequestMetricsRecord:
    """Build one structured metrics record for one request.

    False-confident proxy logic is deterministic by design (proxy only, not a
    semantic hallucination detector):
    - grounded answer with zero citations,
    - answerability says should_answer=False while final answer appears grounded,
    - sufficient_context=True while the final route is insufficient.
    """

    grounded = bool(getattr(final_answer, "grounded", False))
    sufficient_context = bool(getattr(final_answer, "sufficient_context", False))
    citations = list(getattr(final_answer, "citations", []) or [])
    citation_count = len(citations)

    answerability = state.get("answerability_result")
    should_answer = getattr(answerability, "should_answer", None)
    route = str(state.get("response_route") or "")

    insufficient = int(not sufficient_context)
    proxy = int(
        (grounded and citation_count == 0)
        or (bool(should_answer) and sufficient_context and citation_count == 0)
        or (should_answer is False and grounded)
        or (route.startswith("insufficient") and sufficient_context)
    )

    trace = state.get("trace")
    request_id = trace.get("request_id") if isinstance(trace, Mapping) else None
    trace_id = trace.get("trace_id") if isinstance(trace, Mapping) else None
    latency_ms = trace.get("total_latency_ms") if isinstance(trace, Mapping) else None
    stable_latency = int(latency_ms) if isinstance(latency_ms, (int, float)) else None

    family = _extract_family(state)
    doc_type = _extract_document_type(state)
    cost = _extract_cost_usd(state)
    input_tokens, output_tokens, total_tokens = _extract_token_usage(state)

    return RequestMetricsRecord(
        request_id=request_id if isinstance(request_id, str) else None,
        trace_id=trace_id if isinstance(trace_id, str) else None,
        legal_family=family,
        document_type=doc_type,
        grounded_answer=int(grounded),
        insufficient_answer=insufficient,
        false_confident_proxy=proxy,
        family_route_present=int(bool(family)),
        latency_ms=stable_latency,
        cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        model_version=(trace.get("model_version") if isinstance(trace, Mapping) else None),
        pipeline_mode="local",
        selected_document_count=len(list(state.get("selected_documents", []) or [])),
        decomposition_used=bool(state.get("needs_decomposition", False)),
    )


def _safe_rate(total: int, numerator: int) -> float:
    if total <= 0:
        return 0.0
    return numerator / total


def _p95_latency(values: Sequence[int]) -> int | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, ceil(0.95 * len(sorted_values)) - 1)
    return int(sorted_values[index])


def aggregate_metrics(records: Sequence[RequestMetricsRecord]) -> MetricsAggregate:
    """Aggregate request metrics with percentile-ready latency rollup."""

    total = len(records)
    grounded_sum = sum(item.grounded_answer for item in records)
    insufficient_sum = sum(item.insufficient_answer for item in records)
    false_confident_sum = sum(item.false_confident_proxy for item in records)
    family_routed_sum = sum(item.family_route_present for item in records)

    latency_values = [item.latency_ms for item in records if isinstance(item.latency_ms, int)]
    cost_values = [item.cost_usd for item in records if isinstance(item.cost_usd, float)]

    return MetricsAggregate(
        request_count=total,
        grounded_answer_rate=_safe_rate(total, grounded_sum),
        insufficient_answer_rate=_safe_rate(total, insufficient_sum),
        false_confident_proxy_rate=_safe_rate(total, false_confident_sum),
        family_routing_rate=_safe_rate(total, family_routed_sum),
        p95_latency_ms=_p95_latency(latency_values),
        avg_cost_usd=(sum(cost_values) / len(cost_values)) if cost_values else None,
        total_input_tokens=sum(item.input_tokens or 0 for item in records),
        total_output_tokens=sum(item.output_tokens or 0 for item in records),
        total_tokens=sum(item.total_tokens or 0 for item in records),
    )


def aggregate_metrics_by_family(records: Sequence[RequestMetricsRecord]) -> dict[str, MetricsAggregate]:
    grouped: dict[str, list[RequestMetricsRecord]] = defaultdict(list)
    for item in records:
        grouped[item.legal_family or "unknown"].append(item)
    return {family: aggregate_metrics(items) for family, items in grouped.items()}


def aggregate_metrics_by_document_type(records: Sequence[RequestMetricsRecord]) -> dict[str, MetricsAggregate]:
    grouped: dict[str, list[RequestMetricsRecord]] = defaultdict(list)
    for item in records:
        grouped[item.document_type or "unknown"].append(item)
    return {doc_type: aggregate_metrics(items) for doc_type, items in grouped.items()}
