from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACE_STAGE_ORDER: tuple[str, ...] = (
    "classification",
    "rewrite",
    "decomposition",
    "retrieval",
    "rerank",
    "answerability",
    "final_answer",
)


@dataclass(frozen=True, slots=True)
class TraceRun:
    run_id: str
    source_path: Path
    cases: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class StageStatus:
    stage: str
    status: str
    reason: str | None = None


def discover_trace_run_files(run_dir: str | Path) -> list[Path]:
    directory = Path(run_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted([path for path in directory.glob("*.json") if path.is_file()])


def load_trace_runs(run_files: Sequence[str | Path]) -> list[TraceRun]:
    runs: list[TraceRun] = []
    for run_file in sorted(Path(item) for item in run_files):
        try:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(blob, Mapping):
            continue
        cases = [dict(case) for case in blob.get("cases", []) if isinstance(case, Mapping)]
        if not cases:
            continue
        run_id = str(blob.get("generated_at_utc") or blob.get("run_id") or run_file.stem)
        runs.append(TraceRun(run_id=run_id, source_path=run_file, cases=cases))
    return runs


def build_trace_drilldown(case: Mapping[str, Any]) -> dict[str, Any]:
    debug_payload = _as_mapping(case.get("debug_payload"))
    system_result = _as_mapping(case.get("system_result"))
    trace = _as_mapping(debug_payload.get("trace"))

    query_understanding = _span(trace, "query_understanding")
    decomposition_span = _span(trace, "decomposition")
    retrieval_span = _span(trace, "retrieval")
    rerank_span = _span(trace, "rerank")
    answerability_span = _span(trace, "answerability")
    final_span = _span(trace, "final_synthesis")

    classification = {
        "original_query": str(case.get("query") or trace.get("query") or ""),
        "normalized_query": _pick(
            query_understanding.get("outputs_summary", {}).get("normalized_query"),
            _as_mapping(debug_payload.get("query_classification")).get("normalized_query"),
        ),
        "question_type": _pick(
            query_understanding.get("outputs_summary", {}).get("question_type"),
            _as_mapping(debug_payload.get("query_classification")).get("question_type"),
        ),
        "legal_family": _pick(
            query_understanding.get("outputs_summary", {}).get("legal_question_family"),
            trace.get("active_family"),
            case.get("family"),
        ),
        "routing_notes": _pick_list(
            query_understanding.get("outputs_summary", {}).get("routing_notes"),
            _as_mapping(debug_payload.get("query_classification")).get("routing_notes"),
        ),
        "is_followup": _pick(
            query_understanding.get("outputs_summary", {}).get("is_followup"),
            _as_mapping(debug_payload.get("query_classification")).get("is_followup"),
        ),
        "is_document_scoped": _pick(
            query_understanding.get("outputs_summary", {}).get("is_document_scoped"),
            _as_mapping(debug_payload.get("query_classification")).get("is_document_scoped"),
        ),
        "is_context_dependent": _pick(
            _as_mapping(debug_payload.get("query_classification")).get("is_context_dependent"),
        ),
    }

    rewrite = {
        "resolved_query": debug_payload.get("resolved_query"),
        "effective_query": _pick(
            debug_payload.get("effective_query"),
            retrieval_span.get("outputs_summary", {}).get("effective_query"),
        ),
        "rewrite_occurred": _rewrite_occurred(debug_payload),
        "notes": _pick_list(
            _as_mapping(debug_payload.get("rewritten_query")).get("rewrite_notes"),
        ),
        "warnings": _warning_messages(_span(trace, "query_understanding"), code_filter="rewrite"),
    }

    decomposition_debug = _as_mapping(debug_payload.get("decomposition"))
    decomposition = {
        "needs_decomposition": _pick(
            decomposition_span.get("outputs_summary", {}).get("needs_decomposition"),
            decomposition_debug.get("needs_decomposition"),
        ),
        "decomposition_reasons": _pick_list(
            decomposition_span.get("outputs_summary", {}).get("decomposition_gate_reasons"),
            decomposition_debug.get("decomposition_gate_reasons"),
        ),
        "plan_summary": _pick(
            decomposition_span.get("outputs_summary", {}).get("strategy"),
        ),
        "subquery_ids": _pick_list(decomposition_span.get("outputs_summary", {}).get("subquery_ids")),
        "validation_result": _pick(decomposition_span.get("outputs_summary", {}).get("validation_outcome")),
    }

    retrieval = {
        "selected_document_scope": _pick_list(
            retrieval_span.get("outputs_summary", {}).get("selected_document_scope"),
            case.get("selected_document_ids"),
        ),
        "retrieved_child_count": _pick(retrieval_span.get("outputs_summary", {}).get("retrieved_child_count")),
        "top_child_chunk_ids": _pick_list(retrieval_span.get("outputs_summary", {}).get("top_child_chunk_ids")),
        "retrieval_mode": retrieval_span.get("outputs_summary", {}).get("retrieval_mode"),
        "warnings": _warning_messages(retrieval_span),
    }

    rerank = {
        "input_candidate_count": _pick(rerank_span.get("outputs_summary", {}).get("input_candidate_count"), 0),
        "output_candidate_count": _pick(rerank_span.get("outputs_summary", {}).get("output_candidate_count")),
        "top_reranked_child_ids": _pick_list(rerank_span.get("outputs_summary", {}).get("top_reranked_child_ids")),
        "ranking_source": rerank_span.get("outputs_summary", {}).get("ranking_source"),
        "warnings": _warning_messages(rerank_span),
    }

    answerability_debug = _as_mapping(debug_payload.get("answerability_result"))
    answerability = {
        "sufficient_context": _pick(
            answerability_span.get("outputs_summary", {}).get("sufficient_context"),
            answerability_debug.get("sufficient_context"),
        ),
        "support_level": _pick(
            answerability_span.get("outputs_summary", {}).get("support_level"),
            answerability_debug.get("support_level"),
        ),
        "should_answer": _pick(
            answerability_span.get("outputs_summary", {}).get("should_answer"),
            answerability_debug.get("should_answer"),
        ),
        "insufficiency_reason": _pick(
            answerability_span.get("outputs_summary", {}).get("insufficiency_reason"),
            answerability_debug.get("insufficiency_reason"),
        ),
        "evidence_notes": _pick_list(answerability_debug.get("evidence_notes")),
        "matched_headings": _pick_list(
            answerability_span.get("outputs_summary", {}).get("matched_headings"),
            answerability_debug.get("matched_headings"),
        ),
        "matched_parent_ids": _pick_list(
            answerability_span.get("outputs_summary", {}).get("matched_parent_chunk_ids"),
            answerability_debug.get("matched_parent_chunk_ids"),
        ),
    }

    final_answer = {
        "answer_text": str(system_result.get("answer_text") or ""),
        "grounded": system_result.get("grounded"),
        "sufficient_context": system_result.get("sufficient_context"),
        "final_status": _pick(final_span.get("outputs_summary", {}).get("final_output_status"), case.get("runner_status")),
        "warnings": [str(item) for item in system_result.get("warnings", []) if str(item).strip()],
    }

    citations = [
        {
            "source_name": str(citation.get("source_name") or "Unknown source"),
            "heading": citation.get("heading"),
            "supporting_excerpt": citation.get("supporting_excerpt"),
        }
        for citation in system_result.get("citations", [])
        if isinstance(citation, Mapping)
    ]

    warnings = _group_warnings(trace=trace, debug_payload=debug_payload, final_answer=final_answer)
    stage_statuses = _build_stage_statuses(
        query_understanding=query_understanding,
        rewrite=rewrite,
        decomposition_span=decomposition_span,
        retrieval=retrieval,
        rerank=rerank,
        answerability=answerability,
        final_answer=final_answer,
    )

    return {
        "case_id": str(case.get("case_id") or case.get("id") or "unknown_case"),
        "family": str(case.get("family") or classification["legal_family"] or "unknown"),
        "classification": classification,
        "rewrite": rewrite,
        "decomposition": decomposition,
        "retrieval": retrieval,
        "rerank": rerank,
        "answerability": answerability,
        "final_answer": final_answer,
        "citations": citations,
        "citation_count": len(citations),
        "warnings": warnings,
        "stage_statuses": [asdict(status) for status in stage_statuses],
        "failure_layer": _first_non_ok_stage(stage_statuses),
        "raw": {"case": dict(case), "trace": trace, "debug_payload": debug_payload},
    }


def _group_warnings(*, trace: Mapping[str, Any], debug_payload: Mapping[str, Any], final_answer: Mapping[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for span in trace.get("spans", []):
        if not isinstance(span, Mapping):
            continue
        stage = str(span.get("stage") or "unknown")
        messages = _warning_messages(span)
        if messages:
            grouped[stage] = messages
    payload_warnings = [str(item) for item in debug_payload.get("warnings", []) if str(item).strip()]
    if payload_warnings:
        grouped["pipeline"] = payload_warnings
    if final_answer.get("warnings"):
        grouped["final_answer"] = [str(item) for item in final_answer.get("warnings", [])]
    return grouped


def _build_stage_statuses(
    *,
    query_understanding: Mapping[str, Any],
    rewrite: Mapping[str, Any],
    decomposition_span: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    rerank: Mapping[str, Any],
    answerability: Mapping[str, Any],
    final_answer: Mapping[str, Any],
) -> list[StageStatus]:
    statuses: list[StageStatus] = []
    statuses.append(_status_from_span("classification", query_understanding))

    rewrite_notes = rewrite.get("warnings") or rewrite.get("notes") or []
    if rewrite_notes:
        statuses.append(StageStatus(stage="rewrite", status="warning", reason="rewrite notes/warnings present"))
    else:
        statuses.append(StageStatus(stage="rewrite", status="ok"))

    statuses.append(_status_from_span("decomposition", decomposition_span))

    retrieved_child_count = _safe_count_value(retrieval.get("retrieved_child_count"))
    if retrieved_child_count is None:
        statuses.append(
            StageStatus(
                stage="retrieval",
                status="warning",
                reason="retrieved_child_count not available",
            )
        )
    elif retrieved_child_count <= 0:
        statuses.append(StageStatus(stage="retrieval", status="suspicious", reason="no retrieved evidence"))
    elif retrieval.get("warnings"):
        statuses.append(StageStatus(stage="retrieval", status="warning", reason="retrieval warnings"))
    else:
        statuses.append(StageStatus(stage="retrieval", status="ok"))

    output_candidate_count = _safe_count_value(rerank.get("output_candidate_count"))
    if output_candidate_count is None:
        statuses.append(
            StageStatus(
                stage="rerank",
                status="warning",
                reason="output_candidate_count not available",
            )
        )
    elif output_candidate_count <= 0:
        statuses.append(StageStatus(stage="rerank", status="suspicious", reason="no reranked evidence"))
    elif rerank.get("warnings"):
        statuses.append(StageStatus(stage="rerank", status="warning", reason="rerank warnings"))
    else:
        statuses.append(StageStatus(stage="rerank", status="ok"))

    if answerability.get("sufficient_context") is False or answerability.get("should_answer") is False:
        statuses.append(StageStatus(stage="answerability", status="suspicious", reason="answerability rejected context"))
    else:
        statuses.append(StageStatus(stage="answerability", status="ok"))

    if final_answer.get("grounded") is False:
        statuses.append(StageStatus(stage="final_answer", status="failed", reason="final answer is ungrounded"))
    elif final_answer.get("warnings"):
        statuses.append(StageStatus(stage="final_answer", status="warning", reason="final answer warnings"))
    else:
        statuses.append(StageStatus(stage="final_answer", status="ok"))
    return statuses


def _status_from_span(stage: str, span: Mapping[str, Any]) -> StageStatus:
    if not span:
        return StageStatus(stage=stage, status="not_available", reason="stage not present")
    status = str(span.get("status") or "unknown")
    if status in {"failed"}:
        return StageStatus(stage=stage, status="failed", reason="stage failed")
    if _warning_messages(span):
        return StageStatus(stage=stage, status="warning", reason="stage warnings")
    if status in {"partial", "skipped"}:
        return StageStatus(stage=stage, status="suspicious", reason=f"stage status={status}")
    return StageStatus(stage=stage, status="ok")


def _first_non_ok_stage(statuses: Sequence[StageStatus]) -> dict[str, Any] | None:
    for status in statuses:
        if status.status not in {"ok"}:
            return {"stage": status.stage, "status": status.status, "reason": status.reason}
    return None


def _span(trace: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    spans = trace.get("spans") if isinstance(trace, Mapping) else []
    if not isinstance(spans, list):
        return {}
    for span in spans:
        if isinstance(span, Mapping) and str(span.get("stage")) == stage:
            return span
    return {}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pick(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _pick_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return [item for item in value]
    return []


def _rewrite_occurred(debug_payload: Mapping[str, Any]) -> bool | None:
    resolved = debug_payload.get("resolved_query")
    effective = debug_payload.get("effective_query")
    if isinstance(resolved, str) and isinstance(effective, str):
        return resolved.strip() != effective.strip()
    return None


def _warning_messages(span: Mapping[str, Any], code_filter: str | None = None) -> list[str]:
    values = span.get("warnings") if isinstance(span, Mapping) else []
    if not isinstance(values, list):
        return []
    messages: list[str] = []
    for warning in values:
        if isinstance(warning, Mapping):
            code = str(warning.get("code") or "")
            message = str(warning.get("message") or code)
        else:
            code = str(warning)
            message = code
        if code_filter and code_filter not in code:
            continue
        if message:
            messages.append(message)
    return messages


def _safe_count_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer() or value < 0:
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"n/a", "na", "none", "null"}:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None
