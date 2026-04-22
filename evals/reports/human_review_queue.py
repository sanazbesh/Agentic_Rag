from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.reports.triage_workflow import REGRESSION_DRAFTS_PATH, discover_eval_run_files

REVIEW_STORE_PATH = Path("data/review/human_review_records.json")
SAMPLED_TRAFFIC_PATH = Path("data/sampling/production_traffic_samples.jsonl")
SHADOW_EVAL_PATH = Path("data/evals/online_shadow_eval_results.jsonl")

REVIEW_CATEGORIES: tuple[str, ...] = (
    "suspected_hallucination",
    "missing_citation",
    "family_routing_disagreement",
    "legal_issue_spotting_disagreement",
)
REVIEW_ACTIONS: tuple[str, ...] = ("approve", "reject", "relabel", "escalate")
REVIEW_STATUSES: tuple[str, ...] = (
    "pending_review",
    "approved",
    "rejected",
    "relabeled",
    "escalated",
    "resolved",
)
DATASET_FEEDBACK_ACTIONS: tuple[str, ...] = (
    "none",
    "no_dataset_update",
    "needs_regression_case",
    "regression_case_draft_created",
    "linked_existing_regression_case",
)


@dataclass(frozen=True, slots=True)
class ReviewQueueItem:
    review_item_id: str
    source_type: str
    source_id: str
    run_id: str
    run_file: str | None
    query: str
    family: str
    selected_document_ids: list[str]
    final_answer: str
    citations: list[dict[str, Any]]
    warnings: list[str]
    trace_id: str | None
    trigger_reasons: list[str]
    category: str
    review_status: str


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    review_item_id: str
    source_id: str
    review_status: str
    review_action: str
    family: str
    failure_taxonomy: str | None
    severity: str | None
    reviewer_note: str
    dataset_feedback_action: str
    trace_id: str | None
    regression_case_id: str | None
    regression_dataset_file: str | None
    relabeled_family: str | None
    relabeled_category: str | None
    fixed_version: str | None
    created_at_utc: str
    updated_at_utc: str


def load_review_queue(
    *,
    run_dir: str | Path = "evals/runs",
    review_store_path: str | Path = REVIEW_STORE_PATH,
    sampled_traffic_path: str | Path = SAMPLED_TRAFFIC_PATH,
    shadow_eval_path: str | Path = SHADOW_EVAL_PATH,
) -> list[ReviewQueueItem]:
    records = load_review_records(review_store_path)
    sampled_by_trace = _load_sampled_traffic_index(sampled_traffic_path)
    queue: dict[str, ReviewQueueItem] = {}

    for run_file in discover_eval_run_files(run_dir):
        for item in _load_run_file_candidates(run_file):
            queue[item.review_item_id] = _merge_status(item, records)

    for item in _load_shadow_candidates(shadow_eval_path, sampled_by_trace):
        queue.setdefault(item.review_item_id, _merge_status(item, records))

    rows = list(queue.values())
    rows.sort(key=lambda row: (0 if row.review_status == "pending_review" else 1, row.category, row.review_item_id))
    return rows


def load_review_records(path: str | Path = REVIEW_STORE_PATH) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        review_item_id = str(record.get("review_item_id") or "").strip()
        if review_item_id:
            indexed[review_item_id] = dict(record)
    return indexed


def save_review_record(record: ReviewRecord, path: str | Path = REVIEW_STORE_PATH) -> None:
    p = Path(path)
    existing = load_review_records(p)
    existing[record.review_item_id] = asdict(record)
    payload = {
        "schema_version": 1,
        "updated_at_utc": _utc_now_iso(),
        "records": [existing[key] for key in sorted(existing.keys())],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_review_decision(
    *,
    item: ReviewQueueItem,
    action: str,
    reviewer_note: str,
    review_store_path: str | Path = REVIEW_STORE_PATH,
    family: str | None = None,
    category: str | None = None,
    failure_taxonomy: str | None = None,
    severity: str | None = None,
    dataset_feedback_action: str = "none",
    regression_case_id: str | None = None,
    regression_dataset_file: str | None = None,
    fixed_version: str | None = None,
    create_regression_draft: bool = False,
    regression_drafts_path: str | Path = REGRESSION_DRAFTS_PATH,
) -> ReviewRecord:
    stable_action = action if action in REVIEW_ACTIONS else "reject"
    now = _utc_now_iso()
    existing = load_review_records(review_store_path).get(item.review_item_id, {})
    created_at = str(existing.get("created_at_utc") or now)

    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "relabel": "relabeled",
        "escalate": "escalated",
    }
    relabeled_family = family if stable_action == "relabel" and family else None
    relabeled_category = category if stable_action == "relabel" and category else None

    resolved_feedback = dataset_feedback_action if dataset_feedback_action in DATASET_FEEDBACK_ACTIONS else "none"
    if create_regression_draft:
        draft = build_review_regression_case_draft(item=item, action=stable_action, reviewer_note=reviewer_note)
        append_regression_case_draft_from_review(draft, path=regression_drafts_path)
        resolved_feedback = "regression_case_draft_created"

    record = ReviewRecord(
        review_item_id=item.review_item_id,
        source_id=item.source_id,
        review_status=status_map[stable_action],
        review_action=stable_action,
        family=family or item.family,
        failure_taxonomy=_blank_to_none(failure_taxonomy),
        severity=_blank_to_none(severity),
        reviewer_note=reviewer_note.strip(),
        dataset_feedback_action=resolved_feedback,
        trace_id=item.trace_id,
        regression_case_id=_blank_to_none(regression_case_id),
        regression_dataset_file=_blank_to_none(regression_dataset_file),
        relabeled_family=_blank_to_none(relabeled_family),
        relabeled_category=_blank_to_none(relabeled_category),
        fixed_version=_blank_to_none(fixed_version),
        created_at_utc=created_at,
        updated_at_utc=now,
    )
    save_review_record(record, path=review_store_path)
    return record


def build_review_regression_case_draft(*, item: ReviewQueueItem, action: str, reviewer_note: str) -> dict[str, Any]:
    base_id = f"reg.review.{item.family}.{item.source_id}".replace(":", "-").replace("_", "-")
    return {
        "id": base_id,
        "family": item.family,
        "query": item.query,
        "selected_document_ids": item.selected_document_ids,
        "expected_answer_type": "safe_failure_response",
        "expected_outcome": "safe_failure_insufficient_evidence",
        "answerability_expected": "unanswerable",
        "gold_evidence_ids": [],
        "evidence_requirement": "none",
        "safe_failure_expected": True,
        "difficulty": "medium",
        "tags": ["tier:regression", f"review_action:{action}", f"review_item:{item.review_item_id}"],
        "regression_source": f"human_review:{item.review_item_id}",
        "notes": reviewer_note or f"Drafted from {item.category} review queue item.",
    }


def append_regression_case_draft_from_review(draft: Mapping[str, Any], path: str | Path = REGRESSION_DRAFTS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(draft), ensure_ascii=False) + "\n")


def _load_run_file_candidates(run_file: Path) -> list[ReviewQueueItem]:
    try:
        blob = json.loads(run_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(blob, Mapping):
        return []

    run_id = str(blob.get("generated_at_utc") or blob.get("run_id") or run_file.stem)
    queue: list[ReviewQueueItem] = []
    for raw_case in blob.get("cases", []):
        if not isinstance(raw_case, Mapping):
            continue
        case = dict(raw_case)
        categories = _extract_sensitive_categories(case)
        for category in categories:
            case_id = str(case.get("case_id") or case.get("id") or "unknown_case")
            source_id = f"{run_id}:{case_id}"
            review_item_id = f"review:{category}:{source_id}"
            final_result = case.get("system_output") if isinstance(case.get("system_output"), Mapping) else {}
            citations = final_result.get("citations") if isinstance(final_result.get("citations"), list) else []
            warnings = final_result.get("warnings") if isinstance(final_result.get("warnings"), list) else []
            queue.append(
                ReviewQueueItem(
                    review_item_id=review_item_id,
                    source_type="offline_eval_run",
                    source_id=source_id,
                    run_id=run_id,
                    run_file=str(run_file),
                    query=str(case.get("query") or ""),
                    family=str(case.get("family") or "unknown"),
                    selected_document_ids=_extract_selected_document_ids(case),
                    final_answer=str(final_result.get("answer_text") or case.get("final_answer") or ""),
                    citations=[row for row in citations if isinstance(row, Mapping)],
                    warnings=[str(w) for w in warnings if isinstance(w, str)],
                    trace_id=_extract_trace_id(case),
                    trigger_reasons=[category],
                    category=category,
                    review_status="pending_review",
                )
            )
    return queue


def _load_sampled_traffic_index(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        trace_id = _blank_to_none(record.get("trace_id"))
        if trace_id:
            indexed[trace_id] = dict(record)
    return indexed


def _load_shadow_candidates(path: str | Path, sampled_by_trace: Mapping[str, Mapping[str, Any]]) -> list[ReviewQueueItem]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    queue: list[ReviewQueueItem] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        categories = _extract_shadow_categories(record)
        if not categories:
            continue

        trace_id = _blank_to_none(record.get("trace_id"))
        sample = sampled_by_trace.get(trace_id or "", {})
        final_result = sample.get("final_result") if isinstance(sample.get("final_result"), Mapping) else {}
        source_id = str(record.get("shadow_eval_id") or "unknown_shadow_eval")
        for category in categories:
            review_item_id = f"review:{category}:{source_id}"
            queue.append(
                ReviewQueueItem(
                    review_item_id=review_item_id,
                    source_type="online_shadow_eval",
                    source_id=source_id,
                    run_id=str(record.get("timestamp_utc") or "unknown_run"),
                    run_file=None,
                    query=str(sample.get("query") or ""),
                    family=str(sample.get("family") or record.get("family") or "unknown"),
                    selected_document_ids=[
                        str(v)
                        for v in (sample.get("selected_document_ids") if isinstance(sample.get("selected_document_ids"), list) else [])
                    ],
                    final_answer=str(final_result.get("answer_text") or ""),
                    citations=[row for row in (final_result.get("citations") or []) if isinstance(row, Mapping)],
                    warnings=[str(w) for w in (final_result.get("warnings") or []) if isinstance(w, str)],
                    trace_id=trace_id,
                    trigger_reasons=[category],
                    category=category,
                    review_status="pending_review",
                )
            )
    return queue


def _extract_sensitive_categories(case: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    deterministic = case.get("deterministic_eval_results") if isinstance(case.get("deterministic_eval_results"), Mapping) else {}
    llm_judges = case.get("llm_judge_results") if isinstance(case.get("llm_judge_results"), Mapping) else {}
    final_result = case.get("system_output") if isinstance(case.get("system_output"), Mapping) else {}

    citation_check = deterministic.get("citation_checks") if isinstance(deterministic.get("citation_checks"), Mapping) else {}
    if _is_failed_result(citation_check) or (isinstance(final_result.get("citations"), list) and len(final_result.get("citations") or []) == 0):
        categories.append("missing_citation")

    family_routing = deterministic.get("family_routing") if isinstance(deterministic.get("family_routing"), Mapping) else {}
    if _is_failed_result(family_routing):
        categories.append("family_routing_disagreement")

    groundedness = llm_judges.get("groundedness") if isinstance(llm_judges.get("groundedness"), Mapping) else {}
    if _is_failed_result(groundedness) or any(str(w).lower().find("hallucin") >= 0 for w in (final_result.get("warnings") or [])):
        categories.append("suspected_hallucination")

    family = str(case.get("family") or "")
    if family == "policy_issue_spotting":
        if str(case.get("runner_status") or "") != "ok" or any(_is_failed_result(result) for result in list(deterministic.values()) + list(llm_judges.values()) if isinstance(result, Mapping)):
            categories.append("legal_issue_spotting_disagreement")

    return list(dict.fromkeys(category for category in categories if category in REVIEW_CATEGORIES))


def _extract_shadow_categories(record: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    deterministic = record.get("deterministic_results") if isinstance(record.get("deterministic_results"), Mapping) else {}
    model = record.get("model_results") if isinstance(record.get("model_results"), Mapping) else {}

    family_routing = deterministic.get("family_routing") if isinstance(deterministic.get("family_routing"), Mapping) else {}
    if str(family_routing.get("status") or "") in {"error"}:
        categories.append("family_routing_disagreement")

    groundedness = model.get("groundedness") if isinstance(model.get("groundedness"), Mapping) else {}
    if str(groundedness.get("status") or "") in {"error"}:
        categories.append("suspected_hallucination")

    return list(dict.fromkeys(categories))


def _extract_selected_document_ids(case: Mapping[str, Any]) -> list[str]:
    debug_payload = case.get("debug_payload") if isinstance(case.get("debug_payload"), Mapping) else {}
    trace = debug_payload.get("trace") if isinstance(debug_payload.get("trace"), Mapping) else {}
    selected_ids = trace.get("selected_document_ids") if isinstance(trace.get("selected_document_ids"), list) else []
    return [str(item) for item in selected_ids if str(item).strip()]


def _extract_trace_id(case: Mapping[str, Any]) -> str | None:
    debug_payload = case.get("debug_payload")
    if not isinstance(debug_payload, Mapping):
        return None
    trace = debug_payload.get("trace")
    if not isinstance(trace, Mapping):
        return None
    return _blank_to_none(trace.get("trace_id"))


def _is_failed_result(result: Mapping[str, Any]) -> bool:
    status = str(result.get("status") or "")
    if status == "error":
        return True
    for key in ("passed", "is_correct"):
        if isinstance(result.get(key), bool) and result.get(key) is False:
            return True
    return False


def _merge_status(item: ReviewQueueItem, records: Mapping[str, Mapping[str, Any]]) -> ReviewQueueItem:
    record = records.get(item.review_item_id, {})
    status = str(record.get("review_status") or item.review_status)
    if status not in REVIEW_STATUSES:
        status = "pending_review"
    return ReviewQueueItem(
        review_item_id=item.review_item_id,
        source_type=item.source_type,
        source_id=item.source_id,
        run_id=item.run_id,
        run_file=item.run_file,
        query=item.query,
        family=item.family,
        selected_document_ids=list(item.selected_document_ids),
        final_answer=item.final_answer,
        citations=list(item.citations),
        warnings=list(item.warnings),
        trace_id=item.trace_id,
        trigger_reasons=list(item.trigger_reasons),
        category=item.category,
        review_status=status,
    )


def _blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "DATASET_FEEDBACK_ACTIONS",
    "REVIEW_ACTIONS",
    "REVIEW_CATEGORIES",
    "REVIEW_STATUSES",
    "REVIEW_STORE_PATH",
    "ReviewQueueItem",
    "ReviewRecord",
    "apply_review_decision",
    "build_review_regression_case_draft",
    "load_review_queue",
    "load_review_records",
    "save_review_record",
]
