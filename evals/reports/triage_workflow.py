from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SEVERITY_OPTIONS: tuple[str, ...] = ("P0", "P1", "P2", "P3")
FAMILY_OPTIONS: tuple[str, ...] = (
    "party_role_verification",
    "chronology_date_event",
    "employment_lifecycle",
    "matter_document_metadata",
    "correspondence_litigation_milestones",
    "employment_mitigation",
    "financial_entitlement",
    "policy_issue_spotting",
)
FAILURE_TAXONOMY_OPTIONS: tuple[str, ...] = (
    "family_misclassification",
    "retrieval_miss",
    "rerank_miss",
    "answerability_false_positive",
    "answerability_false_negative",
    "citation_failure",
    "synthesis_failure",
    "safe_failure_wording_issue",
    "document_scope_resolution_failure",
    "evidence_selection_failure",
    "output_contract_failure",
)

TRIAGE_STORE_PATH = Path("data/triage/triage_records.json")
REGRESSION_DRAFTS_PATH = Path("data/triage/regression_case_drafts.jsonl")


@dataclass(frozen=True, slots=True)
class TriageQueueItem:
    issue_id: str
    run_id: str
    run_file: str
    case_id: str
    query: str
    family: str
    runner_status: str
    status: str
    trace_id: str | None
    dataset_file: str | None
    failure_summary: str


@dataclass(frozen=True, slots=True)
class TriageRecord:
    issue_id: str
    run_id: str
    run_file: str
    case_id: str
    trace_id: str | None
    dataset_file: str | None
    query: str
    family: str
    severity: str
    failure_taxonomy: str
    reproduced: bool
    fixed_version: str | None
    regression_case_id: str | None
    regression_dataset_file: str | None
    notes: str
    status: str
    updated_at_utc: str


def discover_eval_run_files(run_dir: str | Path) -> list[Path]:
    directory = Path(run_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def load_triage_queue(run_files: Sequence[str | Path], triage_store_path: str | Path = TRIAGE_STORE_PATH) -> list[TriageQueueItem]:
    triage_records = load_triage_records(triage_store_path)
    queue: list[TriageQueueItem] = []

    for run_file in sorted(Path(item) for item in run_files):
        try:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(blob, Mapping):
            continue

        run_id = str(blob.get("generated_at_utc") or blob.get("run_id") or run_file.stem)
        for raw_case in blob.get("cases", []):
            if not isinstance(raw_case, Mapping):
                continue
            case = dict(raw_case)
            if not _is_failure_candidate(case):
                continue

            case_id = str(case.get("case_id") or case.get("id") or "unknown_case")
            issue_id = f"{run_id}:{case_id}"
            trace_id = _extract_trace_id(case)
            summary = _summarize_failure(case)
            triaged = triage_records.get(issue_id)

            queue.append(
                TriageQueueItem(
                    issue_id=issue_id,
                    run_id=run_id,
                    run_file=str(run_file),
                    case_id=case_id,
                    query=str(case.get("query") or ""),
                    family=str(case.get("family") or "unknown"),
                    runner_status=str(case.get("runner_status") or "unknown"),
                    status=str((triaged or {}).get("status") or "untriaged"),
                    trace_id=trace_id,
                    dataset_file=_safe_optional_str(case.get("_dataset_file")),
                    failure_summary=summary,
                )
            )

    queue.sort(key=lambda row: (0 if row.status == "untriaged" else 1, row.run_id, row.case_id))
    return queue


def load_triage_records(path: str | Path = TRIAGE_STORE_PATH) -> dict[str, dict[str, Any]]:
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
        issue_id = str(record.get("issue_id") or "").strip()
        if issue_id:
            indexed[issue_id] = dict(record)
    return indexed


def save_triage_record(record: TriageRecord, path: str | Path = TRIAGE_STORE_PATH) -> None:
    p = Path(path)
    existing = load_triage_records(p)
    existing[record.issue_id] = asdict(record)

    payload = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [existing[key] for key in sorted(existing.keys())],
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_regression_dataset_files(dataset_dir: str | Path = "evals/datasets") -> list[Path]:
    directory = Path(dataset_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("regressions*.jsonl") if path.is_file())


def find_existing_regression_case(item: TriageQueueItem, dataset_files: Sequence[str | Path]) -> tuple[str | None, str | None]:
    for dataset_file in [Path(path) for path in dataset_files]:
        try:
            lines = dataset_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, Mapping):
                continue
            if str(row.get("id") or "") == item.case_id:
                return str(row.get("id")), str(dataset_file)
            if str(row.get("query") or "") == item.query and str(row.get("family") or "") == item.family:
                return str(row.get("id") or ""), str(dataset_file)
    return None, None


def build_regression_case_draft(item: TriageQueueItem, triage_record: TriageRecord) -> dict[str, Any]:
    failure_tag = f"failure_class:{triage_record.failure_taxonomy}"
    base_id = f"reg.{item.family}.{item.case_id}".replace(":", "-").replace("_", "-")
    return {
        "id": base_id,
        "family": item.family,
        "query": item.query,
        "selected_document_ids": [],
        "expected_answer_type": "safe_failure_response",
        "expected_outcome": "safe_failure_insufficient_evidence",
        "answerability_expected": "unanswerable",
        "gold_evidence_ids": [],
        "evidence_requirement": "none",
        "safe_failure_expected": True,
        "difficulty": "medium",
        "tags": ["tier:regression", failure_tag, f"triage_issue:{item.issue_id}"],
        "regression_source": f"triage:{item.issue_id}",
        "notes": triage_record.notes or "Drafted from triaged production/offline failure.",
    }


def append_regression_case_draft(draft: Mapping[str, Any], path: str | Path = REGRESSION_DRAFTS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(draft), ensure_ascii=False) + "\n")


def _is_failure_candidate(case: Mapping[str, Any]) -> bool:
    if case.get("manual_flagged_bad_answer") is True:
        return True
    if str(case.get("runner_status") or "") != "ok":
        return True
    for group_key in ("deterministic_eval_results", "llm_judge_results"):
        group = case.get(group_key)
        if not isinstance(group, Mapping):
            continue
        for result in group.values():
            if not isinstance(result, Mapping):
                continue
            if result.get("status") == "error":
                return True
            if result.get("status") == "skipped":
                continue
            for key in ("passed", "is_correct"):
                if isinstance(result.get(key), bool) and result.get(key) is False:
                    return True
    return False


def _extract_trace_id(case: Mapping[str, Any]) -> str | None:
    debug_payload = case.get("debug_payload")
    if not isinstance(debug_payload, Mapping):
        return None
    trace = debug_payload.get("trace")
    if not isinstance(trace, Mapping):
        return None
    return _safe_optional_str(trace.get("trace_id"))


def _safe_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _summarize_failure(case: Mapping[str, Any]) -> str:
    if str(case.get("runner_status") or "") != "ok":
        return str(case.get("error") or "runner failed")
    failing_checks: list[str] = []
    for group_key in ("deterministic_eval_results", "llm_judge_results"):
        group = case.get(group_key)
        if not isinstance(group, Mapping):
            continue
        for name, result in group.items():
            if not isinstance(result, Mapping):
                continue
            if result.get("status") == "error":
                failing_checks.append(f"{group_key}.{name}:error")
                continue
            for key in ("passed", "is_correct"):
                if isinstance(result.get(key), bool) and result.get(key) is False:
                    failing_checks.append(f"{group_key}.{name}")
                    break
    if not failing_checks and case.get("manual_flagged_bad_answer") is True:
        return "manually flagged bad answer"
    return ", ".join(failing_checks) if failing_checks else "unknown failure"
