from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from evals.reports.triage_workflow import (
    TriageRecord,
    append_regression_case_draft,
    build_regression_case_draft,
    discover_eval_run_files,
    find_existing_regression_case,
    load_triage_queue,
    load_triage_records,
    save_triage_record,
)


def _failed_case(case_id: str = "case-1") -> dict:
    return {
        "case_id": case_id,
        "family": "party_role_verification",
        "query": "Who is the employer?",
        "runner_status": "ok",
        "_dataset_file": "evals/datasets/tier1_party_role.jsonl",
        "debug_payload": {"trace": {"trace_id": "trace-1"}},
        "deterministic_eval_results": {
            "citation_checks": {"status": "ok", "passed": False},
        },
    }


def _write_run(path: Path, case: dict) -> None:
    payload = {"generated_at_utc": "2026-04-21T00:00:00Z", "cases": [case]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_triage_queue_loads_failure_candidates_from_stored_outputs(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    _write_run(run_file, _failed_case())

    queue = load_triage_queue(discover_eval_run_files(tmp_path), triage_store_path=tmp_path / "triage.json")

    assert len(queue) == 1
    assert queue[0].issue_id.endswith(":case-1")
    assert queue[0].trace_id == "trace-1"


def test_issue_labeling_fields_save_with_stable_local_storage(tmp_path: Path) -> None:
    record = TriageRecord(
        issue_id="2026-04-21T00:00:00Z:case-1",
        run_id="2026-04-21T00:00:00Z",
        run_file="evals/runs/run.json",
        case_id="case-1",
        trace_id="trace-1",
        dataset_file="evals/datasets/tier1_party_role.jsonl",
        query="Who is the employer?",
        family="party_role_verification",
        severity="P1",
        failure_taxonomy="citation_failure",
        reproduced=True,
        fixed_version="retrieval.v2",
        regression_case_id=None,
        regression_dataset_file=None,
        notes="Reproduced locally from run artifact.",
        status="triaged",
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    store = tmp_path / "triage_records.json"
    save_triage_record(record, path=store)

    loaded = load_triage_records(store)
    assert record.issue_id in loaded
    stored = loaded[record.issue_id]
    assert stored["family"] == "party_role_verification"
    assert stored["severity"] == "P1"
    assert stored["failure_taxonomy"] == "citation_failure"
    assert stored["reproduced"] is True
    assert stored["fixed_version"] == "retrieval.v2"


def test_triage_records_distinguish_untriaged_and_triaged_items(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    _write_run(run_file, _failed_case())

    queue_before = load_triage_queue(discover_eval_run_files(tmp_path), triage_store_path=tmp_path / "triage.json")
    assert queue_before[0].status == "untriaged"

    record = TriageRecord(
        issue_id=queue_before[0].issue_id,
        run_id=queue_before[0].run_id,
        run_file=queue_before[0].run_file,
        case_id=queue_before[0].case_id,
        trace_id=queue_before[0].trace_id,
        dataset_file=queue_before[0].dataset_file,
        query=queue_before[0].query,
        family=queue_before[0].family,
        severity="P2",
        failure_taxonomy="retrieval_miss",
        reproduced=False,
        fixed_version=None,
        regression_case_id=None,
        regression_dataset_file=None,
        notes="needs repro",
        status="triaged",
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    store = tmp_path / "triage.json"
    save_triage_record(record, path=store)

    queue_after = load_triage_queue(discover_eval_run_files(tmp_path), triage_store_path=store)
    assert queue_after[0].status == "triaged"


def test_queue_handles_malformed_or_partial_items_without_crashing(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    payload = {
        "generated_at_utc": "2026-04-21T00:00:00Z",
        "cases": [
            _failed_case("good-case"),
            {"case_id": "bad-case", "runner_status": "failed"},
            "not-a-mapping",
        ],
    }
    run_file.write_text(json.dumps(payload), encoding="utf-8")

    queue = load_triage_queue(discover_eval_run_files(tmp_path), triage_store_path=tmp_path / "triage.json")
    issue_ids = {item.case_id for item in queue}
    assert "good-case" in issue_ids
    assert "bad-case" in issue_ids


def test_triage_links_to_existing_regression_case_and_can_create_draft(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    _write_run(run_file, _failed_case())
    queue = load_triage_queue(discover_eval_run_files(tmp_path), triage_store_path=tmp_path / "triage.json")
    item = queue[0]

    reg_dataset = tmp_path / "regressions_party_role.jsonl"
    reg_dataset.write_text(
        json.dumps(
            {
                "id": "reg.party.001",
                "family": "party_role_verification",
                "query": "Who is the employer?",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reg_id, reg_file = find_existing_regression_case(item, [reg_dataset])
    assert reg_id == "reg.party.001"
    assert reg_file == str(reg_dataset)

    record = TriageRecord(
        issue_id=item.issue_id,
        run_id=item.run_id,
        run_file=item.run_file,
        case_id=item.case_id,
        trace_id=item.trace_id,
        dataset_file=item.dataset_file,
        query=item.query,
        family=item.family,
        severity="P1",
        failure_taxonomy="answerability_false_positive",
        reproduced=True,
        fixed_version=None,
        regression_case_id=reg_id,
        regression_dataset_file=reg_file,
        notes="convert to regression",
        status="regressed",
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    draft = build_regression_case_draft(item, record)
    assert draft["family"] == item.family
    assert any(str(tag).startswith("triage_issue:") for tag in draft["tags"])

    drafts_path = tmp_path / "regression_case_drafts.jsonl"
    append_regression_case_draft(draft, path=drafts_path)
    lines = drafts_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["query"] == "Who is the employer?"
