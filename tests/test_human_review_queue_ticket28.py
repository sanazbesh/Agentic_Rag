from __future__ import annotations

import json
from pathlib import Path

from evals.reports.human_review_queue import (
    apply_review_decision,
    load_review_queue,
    load_review_records,
)


def _write_run(path: Path) -> None:
    payload = {
        "generated_at_utc": "2026-04-22T00:00:00Z",
        "cases": [
            {
                "case_id": "missing-cite",
                "query": "What is the statute for at-will employment?",
                "family": "employment_lifecycle",
                "runner_status": "ok",
                "system_output": {"answer_text": "answer", "citations": [], "warnings": []},
                "debug_payload": {"trace": {"trace_id": "tr-cite", "selected_document_ids": ["doc-1"]}},
                "deterministic_eval_results": {"citation_checks": {"status": "ok", "passed": False}},
            },
            {
                "case_id": "routing-disagree",
                "query": "Who is counsel of record?",
                "family": "party_role_verification",
                "runner_status": "ok",
                "system_output": {"answer_text": "answer", "citations": [{"document_id": "doc-2"}], "warnings": []},
                "debug_payload": {"trace": {"trace_id": "tr-route", "selected_document_ids": ["doc-2"]}},
                "deterministic_eval_results": {"family_routing": {"status": "ok", "passed": False}},
            },
            {
                "case_id": "hallucination",
                "query": "What did the contract guarantee?",
                "family": "financial_entitlement",
                "runner_status": "ok",
                "system_output": {
                    "answer_text": "answer",
                    "citations": [{"document_id": "doc-3"}],
                    "warnings": ["possible_hallucination"],
                },
                "debug_payload": {"trace": {"trace_id": "tr-hall", "selected_document_ids": ["doc-3"]}},
                "llm_judge_results": {"groundedness": {"status": "ok", "is_correct": False}},
            },
            {
                "case_id": "issue-spotting",
                "query": "Spot policy issues in this termination memo",
                "family": "policy_issue_spotting",
                "runner_status": "failed",
                "system_output": {"answer_text": "answer", "citations": [{"document_id": "doc-4"}], "warnings": []},
                "debug_payload": {"trace": {"trace_id": "tr-issue", "selected_document_ids": ["doc-4"]}},
                "deterministic_eval_results": {"contract_checks": {"status": "error"}},
            },
            "bad-row",
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pick(queue: list, category: str):
    for item in queue:
        if item.category == category:
            return item
    raise AssertionError(f"missing category {category}")


def test_review_queue_loads_sensitive_items_and_trace_linkage(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs / "run.json")

    queue = load_review_queue(run_dir=runs, review_store_path=tmp_path / "review.json")

    categories = {item.category for item in queue}
    assert "missing_citation" in categories
    assert "suspected_hallucination" in categories
    assert "family_routing_disagreement" in categories
    assert "legal_issue_spotting_disagreement" in categories
    assert _pick(queue, "missing_citation").trace_id == "tr-cite"


def test_review_actions_approve_reject_relabel_escalate_save_stably(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs / "run.json")
    store = tmp_path / "review_records.json"

    queue = load_review_queue(run_dir=runs, review_store_path=store)

    approved = apply_review_decision(
        item=_pick(queue, "missing_citation"),
        action="approve",
        reviewer_note="Citations are acceptable for this sample.",
        review_store_path=store,
        dataset_feedback_action="no_dataset_update",
    )
    rejected = apply_review_decision(
        item=_pick(queue, "family_routing_disagreement"),
        action="reject",
        reviewer_note="Routing label is wrong.",
        review_store_path=store,
        dataset_feedback_action="needs_regression_case",
    )
    relabeled = apply_review_decision(
        item=_pick(queue, "suspected_hallucination"),
        action="relabel",
        reviewer_note="Better classified as missing citation.",
        review_store_path=store,
        family="employment_lifecycle",
        category="missing_citation",
    )
    escalated = apply_review_decision(
        item=_pick(queue, "legal_issue_spotting_disagreement"),
        action="escalate",
        reviewer_note="High risk issue spotting miss.",
        review_store_path=store,
        dataset_feedback_action="needs_regression_case",
    )

    assert approved.review_status == "approved"
    assert rejected.review_status == "rejected"
    assert relabeled.review_status == "relabeled"
    assert relabeled.relabeled_category == "missing_citation"
    assert escalated.review_status == "escalated"

    loaded = load_review_records(store)
    assert len(loaded) == 4
    assert loaded[approved.review_item_id]["dataset_feedback_action"] == "no_dataset_update"
    assert loaded[escalated.review_item_id]["review_action"] == "escalate"


def test_escalation_can_create_dataset_feedback_regression_draft(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs / "run.json")

    queue = load_review_queue(run_dir=runs, review_store_path=tmp_path / "review.json")
    item = _pick(queue, "legal_issue_spotting_disagreement")

    drafts = tmp_path / "regression_case_drafts.jsonl"
    record = apply_review_decision(
        item=item,
        action="escalate",
        reviewer_note="convert into regression coverage",
        review_store_path=tmp_path / "review.json",
        create_regression_draft=True,
        regression_drafts_path=drafts,
    )

    assert record.dataset_feedback_action == "regression_case_draft_created"
    lines = drafts.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["regression_source"].startswith("human_review:")


def test_malformed_queue_inputs_do_not_crash_workflow(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "broken.json").write_text('{"cases": ["bad", {"case_id": "x"}], "generated_at_utc": "2026"', encoding="utf-8")

    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("not-json\n" + json.dumps({"shadow_eval_id": "se1", "trace_id": None}) + "\n", encoding="utf-8")

    queue = load_review_queue(
        run_dir=runs,
        review_store_path=tmp_path / "review.json",
        sampled_traffic_path=tmp_path / "samples.jsonl",
        shadow_eval_path=shadow,
    )
    assert queue == []


def test_local_first_solo_setup_works_with_only_local_files(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    _write_run(runs / "run.json")
    store = tmp_path / "review_records.json"

    queue = load_review_queue(
        run_dir=runs,
        review_store_path=store,
        sampled_traffic_path=tmp_path / "samples.jsonl",
        shadow_eval_path=tmp_path / "shadow.jsonl",
    )
    assert queue

    item = queue[0]
    apply_review_decision(item=item, action="approve", reviewer_note="local pass", review_store_path=store)
    assert store.exists()
