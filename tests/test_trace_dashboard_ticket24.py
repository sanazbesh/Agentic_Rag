from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.reports.trace_dashboard_data import build_trace_drilldown, discover_trace_run_files, load_trace_runs


def _case(*, include_rewrite: bool = True, include_decomposition: bool = True) -> dict[str, Any]:
    trace = {
        "trace_id": "tr_1",
        "query": "Who is the employer?",
        "active_family": "party_role_verification",
        "spans": [
            {
                "stage": "query_understanding",
                "status": "success",
                "outputs_summary": {
                    "normalized_query": "who is the employer",
                    "question_type": "extractive_fact_query",
                    "legal_question_family": "party_role_verification",
                    "routing_notes": ["family routed"],
                    "is_followup": False,
                    "is_document_scoped": True,
                },
                "warnings": [],
            },
            {
                "stage": "decomposition",
                "status": "success",
                "outputs_summary": {
                    "needs_decomposition": include_decomposition,
                    "decomposition_gate_reasons": ["single_hop"],
                    "strategy": "none",
                    "subquery_ids": [],
                    "validation_outcome": "not_applicable",
                },
                "warnings": [],
            },
            {
                "stage": "retrieval",
                "status": "success",
                "outputs_summary": {
                    "effective_query": "who is the employer",
                    "selected_document_scope": ["doc-1"],
                    "retrieved_child_count": 3,
                    "top_child_chunk_ids": ["c1", "c2"],
                    "retrieval_mode": "hybrid",
                },
                "warnings": [],
            },
            {
                "stage": "rerank",
                "status": "success",
                "outputs_summary": {
                    "input_candidate_count": 3,
                    "output_candidate_count": 2,
                    "top_reranked_child_ids": ["c2", "c1"],
                    "ranking_source": "bm25+dense",
                },
                "warnings": [],
            },
            {
                "stage": "answerability",
                "status": "success",
                "outputs_summary": {
                    "sufficient_context": True,
                    "support_level": "sufficient",
                    "should_answer": True,
                    "insufficiency_reason": None,
                    "matched_headings": ["Parties"],
                    "matched_parent_chunk_ids": ["p1"],
                },
                "warnings": [],
            },
            {
                "stage": "final_synthesis",
                "status": "success",
                "outputs_summary": {"final_output_status": "answered"},
                "warnings": [],
            },
        ],
    }
    debug_payload = {
        "trace": trace,
        "query_classification": {"is_context_dependent": False},
        "resolved_query": "who is the employer",
        "effective_query": "who is the employer" if include_rewrite else "who the employer is",
        "decomposition": {
            "needs_decomposition": include_decomposition,
            "decomposition_gate_reasons": ["single_hop"],
        },
        "answerability_result": {
            "sufficient_context": True,
            "support_level": "sufficient",
            "should_answer": True,
            "insufficiency_reason": None,
            "evidence_notes": ["direct match"],
            "matched_headings": ["Parties"],
            "matched_parent_chunk_ids": ["p1"],
        },
        "warnings": ["pipeline_note"],
    }
    return {
        "case_id": "case-1",
        "family": "party_role_verification",
        "query": "Who is the employer?",
        "selected_document_ids": ["doc-1"],
        "runner_status": "ok",
        "debug_payload": debug_payload,
        "system_result": {
            "answer_text": "The employer is Acme Corp.",
            "grounded": True,
            "sufficient_context": True,
            "warnings": ["final_note"],
            "citations": [
                {
                    "source_name": "Employment Agreement",
                    "heading": "Parties",
                    "supporting_excerpt": "Acme Corp is the employer.",
                }
            ],
        },
    }


def test_loader_reads_stored_run_outputs(tmp_path: Path) -> None:
    payload = {"generated_at_utc": "2026-04-20T00:00:00Z", "cases": [_case()]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    run_files = discover_trace_run_files(tmp_path)
    runs = load_trace_runs(run_files)

    assert len(run_files) == 1
    assert len(runs) == 1
    assert runs[0].cases[0]["case_id"] == "case-1"


def test_classification_section_renders_expected_fields() -> None:
    drilldown = build_trace_drilldown(_case())
    section = drilldown["classification"]
    assert section["original_query"] == "Who is the employer?"
    assert section["normalized_query"] == "who is the employer"
    assert section["question_type"] == "extractive_fact_query"
    assert section["legal_family"] == "party_role_verification"


def test_rewrite_section_renders_expected_fields_or_absent_state() -> None:
    with_rewrite = build_trace_drilldown(_case(include_rewrite=False))
    assert with_rewrite["rewrite"]["rewrite_occurred"] is True

    without_fields = build_trace_drilldown({"case_id": "c", "debug_payload": {}, "system_result": {}})
    assert without_fields["rewrite"]["resolved_query"] is None


def test_decomposition_section_handles_present_or_absent_state() -> None:
    present = build_trace_drilldown(_case(include_decomposition=True))
    assert present["decomposition"]["needs_decomposition"] is True

    absent = build_trace_drilldown({"case_id": "c", "debug_payload": {}, "system_result": {}})
    assert absent["decomposition"]["needs_decomposition"] is None


def test_retrieval_and_rerank_sections_render_candidate_data() -> None:
    drilldown = build_trace_drilldown(_case())
    assert drilldown["retrieval"]["retrieved_child_count"] == 3
    assert drilldown["retrieval"]["top_child_chunk_ids"] == ["c1", "c2"]
    assert drilldown["rerank"]["output_candidate_count"] == 2
    assert drilldown["rerank"]["top_reranked_child_ids"] == ["c2", "c1"]


def test_answerability_section_renders_support_decision_fields() -> None:
    drilldown = build_trace_drilldown(_case())
    section = drilldown["answerability"]
    assert section["sufficient_context"] is True
    assert section["support_level"] == "sufficient"
    assert section["should_answer"] is True
    assert section["matched_headings"] == ["Parties"]


def test_final_answer_citations_and_warnings_sections_render() -> None:
    drilldown = build_trace_drilldown(_case())
    assert drilldown["final_answer"]["answer_text"] == "The employer is Acme Corp."
    assert drilldown["citation_count"] == 1
    assert drilldown["citations"][0]["source_name"] == "Employment Agreement"
    assert "pipeline" in drilldown["warnings"]


def test_failure_layer_summary_identifies_first_suspicious_stage() -> None:
    case = _case()
    case["debug_payload"]["trace"]["spans"][2]["outputs_summary"]["retrieved_child_count"] = 0
    drilldown = build_trace_drilldown(case)
    assert drilldown["failure_layer"] is not None
    assert drilldown["failure_layer"]["stage"] == "retrieval"


def test_non_numeric_retrieved_child_count_does_not_crash_and_degrades_to_warning() -> None:
    case = _case()
    case["debug_payload"]["trace"]["spans"][2]["outputs_summary"]["retrieved_child_count"] = "n/a"
    drilldown = build_trace_drilldown(case)
    retrieval_status = next(row for row in drilldown["stage_statuses"] if row["stage"] == "retrieval")
    assert retrieval_status["status"] == "warning"
    assert "not available" in str(retrieval_status.get("reason") or "")


def test_non_numeric_output_candidate_count_does_not_crash_and_degrades_to_warning() -> None:
    case = _case()
    case["debug_payload"]["trace"]["spans"][3]["outputs_summary"]["output_candidate_count"] = "n/a"
    drilldown = build_trace_drilldown(case)
    rerank_status = next(row for row in drilldown["stage_statuses"] if row["stage"] == "rerank")
    assert rerank_status["status"] == "warning"
    assert "not available" in str(rerank_status.get("reason") or "")


def test_valid_numeric_string_counts_preserve_existing_stage_behavior() -> None:
    case = _case()
    case["debug_payload"]["trace"]["spans"][2]["outputs_summary"]["retrieved_child_count"] = "12"
    case["debug_payload"]["trace"]["spans"][3]["outputs_summary"]["output_candidate_count"] = "5"
    drilldown = build_trace_drilldown(case)
    retrieval_status = next(row for row in drilldown["stage_statuses"] if row["stage"] == "retrieval")
    rerank_status = next(row for row in drilldown["stage_statuses"] if row["stage"] == "rerank")
    assert retrieval_status["status"] == "ok"
    assert rerank_status["status"] == "ok"


def test_missing_stage_counts_render_without_crashing() -> None:
    case = _case()
    del case["debug_payload"]["trace"]["spans"][2]["outputs_summary"]["retrieved_child_count"]
    del case["debug_payload"]["trace"]["spans"][3]["outputs_summary"]["output_candidate_count"]
    drilldown = build_trace_drilldown(case)
    statuses = {row["stage"]: row for row in drilldown["stage_statuses"]}
    assert statuses["retrieval"]["status"] == "warning"
    assert statuses["rerank"]["status"] == "warning"


def test_one_malformed_record_does_not_break_multi_case_iteration() -> None:
    good = _case()
    bad = _case()
    bad["case_id"] = "case-bad"
    bad["debug_payload"]["trace"]["spans"][2]["outputs_summary"]["retrieved_child_count"] = "not-a-number"
    bad["debug_payload"]["trace"]["spans"][3]["outputs_summary"]["output_candidate_count"] = ""

    drilldowns = [build_trace_drilldown(case) for case in [good, bad]]
    assert [item["case_id"] for item in drilldowns] == ["case-1", "case-bad"]
    bad_statuses = {row["stage"]: row for row in drilldowns[1]["stage_statuses"]}
    assert bad_statuses["retrieval"]["status"] == "warning"
    assert bad_statuses["rerank"]["status"] == "warning"


def test_partial_or_malformed_trace_does_not_crash() -> None:
    malformed = {
        "case_id": "broken",
        "debug_payload": {"trace": {"spans": [{"stage": "retrieval", "warnings": "not-a-list"}]}, "warnings": [123]},
        "system_result": {"citations": ["not-a-mapping"]},
    }
    drilldown = build_trace_drilldown(malformed)
    assert drilldown["case_id"] == "broken"
    assert isinstance(drilldown["stage_statuses"], list)
    assert isinstance(drilldown["warnings"], dict)
