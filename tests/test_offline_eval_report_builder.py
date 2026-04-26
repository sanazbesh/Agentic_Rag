from __future__ import annotations

import json
from pathlib import Path

from evals.reports.build_report import build_report


def _run_blob(cases: list[dict], *, generated_at: str) -> dict:
    return {
        "runner": "offline_eval_runner_v1",
        "generated_at_utc": generated_at,
        "summary": {
            "case_count": len(cases),
            "failed_case_count": sum(1 for case in cases if case.get("runner_status") != "ok"),
        },
        "cases": cases,
    }


def _case(
    case_id: str,
    family: str,
    *,
    passed: bool,
    failure_label: str | None = None,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
) -> dict:
    deterministic = {
        "answerability_checks": {
            "passed": passed,
            "classification": "correct_supported_answer" if passed else (failure_label or "false_positive"),
        },
        "family_routing": {"is_correct": passed},
    }
    row = {
        "case_id": case_id,
        "family": family,
        "runner_status": "ok",
        "deterministic_eval_results": deterministic,
        "llm_judge_results": {},
        "system_result": {},
        "debug_payload": {},
    }
    if latency_ms is not None:
        row["system_result"]["latency_ms"] = latency_ms
    if cost_usd is not None:
        row["system_result"]["cost_usd"] = cost_usd
    return row


def test_single_run_report_generation_includes_required_sections(tmp_path: Path) -> None:
    run = _run_blob(
        [
            _case("c1", "party_role_verification", passed=True, latency_ms=120.0, cost_usd=0.01),
            _case("c2", "party_role_verification", passed=False, failure_label="false_negative", latency_ms=240.0, cost_usd=0.02),
            _case("c3", "chronology_date_event", passed=True, latency_ms=100.0, cost_usd=0.03),
        ],
        generated_at="2026-04-19T00:00:00+00:00",
    )
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    output_path = tmp_path / "report.md"
    built = build_report(candidate_run=run_path, output_path=output_path)

    assert built.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "# Offline Eval Report" in text
    assert "## Overall results" in text
    assert "## Family score breakdown" in text
    assert "## Failure breakdown" in text
    assert "## Worst regressions" in text
    assert "## Latency and cost summary" in text
    assert "Overall score (pass rate)" in text


def test_comparison_report_shows_family_regressions_and_worst_regressions(tmp_path: Path) -> None:
    baseline = _run_blob(
        [
            _case("shared-1", "party_role_verification", passed=True, latency_ms=100.0, cost_usd=0.01),
            _case("shared-2", "party_role_verification", passed=True, latency_ms=110.0, cost_usd=0.01),
            _case("shared-3", "chronology_date_event", passed=True, latency_ms=90.0, cost_usd=0.02),
        ],
        generated_at="2026-04-18T00:00:00+00:00",
    )
    candidate = _run_blob(
        [
            _case("shared-1", "party_role_verification", passed=False, failure_label="false_positive", latency_ms=130.0, cost_usd=0.02),
            _case("shared-2", "party_role_verification", passed=True, latency_ms=115.0, cost_usd=0.01),
            _case("shared-3", "chronology_date_event", passed=True, latency_ms=95.0, cost_usd=0.02),
        ],
        generated_at="2026-04-19T00:00:00+00:00",
    )

    base_path = tmp_path / "baseline.json"
    cand_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(baseline), encoding="utf-8")
    cand_path.write_text(json.dumps(candidate), encoding="utf-8")

    output_path = tmp_path / "comparison.md"
    build_report(candidate_run=cand_path, baseline_run=base_path, output_path=output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "## Baseline vs candidate summary" in text
    assert "## Family-level regressions" in text
    assert "🔻 REGRESSION" in text
    assert "Families that worsened" in text
    assert "## Worst regressions" in text
    assert "`shared-1` (party_role_verification)" in text


def test_missing_latency_and_cost_are_reported_as_not_available(tmp_path: Path) -> None:
    run = _run_blob(
        [
            _case("x1", "party_role_verification", passed=True),
            _case("x2", "party_role_verification", passed=False, failure_label="false_negative"),
        ],
        generated_at="2026-04-19T00:00:00+00:00",
    )

    run_path = tmp_path / "run_no_perf.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    output_path = tmp_path / "report_no_perf.md"

    build_report(candidate_run=run_path, output_path=output_path)
    text = output_path.read_text(encoding="utf-8")

    assert "Latency: not available in run output" in text
    assert "Cost: not available in run output" in text
    assert "Failure breakdown" in text
