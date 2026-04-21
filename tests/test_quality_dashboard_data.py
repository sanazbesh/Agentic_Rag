from __future__ import annotations

import json
from pathlib import Path

from evals.reports.quality_dashboard_data import (
    compare_runs,
    compute_family_pass_rates,
    compute_run_trends,
    compute_top_failing_datasets,
    discover_run_files,
    filter_cases_by_family,
    load_eval_runs,
    select_recent_runs,
)


def _run_blob(*, generated_at: str, cases: list[dict]) -> dict:
    return {
        "runner": "offline_eval_runner_v1",
        "generated_at_utc": generated_at,
        "summary": {"case_count": len(cases)},
        "cases": cases,
    }


def _case(
    case_id: str,
    family: str,
    *,
    passed: bool,
    false_positive: bool = False,
    citation_support_match_pass: bool | None = None,
    safe_failure_quality_pass: bool | None = None,
    dataset: str | None = None,
) -> dict:
    deterministic: dict = {
        "answerability_checks": {
            "passed": passed,
            "classification": "false_positive" if false_positive else "correct_supported_answer",
            "aggregation_fields": {
                "false_positive": false_positive,
                "safe_failure_quality_pass": safe_failure_quality_pass,
            },
        }
    }
    if citation_support_match_pass is not None:
        deterministic["citation_checks"] = {
            "passed": citation_support_match_pass,
            "aggregation_fields": {
                "citation_support_match_pass": citation_support_match_pass,
            },
        }

    llm_judges = {}
    if safe_failure_quality_pass is None:
        llm_judges["safe_failure"] = {"passed": passed}

    row = {
        "case_id": case_id,
        "family": family,
        "runner_status": "ok",
        "deterministic_eval_results": deterministic,
        "llm_judge_results": llm_judges,
    }
    if dataset:
        row["_dataset_file"] = dataset
    return row


def test_dashboard_loader_reads_stored_eval_outputs(tmp_path: Path) -> None:
    run_path = tmp_path / "run_1.json"
    run_path.write_text(json.dumps(_run_blob(generated_at="2026-04-20T00:00:00+00:00", cases=[])), encoding="utf-8")

    discovered = discover_run_files(tmp_path)
    runs = load_eval_runs(discovered)

    assert len(discovered) == 1
    assert len(runs) == 1
    assert runs[0].run_id == "2026-04-20T00:00:00+00:00"


def test_family_level_pass_rates_are_computed_correctly() -> None:
    cases = [
        _case("a", "family_a", passed=True),
        _case("b", "family_a", passed=False),
        _case("c", "family_b", passed=True),
    ]

    rows = {row["family"]: row for row in compute_family_pass_rates(cases)}

    assert rows["family_a"]["case_count"] == 2
    assert rows["family_a"]["pass_rate"] == 0.5
    assert rows["family_a"]["fail_count"] == 1
    assert rows["family_b"]["pass_rate"] == 1.0


def test_false_confident_trend_is_computed_correctly(tmp_path: Path) -> None:
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    run1.write_text(
        json.dumps(_run_blob(generated_at="2026-04-19T00:00:00+00:00", cases=[_case("a", "f1", passed=True, false_positive=False)])),
        encoding="utf-8",
    )
    run2.write_text(
        json.dumps(_run_blob(generated_at="2026-04-20T00:00:00+00:00", cases=[_case("b", "f1", passed=False, false_positive=True)])),
        encoding="utf-8",
    )

    trends = compute_run_trends(load_eval_runs([run1, run2]))

    assert trends[0].false_confident_rate == 0.0
    assert trends[1].false_confident_rate == 1.0


def test_citation_correctness_trend_is_computed_correctly(tmp_path: Path) -> None:
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    run1.write_text(
        json.dumps(_run_blob(generated_at="2026-04-19T00:00:00+00:00", cases=[_case("a", "f1", passed=True, citation_support_match_pass=True)])),
        encoding="utf-8",
    )
    run2.write_text(
        json.dumps(_run_blob(generated_at="2026-04-20T00:00:00+00:00", cases=[_case("b", "f1", passed=False, citation_support_match_pass=False)])),
        encoding="utf-8",
    )

    trends = compute_run_trends(load_eval_runs([run1, run2]))

    assert trends[0].citation_correctness_rate == 1.0
    assert trends[1].citation_correctness_rate == 0.0


def test_safe_failure_trend_is_computed_correctly(tmp_path: Path) -> None:
    run1 = tmp_path / "run1.json"
    run2 = tmp_path / "run2.json"
    run1.write_text(
        json.dumps(_run_blob(generated_at="2026-04-19T00:00:00+00:00", cases=[_case("a", "f1", passed=True, safe_failure_quality_pass=True)])),
        encoding="utf-8",
    )
    run2.write_text(
        json.dumps(_run_blob(generated_at="2026-04-20T00:00:00+00:00", cases=[_case("b", "f1", passed=False, safe_failure_quality_pass=False)])),
        encoding="utf-8",
    )

    trends = compute_run_trends(load_eval_runs([run1, run2]))

    assert trends[0].safe_failure_rate == 1.0
    assert trends[1].safe_failure_rate == 0.0


def test_top_failing_datasets_are_identified_correctly() -> None:
    cases = [
        _case("a", "f1", passed=False, dataset="d1"),
        _case("b", "f1", passed=False, dataset="d1"),
        _case("c", "f1", passed=True, dataset="d2"),
    ]

    top = compute_top_failing_datasets(cases)

    assert top[0]["dataset"] == "d1"
    assert top[0]["fail_count"] == 2


def test_time_filter_last_n_runs_works(tmp_path: Path) -> None:
    runs = []
    for idx in range(3):
        path = tmp_path / f"run{idx}.json"
        path.write_text(
            json.dumps(_run_blob(generated_at=f"2026-04-{18+idx:02d}T00:00:00+00:00", cases=[])),
            encoding="utf-8",
        )
        runs.append(path)

    loaded = load_eval_runs(runs)
    recent = select_recent_runs(loaded, last_n=2)

    assert len(recent) == 2
    assert recent[0].run_id == "2026-04-19T00:00:00+00:00"
    assert recent[1].run_id == "2026-04-20T00:00:00+00:00"


def test_family_filtering_works() -> None:
    cases = [
        _case("a", "family_a", passed=True),
        _case("b", "family_b", passed=True),
    ]

    filtered = filter_cases_by_family(cases, ["family_b"])

    assert len(filtered) == 1
    assert filtered[0]["family"] == "family_b"


def test_release_comparison_works(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"

    base_path.write_text(
        json.dumps(
            _run_blob(
                generated_at="2026-04-19T00:00:00+00:00",
                cases=[
                    _case("a", "family_a", passed=True, dataset="d1", citation_support_match_pass=True),
                    _case("b", "family_a", passed=True, dataset="d1", citation_support_match_pass=True),
                ],
            )
        ),
        encoding="utf-8",
    )
    cand_path.write_text(
        json.dumps(
            _run_blob(
                generated_at="2026-04-20T00:00:00+00:00",
                cases=[
                    _case("a", "family_a", passed=False, false_positive=True, dataset="d1", citation_support_match_pass=False),
                    _case("b", "family_a", passed=True, dataset="d1", citation_support_match_pass=True),
                ],
            )
        ),
        encoding="utf-8",
    )

    baseline, candidate = load_eval_runs([base_path, cand_path])
    comparison = compare_runs(baseline, candidate)

    assert comparison["overall_delta"] == -0.5
    assert comparison["false_confident_delta"] == 0.5
    assert comparison["citation_correctness_delta"] == -0.5
    assert comparison["family_deltas"][0]["delta"] == -0.5


def test_missing_optional_fields_are_handled_gracefully(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            _run_blob(
                generated_at="2026-04-20T00:00:00+00:00",
                cases=[
                    {
                        "case_id": "a",
                        "family": "family_a",
                        "runner_status": "ok",
                        "deterministic_eval_results": {},
                        "llm_judge_results": {},
                    }
                ],
            )
        ),
        encoding="utf-8",
    )

    trends = compute_run_trends(load_eval_runs([run_path]))

    assert trends[0].false_confident_rate is None
    assert trends[0].citation_correctness_rate is None
    assert trends[0].safe_failure_rate is None
