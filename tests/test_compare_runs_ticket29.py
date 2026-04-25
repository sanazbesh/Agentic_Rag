from __future__ import annotations

import json
from pathlib import Path

from evals.reports.compare_runs import compare_runs, render_comparison_markdown, write_comparison_report


def _case(
    case_id: str,
    family: str,
    *,
    passed: bool,
    false_positive: bool,
    citation_ok: bool,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
) -> dict:
    deterministic = {
        "answerability_checks": {
            "passed": passed,
            "classification": "false_positive" if false_positive else "correct_supported_answer",
            "aggregation_fields": {"false_positive": false_positive},
        },
        "citation_checks": {
            "passed": citation_ok,
            "aggregation_fields": {"citation_support_match_pass": citation_ok},
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


def _run_blob(cases: list[dict], *, generated_at: str) -> dict:
    return {
        "runner": "offline_eval_runner_v1",
        "generated_at_utc": generated_at,
        "summary": {"case_count": len(cases)},
        "cases": cases,
    }


def _write(path: Path, blob: dict) -> Path:
    path.write_text(json.dumps(blob), encoding="utf-8")
    return path


def test_compare_runs_can_compare_two_valid_artifacts_and_render_report(tmp_path: Path) -> None:
    baseline = _run_blob(
        [
            _case("c1", "party_role", passed=True, false_positive=False, citation_ok=True, latency_ms=100, cost_usd=0.01),
            _case("c2", "chronology", passed=True, false_positive=False, citation_ok=True, latency_ms=120, cost_usd=0.02),
        ],
        generated_at="2026-04-21T00:00:00+00:00",
    )
    candidate = _run_blob(
        [
            _case("c1", "party_role", passed=True, false_positive=False, citation_ok=True, latency_ms=90, cost_usd=0.009),
            _case("c2", "chronology", passed=False, false_positive=True, citation_ok=False, latency_ms=140, cost_usd=0.03),
        ],
        generated_at="2026-04-22T00:00:00+00:00",
    )

    base_path = _write(tmp_path / "baseline.json", baseline)
    cand_path = _write(tmp_path / "candidate.json", candidate)

    comparison, markdown = write_comparison_report(baseline=base_path, candidate=cand_path)

    assert comparison["summary"]["verdict"]
    assert "# Release Comparison Report" in markdown
    assert "## Risk metrics" in markdown
    assert "## Performance / efficiency" in markdown


def test_family_score_deltas_are_computed_correctly(tmp_path: Path) -> None:
    baseline = _run_blob(
        [
            _case("a1", "family_a", passed=True, false_positive=False, citation_ok=True),
            _case("a2", "family_a", passed=False, false_positive=True, citation_ok=False),
        ],
        generated_at="2026-04-21T00:00:00+00:00",
    )
    candidate = _run_blob(
        [
            _case("a1", "family_a", passed=True, false_positive=False, citation_ok=True),
            _case("a2", "family_a", passed=True, false_positive=False, citation_ok=True),
        ],
        generated_at="2026-04-22T00:00:00+00:00",
    )

    cmp = compare_runs(baseline=_write(tmp_path / "b.json", baseline), candidate=_write(tmp_path / "c.json", candidate))
    row = next(item for item in cmp["family_comparison"] if item["family"] == "family_a")

    assert row["baseline"] == 0.5
    assert row["candidate"] == 1.0
    assert row["delta"] == 0.5
    assert row["status"] == "improved"


def test_false_confident_rate_delta_uses_lower_is_better_direction(tmp_path: Path) -> None:
    baseline = _run_blob([_case("c1", "f", passed=True, false_positive=True, citation_ok=True)], generated_at="2026-04-21T00:00:00+00:00")
    candidate = _run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=True)], generated_at="2026-04-22T00:00:00+00:00")

    cmp = compare_runs(baseline=_write(tmp_path / "b.json", baseline), candidate=_write(tmp_path / "c.json", candidate))
    metric = next(item for item in cmp["risk_metrics"] if item.name == "False confident rate")

    assert metric.delta == -1.0
    assert metric.direction == "improved"


def test_citation_correctness_delta_uses_higher_is_better_direction(tmp_path: Path) -> None:
    baseline = _run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=False)], generated_at="2026-04-21T00:00:00+00:00")
    candidate = _run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=True)], generated_at="2026-04-22T00:00:00+00:00")

    cmp = compare_runs(baseline=_write(tmp_path / "b.json", baseline), candidate=_write(tmp_path / "c.json", candidate))
    metric = next(item for item in cmp["risk_metrics"] if item.name == "Citation correctness")

    assert metric.delta == 1.0
    assert metric.direction == "improved"


def test_latency_and_cost_deltas_are_computed_correctly(tmp_path: Path) -> None:
    baseline = _run_blob(
        [
            _case("c1", "f", passed=True, false_positive=False, citation_ok=True, latency_ms=100, cost_usd=0.01),
            _case("c2", "f", passed=True, false_positive=False, citation_ok=True, latency_ms=200, cost_usd=0.02),
        ],
        generated_at="2026-04-21T00:00:00+00:00",
    )
    candidate = _run_blob(
        [
            _case("c1", "f", passed=True, false_positive=False, citation_ok=True, latency_ms=120, cost_usd=0.015),
            _case("c2", "f", passed=True, false_positive=False, citation_ok=True, latency_ms=220, cost_usd=0.025),
        ],
        generated_at="2026-04-22T00:00:00+00:00",
    )

    cmp = compare_runs(baseline=_write(tmp_path / "b.json", baseline), candidate=_write(tmp_path / "c.json", candidate))

    lat_p50 = next(item for item in cmp["performance_metrics"] if item.name == "Latency p50")
    cost_req = next(item for item in cmp["performance_metrics"] if item.name == "Cost / request")

    assert lat_p50.delta == 20.0
    assert lat_p50.direction == "regressed"
    assert round(cost_req.delta or 0.0, 6) == 0.005
    assert cost_req.direction == "regressed"


def test_missing_metrics_and_missing_families_do_not_crash_and_are_marked_unavailable(tmp_path: Path) -> None:
    baseline = _run_blob([_case("c1", "family_only_in_baseline", passed=True, false_positive=False, citation_ok=True)], generated_at="2026-04-21T00:00:00+00:00")
    candidate = _run_blob(
        [
            {
                "case_id": "c2",
                "family": "family_only_in_candidate",
                "runner_status": "ok",
                "deterministic_eval_results": {},
                "llm_judge_results": {},
                "system_result": {},
                "debug_payload": {},
            }
        ],
        generated_at="2026-04-22T00:00:00+00:00",
    )

    cmp = compare_runs(baseline=_write(tmp_path / "b.json", baseline), candidate=_write(tmp_path / "c.json", candidate))
    markdown = render_comparison_markdown(cmp)

    assert any(item.name == "False confident rate" and item.direction == "unavailable" for item in cmp["risk_metrics"])
    families = {row["family"] for row in cmp["family_comparison"]}
    assert "family_only_in_baseline" in families
    assert "family_only_in_candidate" in families
    assert "unavailable" in markdown


def test_partial_artifacts_still_produce_usable_report(tmp_path: Path) -> None:
    baseline = {
        "generated_at_utc": "2026-04-21T00:00:00+00:00",
        "cases": [{"case_id": "c1", "family": "f", "runner_status": "failed"}],
    }
    candidate = {
        "generated_at_utc": "2026-04-22T00:00:00+00:00",
        "cases": [{"case_id": "c1", "family": "f", "runner_status": "ok", "deterministic_eval_results": {}}],
    }

    _, markdown = write_comparison_report(
        baseline=_write(tmp_path / "baseline_partial.json", baseline),
        candidate=_write(tmp_path / "candidate_partial.json", candidate),
    )

    assert "Release notes summary" in markdown
    assert "Family comparison" in markdown


def test_output_labels_improvement_and_regression_directions_correctly(tmp_path: Path) -> None:
    baseline = _run_blob([_case("c1", "f", passed=True, false_positive=True, citation_ok=False, latency_ms=100, cost_usd=0.010)], generated_at="2026-04-21T00:00:00+00:00")
    candidate = _run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=True, latency_ms=120, cost_usd=0.012)], generated_at="2026-04-22T00:00:00+00:00")

    cmp = compare_runs(baseline=_write(tmp_path / "b.json", baseline), candidate=_write(tmp_path / "c.json", candidate))
    false_conf = next(item for item in cmp["risk_metrics"] if item.name == "False confident rate")
    citation = next(item for item in cmp["risk_metrics"] if item.name == "Citation correctness")
    latency = next(item for item in cmp["performance_metrics"] if item.name == "Latency p50")

    assert false_conf.direction == "improved"
    assert citation.direction == "improved"
    assert latency.direction == "regressed"


def test_directory_input_uses_latest_json_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "candidate_dir"
    run_dir.mkdir()
    older = run_dir / "older.json"
    newer = run_dir / "newer.json"
    older.write_text(json.dumps(_run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=True)], generated_at="2026-04-20T00:00:00+00:00")), encoding="utf-8")
    newer.write_text(json.dumps(_run_blob([_case("c1", "f", passed=False, false_positive=True, citation_ok=False)], generated_at="2026-04-21T00:00:00+00:00")), encoding="utf-8")

    baseline = _write(tmp_path / "baseline.json", _run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=True)], generated_at="2026-04-19T00:00:00+00:00"))
    cmp = compare_runs(baseline=baseline, candidate=run_dir)

    assert cmp["candidate"]["run_id"] == "2026-04-21T00:00:00+00:00"


def test_run_id_input_resolves_under_evals_runs(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    runs = repo / "evals" / "runs"
    runs.mkdir(parents=True)

    (runs / "baseline-run.json").write_text(json.dumps(_run_blob([_case("c1", "f", passed=True, false_positive=False, citation_ok=True)], generated_at="2026-04-20T00:00:00+00:00")), encoding="utf-8")
    (runs / "candidate-run.json").write_text(json.dumps(_run_blob([_case("c1", "f", passed=False, false_positive=True, citation_ok=False)], generated_at="2026-04-21T00:00:00+00:00")), encoding="utf-8")

    monkeypatch.chdir(repo)
    cmp = compare_runs(baseline="baseline-run", candidate="candidate-run")

    assert cmp["baseline"]["run_id"] == "2026-04-20T00:00:00+00:00"
    assert cmp["candidate"]["run_id"] == "2026-04-21T00:00:00+00:00"
