from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


def build_report(
    *,
    candidate_run: str | Path,
    output_path: str | Path,
    baseline_run: str | Path | None = None,
) -> Path:
    """Build a markdown report for one run or a baseline-vs-candidate comparison."""

    candidate_blob = _load_run_blob(candidate_run)
    candidate_summary = _summarize_run(candidate_blob)

    baseline_summary: dict[str, Any] | None = None
    if baseline_run is not None:
        baseline_blob = _load_run_blob(baseline_run)
        baseline_summary = _summarize_run(baseline_blob)

    markdown = _render_markdown(candidate_summary=candidate_summary, baseline_summary=baseline_summary)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return output


def _load_run_blob(run_path: str | Path) -> dict[str, Any]:
    path = Path(run_path)
    if path.is_dir():
        candidates = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"no JSON files found under {path}")
        path = candidates[0]

    blob = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        raise ValueError(f"run output must be a JSON object: {path}")
    blob["_source_path"] = str(path)
    return blob


def _summarize_run(blob: Mapping[str, Any]) -> dict[str, Any]:
    cases = [item for item in (blob.get("cases") or []) if isinstance(item, Mapping)]
    run_summary = blob.get("summary") if isinstance(blob.get("summary"), Mapping) else {}

    case_outcomes: list[dict[str, Any]] = []
    failure_counter: Counter[str] = Counter()
    family_bucket: defaultdict[str, list[bool]] = defaultdict(list)

    latency_values = _extract_numeric_series(cases, ["latency_ms", "latency", "duration_ms", "elapsed_ms"])
    cost_values = _extract_numeric_series(cases, ["cost_usd", "cost", "usd_cost"])

    for case in cases:
        case_id = str(case.get("case_id") or "")
        family = str(case.get("family") or "unknown")
        runner_status = str(case.get("runner_status") or "unknown")

        outcome = _evaluate_case_outcome(case)
        family_bucket[family].append(outcome["passed"])
        case_outcomes.append(
            {
                "case_id": case_id,
                "family": family,
                "passed": outcome["passed"],
                "failure_reasons": outcome["failure_reasons"],
            }
        )

        if runner_status != "ok":
            failure_counter["runner_failed"] += 1
        for reason in outcome["failure_reasons"]:
            failure_counter[reason] += 1

    total_cases = len(case_outcomes)
    passed_cases = sum(1 for row in case_outcomes if row["passed"])
    overall_pass_rate = (passed_cases / total_cases) if total_cases else 0.0

    family_scores = {
        family: {
            "family": family,
            "case_count": len(flags),
            "pass_count": sum(1 for flag in flags if flag),
            "pass_rate": (sum(1 for flag in flags if flag) / len(flags)) if flags else 0.0,
        }
        for family, flags in sorted(family_bucket.items())
    }

    worst_failing_cases = [row for row in case_outcomes if not row["passed"]][:10]

    return {
        "source_path": blob.get("_source_path"),
        "run_identifier": blob.get("generated_at_utc") or blob.get("run_id") or blob.get("id"),
        "runner": blob.get("runner"),
        "case_count": total_cases,
        "overall_pass_rate": overall_pass_rate,
        "passed_case_count": passed_cases,
        "failed_case_count": total_cases - passed_cases,
        "family_scores": family_scores,
        "failure_breakdown": dict(failure_counter.most_common()),
        "case_outcomes": case_outcomes,
        "worst_failing_cases": worst_failing_cases,
        "latency_summary": _series_summary(latency_values, unit="ms"),
        "cost_summary": _series_summary(cost_values, unit="usd"),
        "summary_from_runner": dict(run_summary),
    }


def _extract_numeric_series(cases: Sequence[Mapping[str, Any]], candidate_keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for case in cases:
        found = _find_numeric(case, candidate_keys)
        if found is not None:
            values.append(found)
    return values


def _find_numeric(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    nested_maps = [payload]
    for key in ("system_result", "debug_payload", "metrics", "metadata"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            nested_maps.append(value)

    for nested in nested_maps:
        for key in keys:
            value = nested.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _series_summary(values: Sequence[float], *, unit: str) -> dict[str, Any]:
    if not values:
        return {"available": False, "message": "not available in run output", "unit": unit}

    sorted_values = sorted(values)
    idx = max(0, int(0.95 * (len(sorted_values) - 1)))
    return {
        "available": True,
        "count": len(values),
        "avg": mean(values),
        "p95": sorted_values[idx],
        "unit": unit,
    }


def _evaluate_case_outcome(case: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if str(case.get("runner_status") or "") != "ok":
        reasons.append("runner_failed")

    for group_key in ("deterministic_eval_results", "llm_judge_results"):
        group = case.get(group_key)
        if not isinstance(group, Mapping):
            continue
        for evaluator_key, evaluator_result in group.items():
            if not isinstance(evaluator_result, Mapping):
                continue
            if evaluator_result.get("status") == "skipped":
                continue
            if evaluator_result.get("status") == "error":
                reasons.append(f"{evaluator_key}:error")
                continue

            passed_value = _resolve_passed(evaluator_result)
            if passed_value is False:
                label = evaluator_result.get("classification") or evaluator_result.get("label") or "failed"
                reasons.append(f"{evaluator_key}:{label}")

    return {"passed": len(reasons) == 0, "failure_reasons": reasons}


def _resolve_passed(result: Mapping[str, Any]) -> bool | None:
    for key in ("passed", "is_correct"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
    return None


def _render_markdown(*, candidate_summary: Mapping[str, Any], baseline_summary: Mapping[str, Any] | None) -> str:
    sections = [
        "# Offline Eval Report",
        "",
        _render_run_summary(candidate_summary),
        _render_overall(candidate_summary),
        _render_family_scores(candidate_summary),
        _render_failure_breakdown(candidate_summary),
        _render_latency_cost(candidate_summary),
    ]

    if baseline_summary is None:
        sections.append(_render_worst_failures_single_run(candidate_summary))
    else:
        comparison = _build_comparison(baseline=baseline_summary, candidate=candidate_summary)
        sections.extend(
            [
                _render_comparison_summary(comparison),
                _render_family_regressions(comparison),
                _render_worst_regressions(comparison),
                _render_latency_cost_comparison(comparison),
            ]
        )

    return "\n\n".join(sections).strip() + "\n"


def _build_comparison(*, baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_cases = {row["case_id"]: row for row in baseline.get("case_outcomes", []) if row.get("case_id")}
    candidate_cases = {row["case_id"]: row for row in candidate.get("case_outcomes", []) if row.get("case_id")}

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for case_id, candidate_case in candidate_cases.items():
        baseline_case = baseline_cases.get(case_id)
        if baseline_case is None:
            continue
        if baseline_case["passed"] and not candidate_case["passed"]:
            regressions.append(
                {
                    "case_id": case_id,
                    "family": candidate_case["family"],
                    "baseline_reasons": baseline_case["failure_reasons"],
                    "candidate_reasons": candidate_case["failure_reasons"],
                }
            )
        if (not baseline_case["passed"]) and candidate_case["passed"]:
            improvements.append({"case_id": case_id, "family": candidate_case["family"]})

    family_rows: list[dict[str, Any]] = []
    families = sorted(set((baseline.get("family_scores") or {}).keys()) | set((candidate.get("family_scores") or {}).keys()))
    for family in families:
        b = (baseline.get("family_scores") or {}).get(family, {"pass_rate": 0.0, "case_count": 0})
        c = (candidate.get("family_scores") or {}).get(family, {"pass_rate": 0.0, "case_count": 0})
        delta = float(c.get("pass_rate", 0.0)) - float(b.get("pass_rate", 0.0))
        family_rows.append(
            {
                "family": family,
                "baseline": float(b.get("pass_rate", 0.0)),
                "candidate": float(c.get("pass_rate", 0.0)),
                "delta": delta,
                "marker": "🔻 REGRESSION" if delta < 0 else ("🟢 IMPROVEMENT" if delta > 0 else "—"),
            }
        )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "overall_delta": float(candidate.get("overall_pass_rate", 0.0)) - float(baseline.get("overall_pass_rate", 0.0)),
        "family_rows": family_rows,
        "regressions": regressions,
        "improvements": improvements,
    }


def _render_run_summary(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "## Run summary",
            f"- Run identifier: `{summary.get('run_identifier') or 'not available'}`",
            f"- Source file: `{summary.get('source_path')}`",
            f"- Runner: `{summary.get('runner') or 'unknown'}`",
            f"- Total cases: **{summary.get('case_count', 0)}**",
        ]
    )


def _render_overall(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "## Overall results",
            f"- Overall score (pass rate): **{_pct(float(summary.get('overall_pass_rate', 0.0)))}**",
            f"- Passed cases: {summary.get('passed_case_count', 0)}",
            f"- Failed cases: {summary.get('failed_case_count', 0)}",
        ]
    )


def _render_family_scores(summary: Mapping[str, Any]) -> str:
    lines = [
        "## Family score breakdown",
        "| Family | Cases | Score (pass rate) |",
        "| --- | ---: | ---: |",
    ]
    family_scores = summary.get("family_scores") or {}
    for family, row in sorted(family_scores.items()):
        lines.append(f"| `{family}` | {row.get('case_count', 0)} | {_pct(float(row.get('pass_rate', 0.0)))} |")
    return "\n".join(lines)


def _render_failure_breakdown(summary: Mapping[str, Any]) -> str:
    lines = ["## Failure breakdown"]
    breakdown = summary.get("failure_breakdown") or {}
    if not breakdown:
        lines.append("- No failures detected.")
        return "\n".join(lines)

    lines.extend([
        "| Failure class | Count |",
        "| --- | ---: |",
    ])
    for name, count in breakdown.items():
        lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)


def _render_worst_failures_single_run(summary: Mapping[str, Any]) -> str:
    lines = ["## Worst regressions", "- Comparison mode not enabled; showing top failing cases in this run."]
    failing = summary.get("worst_failing_cases") or []
    if not failing:
        lines.append("- No failing cases.")
        return "\n".join(lines)

    for row in failing:
        reasons = ", ".join(row.get("failure_reasons") or ["unknown"])
        lines.append(f"- `{row.get('case_id')}` ({row.get('family')}): {reasons}")
    return "\n".join(lines)


def _render_latency_cost(summary: Mapping[str, Any]) -> str:
    lines = ["## Latency and cost summary"]
    lines.extend(_render_series("Latency", summary.get("latency_summary") or {}))
    lines.extend(_render_series("Cost", summary.get("cost_summary") or {}))
    return "\n".join(lines)


def _render_series(name: str, series: Mapping[str, Any]) -> list[str]:
    if not series.get("available"):
        return [f"- {name}: not available in run output"]
    unit = str(series.get("unit") or "")
    return [
        f"- {name} avg: {float(series.get('avg', 0.0)):.2f} {unit}",
        f"- {name} p95: {float(series.get('p95', 0.0)):.2f} {unit}",
    ]


def _render_comparison_summary(comparison: Mapping[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    return "\n".join(
        [
            "## Baseline vs candidate summary",
            f"- Baseline overall score: **{_pct(float(baseline.get('overall_pass_rate', 0.0)))}**",
            f"- Candidate overall score: **{_pct(float(candidate.get('overall_pass_rate', 0.0)))}**",
            f"- Overall delta: **{_pct(float(comparison.get('overall_delta', 0.0)), signed=True)}**",
            f"- New regressions: **{len(comparison.get('regressions', []))}**",
            f"- Improvements: **{len(comparison.get('improvements', []))}**",
        ]
    )


def _render_family_regressions(comparison: Mapping[str, Any]) -> str:
    lines = [
        "## Family-level regressions",
        "| Family | Baseline | Candidate | Delta | Marker |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in comparison.get("family_rows", []):
        lines.append(
            f"| `{row['family']}` | {_pct(row['baseline'])} | {_pct(row['candidate'])} | {_pct(row['delta'], signed=True)} | {row['marker']} |"
        )

    regressions = [row for row in comparison.get("family_rows", []) if row["delta"] < 0]
    if regressions:
        lines.append("\nFamilies that worsened:")
        for row in regressions:
            lines.append(f"- 🔻 `{row['family']}` dropped by {_pct(row['delta'], signed=True)}")
    else:
        lines.append("\n- No family-level regressions.")
    return "\n".join(lines)


def _render_worst_regressions(comparison: Mapping[str, Any]) -> str:
    lines = ["## Worst regressions"]
    regressions = comparison.get("regressions", [])
    if not regressions:
        lines.append("- No case-level pass→fail regressions detected.")
        return "\n".join(lines)

    for row in regressions[:10]:
        reasons = ", ".join(row.get("candidate_reasons") or ["unknown"])
        lines.append(f"- `{row['case_id']}` ({row['family']}): {reasons}")
    return "\n".join(lines)


def _render_latency_cost_comparison(comparison: Mapping[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines = ["## Latency and cost deltas"]

    lines.extend(_render_series_delta("Latency", baseline.get("latency_summary") or {}, candidate.get("latency_summary") or {}))
    lines.extend(_render_series_delta("Cost", baseline.get("cost_summary") or {}, candidate.get("cost_summary") or {}))
    return "\n".join(lines)


def _render_series_delta(name: str, baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[str]:
    if not baseline.get("available") or not candidate.get("available"):
        return [f"- {name}: not available in run output"]
    unit = str(candidate.get("unit") or "")
    delta_avg = float(candidate.get("avg", 0.0)) - float(baseline.get("avg", 0.0))
    delta_p95 = float(candidate.get("p95", 0.0)) - float(baseline.get("p95", 0.0))
    return [
        f"- {name} avg delta: {delta_avg:+.2f} {unit}",
        f"- {name} p95 delta: {delta_p95:+.2f} {unit}",
    ]


def _pct(value: float, *, signed: bool = False) -> str:
    if signed:
        return f"{value * 100:+.1f}%"
    return f"{value * 100:.1f}%"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a readable markdown report from offline eval JSON output.")
    parser.add_argument("--candidate", required=True, help="Candidate run JSON file or directory.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    parser.add_argument("--baseline", help="Optional baseline run JSON file or directory for comparison mode.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    build_report(candidate_run=args.candidate, baseline_run=args.baseline, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
