from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class MetricComparison:
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    delta_pct: float | None
    direction: str
    higher_is_better: bool
    unit: str


def compare_runs(*, baseline: str | Path, candidate: str | Path) -> dict[str, Any]:
    baseline_blob = _load_run_blob(baseline)
    candidate_blob = _load_run_blob(candidate)

    baseline_summary = _summarize_run(baseline_blob)
    candidate_summary = _summarize_run(candidate_blob)

    overall_cmp = _compare_scalar(
        name="Overall score",
        baseline=baseline_summary.get("overall_pass_rate"),
        candidate=candidate_summary.get("overall_pass_rate"),
        higher_is_better=True,
        unit="rate",
    )

    risk_metrics = [
        _compare_scalar(
            name="False confident rate",
            baseline=baseline_summary.get("false_confident_rate"),
            candidate=candidate_summary.get("false_confident_rate"),
            higher_is_better=False,
            unit="rate",
        ),
        _compare_scalar(
            name="Citation correctness",
            baseline=baseline_summary.get("citation_correctness_rate"),
            candidate=candidate_summary.get("citation_correctness_rate"),
            higher_is_better=True,
            unit="rate",
        ),
    ]

    perf_metrics = [
        _compare_scalar(
            name="Latency p50",
            baseline=_nested_number(baseline_summary, "latency", "p50"),
            candidate=_nested_number(candidate_summary, "latency", "p50"),
            higher_is_better=False,
            unit="ms",
        ),
        _compare_scalar(
            name="Latency p95",
            baseline=_nested_number(baseline_summary, "latency", "p95"),
            candidate=_nested_number(candidate_summary, "latency", "p95"),
            higher_is_better=False,
            unit="ms",
        ),
        _compare_scalar(
            name="Cost / request",
            baseline=_nested_number(baseline_summary, "cost", "avg"),
            candidate=_nested_number(candidate_summary, "cost", "avg"),
            higher_is_better=False,
            unit="usd",
        ),
        _compare_scalar(
            name="Total run cost",
            baseline=_nested_number(baseline_summary, "cost", "total"),
            candidate=_nested_number(candidate_summary, "cost", "total"),
            higher_is_better=False,
            unit="usd",
        ),
    ]

    family_rows = _compare_families(
        baseline=baseline_summary.get("family_scores") or {},
        candidate=candidate_summary.get("family_scores") or {},
    )

    unavailable_metrics = [
        metric.name for metric in [overall_cmp, *risk_metrics, *perf_metrics] if metric.direction == "unavailable"
    ]

    improved = [metric.name for metric in [overall_cmp, *risk_metrics, *perf_metrics] if metric.direction == "improved"]
    regressed = [metric.name for metric in [overall_cmp, *risk_metrics, *perf_metrics] if metric.direction == "regressed"]

    family_improved = [row["family"] for row in family_rows if row["status"] == "improved"]
    family_regressed = [row["family"] for row in family_rows if row["status"] == "regressed"]

    verdict = _quick_verdict(
        overall=overall_cmp.direction,
        risk_metrics=risk_metrics,
        perf_metrics=perf_metrics,
        family_regressions=len(family_regressed),
    )

    return {
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "overall": overall_cmp,
        "risk_metrics": risk_metrics,
        "performance_metrics": perf_metrics,
        "family_comparison": family_rows,
        "summary": {
            "verdict": verdict,
            "improved_metrics": improved,
            "regressed_metrics": regressed,
            "improved_families": family_improved,
            "regressed_families": family_regressed,
            "metrics_unavailable": unavailable_metrics,
        },
    }


def render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    overall: MetricComparison = comparison["overall"]
    summary = comparison["summary"]

    lines = [
        "# Release Comparison Report",
        "",
        "## Header",
        f"- Baseline: `{baseline.get('run_id')}` ({baseline.get('source_path')})",
        f"- Candidate: `{candidate.get('run_id')}` ({candidate.get('source_path')})",
        f"- Baseline timestamp: {baseline.get('timestamp') or 'not available'}",
        f"- Candidate timestamp: {candidate.get('timestamp') or 'not available'}",
        f"- Quick verdict: **{summary.get('verdict')}**",
        "",
        "## Overall summary",
        f"- Overall score: {_fmt_metric(overall)}",
        f"- Key improvements: {_join_or_none(summary.get('improved_metrics') or [])}",
        f"- Key regressions: {_join_or_none(summary.get('regressed_metrics') or [])}",
        f"- Metrics unavailable: {_join_or_none(summary.get('metrics_unavailable') or [])}",
        "",
        "## Family comparison",
        "| Family | Baseline | Candidate | Delta | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for row in comparison.get("family_comparison", []):
        lines.append(
            f"| `{row['family']}` | {_fmt_rate(row['baseline'])} | {_fmt_rate(row['candidate'])} | {_fmt_rate(row['delta'], signed=True)} | {row['status']} |"
        )

    if not comparison.get("family_comparison"):
        lines.append("| _none_ | n/a | n/a | n/a | unavailable |")

    lines.extend([
        "",
        "## Risk metrics",
        "| Metric | Baseline | Candidate | Delta | Delta % | Trend |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for metric in comparison.get("risk_metrics", []):
        lines.append(_metric_row(metric))

    lines.extend([
        "",
        "## Performance / efficiency",
        "| Metric | Baseline | Candidate | Delta | Delta % | Trend |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for metric in comparison.get("performance_metrics", []):
        lines.append(_metric_row(metric))

    improved_families = summary.get("improved_families") or []
    regressed_families = summary.get("regressed_families") or []
    lines.extend(
        [
            "",
            "## Release notes summary",
            f"- Improved: {_release_note_phrase(summary.get('improved_metrics') or [], improved_families)}",
            f"- Regressed: {_release_note_phrase(summary.get('regressed_metrics') or [], regressed_families)}",
            f"- Needs review: {_join_or_none(summary.get('metrics_unavailable') or [])}",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def write_comparison_report(
    *, baseline: str | Path, candidate: str | Path, output_path: str | Path | None = None
) -> tuple[dict[str, Any], str]:
    comparison = compare_runs(baseline=baseline, candidate=candidate)
    markdown = render_comparison_markdown(comparison)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    return comparison, markdown


def _load_run_blob(run_spec: str | Path) -> dict[str, Any]:
    path = _resolve_run_path(run_spec)
    blob = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        raise ValueError(f"run output must be a JSON object: {path}")
    blob["_source_path"] = str(path)
    return blob


def _resolve_run_path(run_spec: str | Path) -> Path:
    spec_text = str(run_spec)
    direct = Path(spec_text)
    if direct.exists():
        return _resolve_file_from_path(direct)

    runs_dir = Path("evals/runs")
    if runs_dir.is_dir():
        candidate = runs_dir / spec_text
        if candidate.exists():
            return _resolve_file_from_path(candidate)

        json_candidate = runs_dir / f"{spec_text}.json"
        if json_candidate.exists():
            return json_candidate

    raise FileNotFoundError(f"unable to resolve run artifact: {run_spec}")


def _resolve_file_from_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        files = sorted(path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"no JSON files found under {path}")
        return files[0]
    raise FileNotFoundError(f"path is neither file nor directory: {path}")


def _summarize_run(blob: Mapping[str, Any]) -> dict[str, Any]:
    cases = [item for item in (blob.get("cases") or []) if isinstance(item, Mapping)]
    family_scores = _family_scores(cases)
    pass_count = sum(1 for case in cases if _case_passed(case))

    false_values = [value for value in (_false_confident_value(case) for case in cases) if value is not None]
    citation_values = [value for value in (_citation_correctness_value(case) for case in cases) if value is not None]

    latency_values = _extract_numeric_series(cases, ["latency_ms", "latency", "duration_ms", "elapsed_ms"])
    cost_values = _extract_numeric_series(cases, ["cost_usd", "cost", "usd_cost"])

    run_id = str(blob.get("run_id") or blob.get("id") or blob.get("generated_at_utc") or Path(str(blob.get("_source_path"))).stem)
    timestamp = _parse_timestamp(blob.get("generated_at_utc"))

    return {
        "source_path": str(blob.get("_source_path") or ""),
        "run_id": run_id,
        "timestamp": timestamp.isoformat() if timestamp else None,
        "case_count": len(cases),
        "overall_pass_rate": _safe_rate(pass_count, len(cases)),
        "family_scores": family_scores,
        "false_confident_rate": (sum(false_values) / len(false_values)) if false_values else None,
        "citation_correctness_rate": (sum(citation_values) / len(citation_values)) if citation_values else None,
        "latency": _series_summary(latency_values, unit="ms"),
        "cost": _series_summary(cost_values, unit="usd"),
    }


def _compare_families(*, baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted(set(baseline.keys()) | set(candidate.keys())):
        b = baseline.get(family) if isinstance(baseline.get(family), Mapping) else {}
        c = candidate.get(family) if isinstance(candidate.get(family), Mapping) else {}

        b_rate = _to_float(b.get("pass_rate"))
        c_rate = _to_float(c.get("pass_rate"))

        delta = None if b_rate is None or c_rate is None else c_rate - b_rate
        status = _trend(delta=delta, higher_is_better=True)
        rows.append(
            {
                "family": family,
                "baseline": b_rate,
                "candidate": c_rate,
                "delta": delta,
                "status": status,
                "baseline_cases": int(b.get("case_count", 0) or 0),
                "candidate_cases": int(c.get("case_count", 0) or 0),
            }
        )
    return rows


def _compare_scalar(*, name: str, baseline: Any, candidate: Any, higher_is_better: bool, unit: str) -> MetricComparison:
    base = _to_float(baseline)
    cand = _to_float(candidate)
    if base is None or cand is None:
        return MetricComparison(
            name=name,
            baseline=base,
            candidate=cand,
            delta=None,
            delta_pct=None,
            direction="unavailable",
            higher_is_better=higher_is_better,
            unit=unit,
        )

    delta = cand - base
    delta_pct = (delta / abs(base)) if abs(base) > 0 else None
    return MetricComparison(
        name=name,
        baseline=base,
        candidate=cand,
        delta=delta,
        delta_pct=delta_pct,
        direction=_trend(delta=delta, higher_is_better=higher_is_better),
        higher_is_better=higher_is_better,
        unit=unit,
    )


def _quick_verdict(*, overall: str, risk_metrics: Sequence[MetricComparison], perf_metrics: Sequence[MetricComparison], family_regressions: int) -> str:
    risk_regressions = sum(1 for metric in risk_metrics if metric.direction == "regressed")
    perf_regressions = sum(1 for metric in perf_metrics if metric.direction == "regressed")

    if overall == "regressed" or risk_regressions > 0 or family_regressions > 0:
        return "Do not release without review"
    if overall == "improved" and perf_regressions == 0:
        return "Release candidate looks better"
    return "Mixed results — review deltas"


def _family_scores(cases: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[bool]] = {}
    for case in cases:
        family = str(case.get("family") or "unknown")
        buckets.setdefault(family, []).append(_case_passed(case))

    out: dict[str, dict[str, Any]] = {}
    for family, outcomes in sorted(buckets.items()):
        case_count = len(outcomes)
        pass_count = sum(1 for value in outcomes if value)
        out[family] = {
            "case_count": case_count,
            "pass_rate": _safe_rate(pass_count, case_count),
        }
    return out


def _case_passed(case: Mapping[str, Any]) -> bool:
    if str(case.get("runner_status") or "") != "ok":
        return False
    for group_key in ("deterministic_eval_results", "llm_judge_results"):
        group = case.get(group_key)
        if not isinstance(group, Mapping):
            continue
        for result in group.values():
            if not isinstance(result, Mapping):
                continue
            if result.get("status") == "skipped":
                continue
            if result.get("status") == "error":
                return False
            passed = _resolve_passed(result)
            if passed is False:
                return False
    return True


def _false_confident_value(case: Mapping[str, Any]) -> float | None:
    answerability = _nested_mapping(case, "deterministic_eval_results", "answerability_checks")
    if answerability is None:
        return None

    fields = answerability.get("aggregation_fields") if isinstance(answerability.get("aggregation_fields"), Mapping) else {}
    false_positive = fields.get("false_positive")
    if isinstance(false_positive, bool):
        return 1.0 if false_positive else 0.0

    classification = str(answerability.get("classification") or "")
    if classification:
        return 1.0 if classification == "false_positive" else 0.0
    return None


def _citation_correctness_value(case: Mapping[str, Any]) -> float | None:
    citation = _nested_mapping(case, "deterministic_eval_results", "citation_checks")
    if citation is None:
        return None

    fields = citation.get("aggregation_fields") if isinstance(citation.get("aggregation_fields"), Mapping) else {}
    match_pass = fields.get("citation_support_match_pass")
    if isinstance(match_pass, bool):
        return 1.0 if match_pass else 0.0

    passed = citation.get("passed")
    if isinstance(passed, bool):
        return 1.0 if passed else 0.0
    return None


def _extract_numeric_series(cases: Sequence[Mapping[str, Any]], candidate_keys: Sequence[str]) -> list[float]:
    values: list[float] = []
    for case in cases:
        found = _find_numeric(case, candidate_keys)
        if found is not None:
            values.append(found)
    return values


def _find_numeric(payload: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    nested = [payload]
    for key in ("system_result", "debug_payload", "metrics", "metadata"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            nested.append(value)

    for mapping in nested:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _series_summary(values: Sequence[float], *, unit: str) -> dict[str, Any]:
    if not values:
        return {"available": False, "message": "not available", "unit": unit}

    sorted_values = sorted(values)
    p50_idx = max(0, int(0.50 * (len(sorted_values) - 1)))
    p95_idx = max(0, int(0.95 * (len(sorted_values) - 1)))

    return {
        "available": True,
        "count": len(values),
        "avg": mean(values),
        "p50": sorted_values[p50_idx],
        "p95": sorted_values[p95_idx],
        "total": sum(values),
        "unit": unit,
    }


def _resolve_passed(result: Mapping[str, Any]) -> bool | None:
    for key in ("passed", "is_correct"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
    return None


def _nested_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _nested_number(payload: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _to_float(current)


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _trend(*, delta: float | None, higher_is_better: bool) -> str:
    if delta is None:
        return "unavailable"
    if abs(delta) < 1e-12:
        return "unchanged"
    if higher_is_better:
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _fmt_metric(metric: MetricComparison) -> str:
    if metric.direction == "unavailable":
        return "unavailable"
    return (
        f"{_format_value(metric.baseline, metric.unit)} → {_format_value(metric.candidate, metric.unit)} "
        f"({_format_signed(metric.delta, metric.unit)}, {metric.direction})"
    )


def _metric_row(metric: MetricComparison) -> str:
    return (
        f"| {metric.name} | {_format_value(metric.baseline, metric.unit)} | {_format_value(metric.candidate, metric.unit)} "
        f"| {_format_signed(metric.delta, metric.unit)} | {_format_pct(metric.delta_pct)} | {metric.direction} |"
    )


def _format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "unavailable"
    if unit == "rate":
        return _fmt_rate(value)
    if unit == "usd":
        return f"${value:.4f}"
    return f"{value:.2f} {unit}".strip()


def _format_signed(value: float | None, unit: str) -> str:
    if value is None:
        return "unavailable"
    if unit == "rate":
        return _fmt_rate(value, signed=True)
    if unit == "usd":
        return f"${value:+.4f}"
    return f"{value:+.2f} {unit}".strip()


def _fmt_rate(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{value * 100:+.2f}%" if signed else f"{value * 100:.2f}%"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}%"


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "none"


def _release_note_phrase(metric_names: Sequence[str], families: Sequence[str]) -> str:
    fragments: list[str] = []
    if metric_names:
        fragments.append(", ".join(metric_names))
    if families:
        fragments.append(f"families: {', '.join(families)}")
    return "; ".join(fragments) if fragments else "none"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two offline eval runs and generate a release decision report.")
    parser.add_argument("--baseline", required=True, help="Baseline run JSON file, run directory, or run id under evals/runs.")
    parser.add_argument("--candidate", required=True, help="Candidate run JSON file, run directory, or run id under evals/runs.")
    parser.add_argument("--output", help="Optional output markdown path. If omitted, report is printed to stdout.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _, markdown = write_comparison_report(
        baseline=args.baseline,
        candidate=args.candidate,
        output_path=args.output,
    )
    if args.output is None:
        print(markdown)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
