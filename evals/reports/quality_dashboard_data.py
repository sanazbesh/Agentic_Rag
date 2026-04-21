from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class EvalRun:
    run_id: str
    source_path: Path
    generated_at_utc: datetime | None
    run_order: int
    cases: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RunMetricPoint:
    run_id: str
    run_order: int
    timestamp: datetime | None
    false_confident_rate: float | None
    citation_correctness_rate: float | None
    safe_failure_rate: float | None
    pass_rate: float
    case_count: int


def discover_run_files(run_dir: str | Path) -> list[Path]:
    directory = Path(run_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted([path for path in directory.glob("*.json") if path.is_file()])


def load_eval_runs(run_files: Sequence[str | Path]) -> list[EvalRun]:
    rows: list[EvalRun] = []
    for order, run_file in enumerate(sorted(Path(item) for item in run_files), start=1):
        blob = json.loads(run_file.read_text(encoding="utf-8"))
        if not isinstance(blob, Mapping):
            continue
        cases = [dict(case) for case in (blob.get("cases") or []) if isinstance(case, Mapping)]
        generated_at = _parse_timestamp(blob.get("generated_at_utc"))
        run_id = str(blob.get("generated_at_utc") or blob.get("run_id") or run_file.stem)
        rows.append(
            EvalRun(
                run_id=run_id,
                source_path=run_file,
                generated_at_utc=generated_at,
                run_order=order,
                cases=cases,
            )
        )
    rows.sort(key=lambda row: (row.generated_at_utc or datetime.min.replace(tzinfo=timezone.utc), row.run_order))
    normalized: list[EvalRun] = []
    for idx, row in enumerate(rows, start=1):
        normalized.append(
            EvalRun(
                run_id=row.run_id,
                source_path=row.source_path,
                generated_at_utc=row.generated_at_utc,
                run_order=idx,
                cases=row.cases,
            )
        )
    return normalized


def select_recent_runs(runs: Sequence[EvalRun], *, last_n: int | None) -> list[EvalRun]:
    if last_n is None or last_n <= 0 or last_n >= len(runs):
        return list(runs)
    return list(runs[-last_n:])


def filter_cases_by_family(cases: Sequence[Mapping[str, Any]], families: Sequence[str] | None) -> list[dict[str, Any]]:
    if not families:
        return [dict(case) for case in cases]
    selected = set(families)
    return [dict(case) for case in cases if str(case.get("family") or "unknown") in selected]


def compute_family_pass_rates(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[bool]] = {}
    for case in cases:
        family = str(case.get("family") or "unknown")
        by_family.setdefault(family, []).append(_case_passed(case))

    rows: list[dict[str, Any]] = []
    for family, outcomes in sorted(by_family.items()):
        case_count = len(outcomes)
        pass_count = sum(1 for passed in outcomes if passed)
        fail_count = case_count - pass_count
        rows.append(
            {
                "family": family,
                "case_count": case_count,
                "pass_rate": _safe_rate(pass_count, case_count),
                "fail_count": fail_count,
                "fail_rate": _safe_rate(fail_count, case_count),
            }
        )
    return rows


def compute_run_trends(runs: Sequence[EvalRun], *, families: Sequence[str] | None = None) -> list[RunMetricPoint]:
    points: list[RunMetricPoint] = []
    for run in runs:
        cases = filter_cases_by_family(run.cases, families)
        pass_count = sum(1 for case in cases if _case_passed(case))
        false_values = [value for value in (_false_confident_value(case) for case in cases) if value is not None]
        citation_values = [value for value in (_citation_correctness_value(case) for case in cases) if value is not None]
        safe_values = [value for value in (_safe_failure_value(case) for case in cases) if value is not None]

        points.append(
            RunMetricPoint(
                run_id=run.run_id,
                run_order=run.run_order,
                timestamp=run.generated_at_utc,
                false_confident_rate=(sum(false_values) / len(false_values)) if false_values else None,
                citation_correctness_rate=(sum(citation_values) / len(citation_values)) if citation_values else None,
                safe_failure_rate=(sum(safe_values) / len(safe_values)) if safe_values else None,
                pass_rate=_safe_rate(pass_count, len(cases)),
                case_count=len(cases),
            )
        )
    return points


def compute_top_failing_datasets(cases: Sequence[Mapping[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, int]] = {}
    for case in cases:
        dataset = str(case.get("_dataset_file") or case.get("dataset") or case.get("family") or "unknown")
        bucket = buckets.setdefault(dataset, {"case_count": 0, "fail_count": 0})
        bucket["case_count"] += 1
        if not _case_passed(case):
            bucket["fail_count"] += 1

    rows = []
    for dataset, counts in buckets.items():
        rows.append(
            {
                "dataset": dataset,
                "case_count": counts["case_count"],
                "fail_count": counts["fail_count"],
                "fail_rate": _safe_rate(counts["fail_count"], counts["case_count"]),
            }
        )
    rows.sort(key=lambda row: (row["fail_count"], row["fail_rate"]), reverse=True)
    return rows[:limit]


def compare_runs(baseline: EvalRun, candidate: EvalRun, *, families: Sequence[str] | None = None) -> dict[str, Any]:
    baseline_cases = filter_cases_by_family(baseline.cases, families)
    candidate_cases = filter_cases_by_family(candidate.cases, families)

    baseline_trend = compute_run_trends([
        EvalRun(
            run_id=baseline.run_id,
            source_path=baseline.source_path,
            generated_at_utc=baseline.generated_at_utc,
            run_order=baseline.run_order,
            cases=baseline_cases,
        )
    ])[0]
    candidate_trend = compute_run_trends([
        EvalRun(
            run_id=candidate.run_id,
            source_path=candidate.source_path,
            generated_at_utc=candidate.generated_at_utc,
            run_order=candidate.run_order,
            cases=candidate_cases,
        )
    ])[0]

    family_baseline = {row["family"]: row for row in compute_family_pass_rates(baseline_cases)}
    family_candidate = {row["family"]: row for row in compute_family_pass_rates(candidate_cases)}

    family_deltas: list[dict[str, Any]] = []
    for family in sorted(set(family_baseline) | set(family_candidate)):
        b = family_baseline.get(family, {"pass_rate": 0.0, "case_count": 0})
        c = family_candidate.get(family, {"pass_rate": 0.0, "case_count": 0})
        family_deltas.append(
            {
                "family": family,
                "baseline_pass_rate": float(b.get("pass_rate", 0.0)),
                "candidate_pass_rate": float(c.get("pass_rate", 0.0)),
                "delta": float(c.get("pass_rate", 0.0)) - float(b.get("pass_rate", 0.0)),
                "baseline_cases": int(b.get("case_count", 0)),
                "candidate_cases": int(c.get("case_count", 0)),
            }
        )

    baseline_fail = {row["dataset"]: row for row in compute_top_failing_datasets(baseline_cases, limit=1000)}
    candidate_fail = {row["dataset"]: row for row in compute_top_failing_datasets(candidate_cases, limit=1000)}
    regressions = []
    for dataset in sorted(set(baseline_fail) | set(candidate_fail)):
        b = baseline_fail.get(dataset, {"fail_count": 0, "fail_rate": 0.0})
        c = candidate_fail.get(dataset, {"fail_count": 0, "fail_rate": 0.0})
        fail_delta = int(c["fail_count"]) - int(b["fail_count"])
        if fail_delta > 0:
            regressions.append(
                {
                    "dataset": dataset,
                    "baseline_fail_count": b["fail_count"],
                    "candidate_fail_count": c["fail_count"],
                    "fail_count_delta": fail_delta,
                    "baseline_fail_rate": b["fail_rate"],
                    "candidate_fail_rate": c["fail_rate"],
                }
            )
    regressions.sort(key=lambda row: row["fail_count_delta"], reverse=True)

    return {
        "baseline": baseline.run_id,
        "candidate": candidate.run_id,
        "overall_delta": candidate_trend.pass_rate - baseline_trend.pass_rate,
        "false_confident_delta": _delta(candidate_trend.false_confident_rate, baseline_trend.false_confident_rate),
        "citation_correctness_delta": _delta(
            candidate_trend.citation_correctness_rate,
            baseline_trend.citation_correctness_rate,
        ),
        "safe_failure_delta": _delta(candidate_trend.safe_failure_rate, baseline_trend.safe_failure_rate),
        "family_deltas": family_deltas,
        "top_regressions": regressions[:10],
    }


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
            if result.get("status") in {"skipped"}:
                continue
            if result.get("status") == "error":
                return False
            if _resolve_passed(result) is False:
                return False
    return True


def _resolve_passed(result: Mapping[str, Any]) -> bool | None:
    for key in ("passed", "is_correct"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
    return None


def _false_confident_value(case: Mapping[str, Any]) -> float | None:
    answerability = _nested_mapping(case, "deterministic_eval_results", "answerability_checks")
    if answerability is None:
        return None
    fields = answerability.get("aggregation_fields") if isinstance(answerability.get("aggregation_fields"), Mapping) else {}
    if isinstance(fields.get("false_positive"), bool):
        return 1.0 if fields["false_positive"] else 0.0
    classification = str(answerability.get("classification") or "")
    if classification:
        return 1.0 if classification == "false_positive" else 0.0
    return None


def _citation_correctness_value(case: Mapping[str, Any]) -> float | None:
    citation = _nested_mapping(case, "deterministic_eval_results", "citation_checks")
    if citation is None:
        return None
    fields = citation.get("aggregation_fields") if isinstance(citation.get("aggregation_fields"), Mapping) else {}
    value = fields.get("citation_support_match_pass")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    passed = citation.get("passed")
    if isinstance(passed, bool):
        return 1.0 if passed else 0.0
    return None


def _safe_failure_value(case: Mapping[str, Any]) -> float | None:
    answerability = _nested_mapping(case, "deterministic_eval_results", "answerability_checks")
    if isinstance(answerability, Mapping):
        fields = answerability.get("aggregation_fields") if isinstance(answerability.get("aggregation_fields"), Mapping) else {}
        metric = fields.get("safe_failure_quality_pass")
        if isinstance(metric, bool):
            return 1.0 if metric else 0.0

    safe_failure = _nested_mapping(case, "llm_judge_results", "safe_failure")
    if isinstance(safe_failure, Mapping):
        passed = safe_failure.get("passed")
        if isinstance(passed, bool):
            return 1.0 if passed else 0.0
    return None


def _nested_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current if isinstance(current, Mapping) else None


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


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
