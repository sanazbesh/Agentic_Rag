"""Deterministic answerability checks for legal RAG eval cases.

This evaluator is intentionally narrow: it grades only answerability decisions
(supported vs insufficient/safe-failure) and does not perform semantic answer
correctness grading.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

DecisionCategory = Literal[
    "correct_supported_answer",
    "correct_insufficient_answer",
    "false_positive",
    "false_negative",
]
ExpectedAnswerability = Literal["answerable", "unanswerable"]
ObservedAnswerability = Literal["supported", "insufficient"]

_SAFE_FAILURE_OUTCOMES = {
    "safe_failure_insufficient_evidence",
    "safe_failure_ambiguous",
    "safe_failure_out_of_scope",
}


@dataclass(frozen=True, slots=True)
class AnswerabilityMetricResult:
    """One deterministic answerability metric outcome."""

    metric_name: str
    value: float | None
    passed: bool | None
    details: dict[str, Any]
    note: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerabilityEvaluationResult:
    """Per-case answerability evaluator output for offline runs."""

    evaluator_name: str
    case_id: str
    case_family: str
    classification: DecisionCategory
    passed: bool
    metrics: list[AnswerabilityMetricResult]
    notes: list[str]
    aggregation_fields: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "case_id": self.case_id,
            "case_family": self.case_family,
            "classification": self.classification,
            "passed": self.passed,
            "metrics": [asdict(metric) for metric in self.metrics],
            "notes": list(self.notes),
            "aggregation_fields": self.aggregation_fields,
            "metadata": self.metadata,
        }


def evaluate_answerability_checks(
    *,
    eval_case: Mapping[str, Any],
    final_result: Mapping[str, Any],
    debug_payload: Mapping[str, Any] | None = None,
) -> AnswerabilityEvaluationResult:
    """Evaluate deterministic answerability checks for one eval case + output."""

    case_id = str(eval_case.get("id") or "")
    case_family = str(eval_case.get("family") or "unknown")

    expected = _resolve_expected_answerability(eval_case)
    observed, observed_details = _resolve_observed_answerability(final_result=final_result, debug_payload=debug_payload)
    classification = _classify(expected=expected, observed=observed)

    false_confident = 1.0 if classification == "false_positive" else 0.0
    false_insufficient = 1.0 if classification == "false_negative" else 0.0

    safe_failure = _evaluate_safe_failure_quality(
        eval_case=eval_case,
        final_result=final_result,
        debug_payload=debug_payload,
        expected=expected,
        observed=observed,
        classification=classification,
    )

    family_correct = 1.0 if classification in {"correct_supported_answer", "correct_insufficient_answer"} else 0.0

    metrics = [
        AnswerabilityMetricResult(
            metric_name="false_confident_answer",
            value=false_confident,
            passed=false_confident == 0.0,
            details={"classification": classification},
        ),
        AnswerabilityMetricResult(
            metric_name="false_insufficient_answer",
            value=false_insufficient,
            passed=false_insufficient == 0.0,
            details={"classification": classification},
        ),
        AnswerabilityMetricResult(
            metric_name="safe_failure_quality",
            value=1.0 if safe_failure["passed"] else 0.0,
            passed=safe_failure["passed"],
            details=safe_failure,
        ),
        AnswerabilityMetricResult(
            metric_name="family_sufficiency_correct",
            value=family_correct,
            passed=family_correct == 1.0,
            details={"family": case_family, "classification": classification},
        ),
    ]

    notes: list[str] = []
    if safe_failure.get("marker_note"):
        notes.append(str(safe_failure["marker_note"]))

    return AnswerabilityEvaluationResult(
        evaluator_name="answerability_checks_v1",
        case_id=case_id,
        case_family=case_family,
        classification=classification,
        passed=classification in {"correct_supported_answer", "correct_insufficient_answer"},
        metrics=metrics,
        notes=notes,
        aggregation_fields={
            "case_id": case_id,
            "family": case_family,
            "classification": classification,
            "expected_answerability": expected,
            "observed_answerability": observed,
            "false_positive": classification == "false_positive",
            "false_negative": classification == "false_negative",
            "correct_supported": classification == "correct_supported_answer",
            "correct_insufficient": classification == "correct_insufficient_answer",
            "safe_failure_quality_pass": bool(safe_failure["passed"]),
            "family_sufficiency_correct": family_correct == 1.0,
        },
        metadata={
            "deterministic": True,
            "model_based_judgment": False,
            "expected_resolution_source": "eval_case",
            "observed_resolution_source": observed_details,
        },
    )


def aggregate_answerability_results_by_family(
    results: Sequence[AnswerabilityEvaluationResult | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-case answerability outputs by legal family."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        payload = result.to_dict() if isinstance(result, AnswerabilityEvaluationResult) else dict(result)
        fields = payload.get("aggregation_fields") or {}
        family = str(payload.get("case_family") or fields.get("family") or "unknown")
        buckets[family].append(payload)

    aggregated: dict[str, dict[str, Any]] = {}
    for family, items in sorted(buckets.items()):
        case_count = len(items)
        false_positive_count = 0
        false_negative_count = 0
        correct_supported_count = 0
        correct_insufficient_count = 0
        safe_failure_quality_pass_count = 0
        family_sufficiency_correct_count = 0

        for item in items:
            fields = item.get("aggregation_fields") or {}
            if fields.get("false_positive") is True:
                false_positive_count += 1
            if fields.get("false_negative") is True:
                false_negative_count += 1
            if fields.get("correct_supported") is True:
                correct_supported_count += 1
            if fields.get("correct_insufficient") is True:
                correct_insufficient_count += 1
            if fields.get("safe_failure_quality_pass") is True:
                safe_failure_quality_pass_count += 1
            if fields.get("family_sufficiency_correct") is True:
                family_sufficiency_correct_count += 1

        aggregated[family] = {
            "family": family,
            "case_count": case_count,
            "false_positive_count": false_positive_count,
            "false_positive_rate": _safe_rate(false_positive_count, case_count),
            "false_negative_count": false_negative_count,
            "false_negative_rate": _safe_rate(false_negative_count, case_count),
            "correct_supported_count": correct_supported_count,
            "correct_supported_rate": _safe_rate(correct_supported_count, case_count),
            "correct_insufficient_count": correct_insufficient_count,
            "correct_insufficient_rate": _safe_rate(correct_insufficient_count, case_count),
            "safe_failure_quality_pass_count": safe_failure_quality_pass_count,
            "safe_failure_quality_pass_rate": _safe_rate(safe_failure_quality_pass_count, case_count),
            "family_sufficiency_correct_count": family_sufficiency_correct_count,
            "family_sufficiency_correct_rate": _safe_rate(family_sufficiency_correct_count, case_count),
        }

    return aggregated


def _resolve_expected_answerability(eval_case: Mapping[str, Any]) -> ExpectedAnswerability:
    direct = eval_case.get("answerability_expected")
    if direct in {"answerable", "unanswerable"}:
        return direct

    expected_outcome = str(eval_case.get("expected_outcome") or "")
    if expected_outcome in _SAFE_FAILURE_OUTCOMES:
        return "unanswerable"

    safe_failure_expected = eval_case.get("safe_failure_expected")
    if isinstance(safe_failure_expected, bool):
        return "unanswerable" if safe_failure_expected else "answerable"

    return "answerable"


def _resolve_observed_answerability(
    *,
    final_result: Mapping[str, Any],
    debug_payload: Mapping[str, Any] | None,
) -> tuple[ObservedAnswerability, dict[str, Any]]:
    final_sufficient = bool(final_result.get("sufficient_context"))
    final_grounded = bool(final_result.get("grounded"))

    answerability_result = debug_payload.get("answerability_result") if isinstance(debug_payload, Mapping) else None
    debug_sufficient = bool(answerability_result.get("sufficient_context")) if isinstance(answerability_result, Mapping) else False
    debug_should_answer = bool(answerability_result.get("should_answer")) if isinstance(answerability_result, Mapping) else False

    supported = final_sufficient or final_grounded or debug_sufficient or debug_should_answer
    observed: ObservedAnswerability = "supported" if supported else "insufficient"

    return observed, {
        "used_final_result": True,
        "final_sufficient_context": final_sufficient,
        "final_grounded": final_grounded,
        "used_debug_answerability": isinstance(answerability_result, Mapping),
        "debug_sufficient_context": debug_sufficient,
        "debug_should_answer": debug_should_answer,
    }


def _classify(*, expected: ExpectedAnswerability, observed: ObservedAnswerability) -> DecisionCategory:
    if expected == "answerable" and observed == "supported":
        return "correct_supported_answer"
    if expected == "unanswerable" and observed == "insufficient":
        return "correct_insufficient_answer"
    if expected == "unanswerable" and observed == "supported":
        return "false_positive"
    return "false_negative"


def _evaluate_safe_failure_quality(
    *,
    eval_case: Mapping[str, Any],
    final_result: Mapping[str, Any],
    debug_payload: Mapping[str, Any] | None,
    expected: ExpectedAnswerability,
    observed: ObservedAnswerability,
    classification: DecisionCategory,
) -> dict[str, Any]:
    safe_failure_expected = bool(eval_case.get("safe_failure_expected"))
    expected_outcome = str(eval_case.get("expected_outcome") or "")

    answerability_result = debug_payload.get("answerability_result") if isinstance(debug_payload, Mapping) else None
    insufficiency_reason = (
        answerability_result.get("insufficiency_reason") if isinstance(answerability_result, Mapping) else None
    )
    warnings = _string_list(final_result.get("warnings"))

    contradictions: list[str] = []
    if bool(final_result.get("sufficient_context")) and not bool(final_result.get("grounded")):
        contradictions.append("supported_without_grounded")
    if (not bool(final_result.get("sufficient_context"))) and bool(final_result.get("grounded")):
        contradictions.append("grounded_with_insufficient_context")
    if (not bool(final_result.get("grounded"))) and bool(final_result.get("citations")):
        contradictions.append("citations_present_while_ungrounded")

    if safe_failure_expected and observed != "insufficient":
        contradictions.append("safe_failure_expected_but_system_marked_supported")
    if expected_outcome in _SAFE_FAILURE_OUTCOMES and observed != "insufficient":
        contradictions.append("expected_safe_failure_outcome_but_system_marked_supported")

    passed = True
    if expected == "unanswerable" and classification == "false_positive":
        passed = False
    if contradictions:
        passed = False

    marker_note: str | None = None
    if expected == "unanswerable" and not insufficiency_reason:
        has_insufficient_warning = any("insufficient" in warning.lower() for warning in warnings)
        if not has_insufficient_warning:
            marker_note = "No explicit insufficiency marker found in answerability_result or warnings."

    return {
        "passed": passed,
        "expected_safe_failure": safe_failure_expected,
        "expected_outcome": expected_outcome,
        "observed_answerability": observed,
        "insufficiency_reason_present": bool(insufficiency_reason),
        "contradictions": contradictions,
        "marker_note": marker_note,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if isinstance(item, str)]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
