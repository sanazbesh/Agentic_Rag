"""Deterministic family-routing evaluator for legal RAG offline evals.

This evaluator stays intentionally lean for a solo-maintained portfolio project:
it scores whether the predicted legal family matches the eval-case family and
provides confusion-matrix friendly outputs for aggregation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_FAMILY_NOTE_PREFIX = "legal_question_family:"

# Query-understanding labels are not always identical to eval dataset labels.
# Keep this mapping explicit and deterministic.
_QUERY_FAMILY_TO_EVAL_FAMILY = {
    "party_role_entity": "party_role_verification",
    "chronology_date_event": "chronology_date_event",
    "matter_document_metadata": "matter_document_metadata",
    "employment_contract_lifecycle": "employment_lifecycle",
    "employment_mitigation": "employment_mitigation",
    "financial_entitlement": "financial_entitlement",
    "policy_issue_spotting": "policy_issue_spotting",
    "correspondence_litigation_milestone": "correspondence_litigation_milestones",
}

_DIRECT_FAMILY_KEYS = (
    "predicted_family",
    "family",
    "legal_family",
    "legal_question_family",
)


@dataclass(frozen=True, slots=True)
class FamilyRoutingEvaluationResult:
    """Per-case deterministic routing-eval output."""

    evaluator_name: str
    case_id: str
    case_family: str
    expected_family: str
    predicted_family: str | None
    is_correct: bool
    confusion_cell: dict[str, Any]
    aggregation_fields: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "case_id": self.case_id,
            "case_family": self.case_family,
            "expected_family": self.expected_family,
            "predicted_family": self.predicted_family,
            "is_correct": self.is_correct,
            "confusion_cell": self.confusion_cell,
            "aggregation_fields": self.aggregation_fields,
            "metadata": self.metadata,
        }


def evaluate_family_routing(
    *,
    eval_case: Mapping[str, Any],
    system_output: Mapping[str, Any],
) -> FamilyRoutingEvaluationResult:
    """Score family routing for one eval case + one runtime output payload."""

    case_id = str(eval_case.get("id") or "")
    expected_family = str(eval_case.get("family") or "unknown")
    predicted_family, resolution_source = _resolve_predicted_family(system_output)
    is_correct = predicted_family == expected_family

    return FamilyRoutingEvaluationResult(
        evaluator_name="family_routing_v1",
        case_id=case_id,
        case_family=expected_family,
        expected_family=expected_family,
        predicted_family=predicted_family,
        is_correct=is_correct,
        confusion_cell={
            "expected_family": expected_family,
            "predicted_family": predicted_family or "unresolved",
            "count": 1,
            "correct": is_correct,
        },
        aggregation_fields={
            "case_id": case_id,
            "family": expected_family,
            "expected_family": expected_family,
            "predicted_family": predicted_family,
            "routing_correct": is_correct,
            "confusion_pair": f"{expected_family}->{predicted_family or 'unresolved'}",
        },
        metadata={
            "deterministic": True,
            "model_based_judgment": False,
            "predicted_family_resolution_source": resolution_source,
            "uses_query_family_mapping": True,
        },
    )


def build_family_confusion_matrix(
    results: Sequence[FamilyRoutingEvaluationResult | Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """Build an explicit expected-vs-predicted family confusion matrix."""

    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        payload = result.to_dict() if isinstance(result, FamilyRoutingEvaluationResult) else dict(result)
        expected = str(payload.get("expected_family") or payload.get("case_family") or "unknown")
        predicted_raw = payload.get("predicted_family")
        predicted = str(predicted_raw) if isinstance(predicted_raw, str) and predicted_raw else "unresolved"
        matrix[expected][predicted] += 1

    return {expected: dict(sorted(predicted_counts.items())) for expected, predicted_counts in sorted(matrix.items())}


def aggregate_family_routing_results(
    results: Sequence[FamilyRoutingEvaluationResult | Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate routing results with per-family stats and confusion matrix."""

    normalized = [result.to_dict() if isinstance(result, FamilyRoutingEvaluationResult) else dict(result) for result in results]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in normalized:
        family = str(payload.get("expected_family") or payload.get("case_family") or "unknown")
        buckets[family].append(payload)

    by_expected_family: dict[str, dict[str, Any]] = {}
    for family, items in sorted(buckets.items()):
        correct_count = sum(1 for item in items if item.get("is_correct") is True)
        by_expected_family[family] = {
            "expected_family": family,
            "case_count": len(items),
            "correct_count": correct_count,
            "accuracy": (correct_count / len(items)) if items else 0.0,
            "confusion_row": build_family_confusion_matrix(items).get(family, {}),
        }

    total = len(normalized)
    correct_total = sum(1 for item in normalized if item.get("is_correct") is True)
    return {
        "case_count": total,
        "correct_count": correct_total,
        "accuracy": (correct_total / total) if total else 0.0,
        "by_expected_family": by_expected_family,
        "family_confusion_matrix": build_family_confusion_matrix(normalized),
    }


def _resolve_predicted_family(system_output: Mapping[str, Any]) -> tuple[str | None, str]:
    for key in _DIRECT_FAMILY_KEYS:
        candidate = _normalize_family_value(system_output.get(key))
        if candidate is not None:
            return candidate, f"system_output.{key}"

    query_classification = system_output.get("query_classification")
    if isinstance(query_classification, Mapping):
        for key in _DIRECT_FAMILY_KEYS:
            candidate = _normalize_family_value(query_classification.get(key))
            if candidate is not None:
                return candidate, f"system_output.query_classification.{key}"

        routed = _extract_from_routing_notes(query_classification.get("routing_notes"))
        if routed is not None:
            return routed, "system_output.query_classification.routing_notes"

    routed = _extract_from_routing_notes(system_output.get("routing_notes"))
    if routed is not None:
        return routed, "system_output.routing_notes"

    return None, "unresolved"


def _normalize_family_value(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str):
        return None
    value = raw_value.strip()
    if not value:
        return None
    if value.startswith(_FAMILY_NOTE_PREFIX):
        value = value[len(_FAMILY_NOTE_PREFIX) :]
    return _QUERY_FAMILY_TO_EVAL_FAMILY.get(value, value)


def _extract_from_routing_notes(raw_notes: Any) -> str | None:
    if not isinstance(raw_notes, Sequence) or isinstance(raw_notes, (str, bytes, bytearray)):
        return None
    for note in raw_notes:
        normalized = _normalize_family_value(note)
        if normalized is not None and isinstance(note, str) and note.startswith(_FAMILY_NOTE_PREFIX):
            return normalized
    return None


__all__ = [
    "FamilyRoutingEvaluationResult",
    "aggregate_family_routing_results",
    "build_family_confusion_matrix",
    "evaluate_family_routing",
]
