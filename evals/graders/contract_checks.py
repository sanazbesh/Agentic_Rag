"""Deterministic contract and shape checks for legal RAG outputs.

This module is intentionally narrow: it validates output structure only and does
not make semantic quality judgments.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agentic_rag.orchestration.legal_rag_graph import FinalAnswerModel

EvaluatorCheckName = Literal[
    "valid_final_answer_contract",
    "citations_present_when_grounded",
    "citations_empty_when_unsupported",
    "selected_document_scope_respected",
    "no_duplicate_warnings",
    "no_malformed_debug_payload",
]


@dataclass(frozen=True, slots=True)
class ContractCheckResult:
    """One machine-readable contract check result."""

    check_name: EvaluatorCheckName
    passed: bool
    failure_code: str | None
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ContractEvaluationResult:
    """Aggregate evaluator output consumable by offline eval runners."""

    evaluator_name: str
    passed: bool
    checks: list[ContractCheckResult]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to plain dict for JSONL/offline reporting."""

        return {
            "evaluator_name": self.evaluator_name,
            "passed": self.passed,
            "checks": [asdict(item) for item in self.checks],
            "metadata": self.metadata,
        }


def evaluate_contract_checks(
    *,
    final_result: Any,
    debug_payload: Mapping[str, Any] | None = None,
) -> ContractEvaluationResult:
    """Run deterministic contract checks for one completed system output."""

    checks: list[ContractCheckResult] = []
    parsed_final, contract_check = _check_valid_final_answer_contract(final_result)
    checks.append(contract_check)

    parsed_debug = dict(debug_payload) if isinstance(debug_payload, Mapping) else None

    checks.append(_check_citations_present_when_grounded(parsed_final=parsed_final, contract_ok=contract_check.passed))
    checks.append(_check_citations_empty_when_unsupported(parsed_final=parsed_final, contract_ok=contract_check.passed))
    checks.append(
        _check_selected_document_scope_respected(
            parsed_final=parsed_final,
            parsed_debug=parsed_debug,
            contract_ok=contract_check.passed,
        )
    )
    checks.append(_check_no_duplicate_warnings(parsed_final=parsed_final, parsed_debug=parsed_debug, contract_ok=contract_check.passed))
    checks.append(_check_no_malformed_debug_payload(parsed_debug=parsed_debug, raw_debug=debug_payload))

    return ContractEvaluationResult(
        evaluator_name="contract_checks_v1",
        passed=all(item.passed for item in checks),
        checks=checks,
        metadata={
            "deterministic": True,
            "model_based_judgment": False,
            "check_count": len(checks),
        },
    )


def _check_valid_final_answer_contract(final_result: Any) -> tuple[FinalAnswerModel | None, ContractCheckResult]:
    if not isinstance(final_result, Mapping):
        return None, ContractCheckResult(
            check_name="valid_final_answer_contract",
            passed=False,
            failure_code="final_result_not_mapping",
            message="Final result must be a mapping with the legal RAG final answer contract.",
            details={"received_type": type(final_result).__name__},
        )

    payload = dict(final_result)
    try:
        parsed = FinalAnswerModel.model_validate(payload)
    except Exception as exc:
        return None, ContractCheckResult(
            check_name="valid_final_answer_contract",
            passed=False,
            failure_code="final_answer_contract_invalid",
            message="Final result failed legal final-answer contract validation.",
            details={"error": str(exc)},
        )

    return parsed, ContractCheckResult(
        check_name="valid_final_answer_contract",
        passed=True,
        failure_code=None,
        message="Final answer contract is valid.",
    )


def _check_citations_present_when_grounded(
    *,
    parsed_final: FinalAnswerModel | None,
    contract_ok: bool,
) -> ContractCheckResult:
    if not contract_ok or parsed_final is None:
        return _skipped("citations_present_when_grounded", "Skipped because final contract is invalid.")

    if parsed_final.grounded and not parsed_final.citations:
        return ContractCheckResult(
            check_name="citations_present_when_grounded",
            passed=False,
            failure_code="grounded_without_citations",
            message="Grounded answers must include at least one citation.",
        )

    return ContractCheckResult(
        check_name="citations_present_when_grounded",
        passed=True,
        failure_code=None,
        message="Citation presence is consistent with grounded flag.",
    )


def _check_citations_empty_when_unsupported(
    *,
    parsed_final: FinalAnswerModel | None,
    contract_ok: bool,
) -> ContractCheckResult:
    if not contract_ok or parsed_final is None:
        return _skipped("citations_empty_when_unsupported", "Skipped because final contract is invalid.")

    # Repo semantics: ungrounded final responses should not claim evidence support.
    if (not parsed_final.grounded) and parsed_final.citations:
        return ContractCheckResult(
            check_name="citations_empty_when_unsupported",
            passed=False,
            failure_code="ungrounded_with_citations",
            message="Ungrounded/unsupported answers must not include citations.",
            details={"citation_count": len(parsed_final.citations)},
        )

    return ContractCheckResult(
        check_name="citations_empty_when_unsupported",
        passed=True,
        failure_code=None,
        message="Citation emptiness is consistent with unsupported/ungrounded output.",
    )


def _check_selected_document_scope_respected(
    *,
    parsed_final: FinalAnswerModel | None,
    parsed_debug: Mapping[str, Any] | None,
    contract_ok: bool,
) -> ContractCheckResult:
    if not contract_ok or parsed_final is None:
        return _skipped("selected_document_scope_respected", "Skipped because final contract is invalid.")

    selected_ids = _extract_selected_document_ids(parsed_debug)
    if not selected_ids:
        return ContractCheckResult(
            check_name="selected_document_scope_respected",
            passed=True,
            failure_code=None,
            message="No selected-document scope is active.",
        )

    out_of_scope_citations = sorted(
        {
            document_id
            for citation in parsed_final.citations
            for document_id in [_citation_document_id(citation)]
            if document_id and document_id not in selected_ids
        }
    )
    if out_of_scope_citations:
        return ContractCheckResult(
            check_name="selected_document_scope_respected",
            passed=False,
            failure_code="citation_outside_selected_document_scope",
            message="Final citations include documents outside selected scope.",
            details={"out_of_scope_document_ids": out_of_scope_citations, "selected_document_ids": sorted(selected_ids)},
        )

    resolved_scope = _extract_resolved_document_scope(parsed_debug)
    if resolved_scope and not resolved_scope.issubset(selected_ids):
        return ContractCheckResult(
            check_name="selected_document_scope_respected",
            passed=False,
            failure_code="resolved_scope_outside_selected_documents",
            message="Resolved document scope is inconsistent with selected-document scope.",
            details={
                "resolved_scope": sorted(resolved_scope),
                "selected_document_ids": sorted(selected_ids),
            },
        )

    return ContractCheckResult(
        check_name="selected_document_scope_respected",
        passed=True,
        failure_code=None,
        message="Selected-document scope is structurally respected.",
    )


def _check_no_duplicate_warnings(
    *,
    parsed_final: FinalAnswerModel | None,
    parsed_debug: Mapping[str, Any] | None,
    contract_ok: bool,
) -> ContractCheckResult:
    if not contract_ok or parsed_final is None:
        return _skipped("no_duplicate_warnings", "Skipped because final contract is invalid.")

    duplicate_final = _duplicates(parsed_final.warnings)
    duplicate_debug = _duplicates(parsed_debug.get("warnings", [])) if parsed_debug else []

    if duplicate_final or duplicate_debug:
        return ContractCheckResult(
            check_name="no_duplicate_warnings",
            passed=False,
            failure_code="duplicate_warnings_detected",
            message="Warnings contain exact duplicates.",
            details={
                "duplicate_final_warnings": duplicate_final,
                "duplicate_debug_warnings": duplicate_debug,
            },
        )

    return ContractCheckResult(
        check_name="no_duplicate_warnings",
        passed=True,
        failure_code=None,
        message="Warnings do not contain exact duplicates.",
    )


def _check_no_malformed_debug_payload(
    *,
    parsed_debug: Mapping[str, Any] | None,
    raw_debug: Any,
) -> ContractCheckResult:
    if raw_debug is None:
        return ContractCheckResult(
            check_name="no_malformed_debug_payload",
            passed=True,
            failure_code=None,
            message="Debug payload is absent (allowed by contract).",
        )

    if not isinstance(raw_debug, Mapping):
        return ContractCheckResult(
            check_name="no_malformed_debug_payload",
            passed=False,
            failure_code="debug_payload_not_mapping",
            message="Debug payload must be a mapping when present.",
            details={"received_type": type(raw_debug).__name__},
        )

    debug_payload = dict(parsed_debug or {})
    errors: list[str] = []

    warnings = debug_payload.get("warnings")
    if warnings is not None and (not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings)):
        errors.append("warnings must be list[str] when present")

    for key in ("meta", "scope", "adapter_meta"):
        value = debug_payload.get(key)
        if value is not None and not isinstance(value, Mapping):
            errors.append(f"{key} must be an object when present")

    resolved_scope = debug_payload.get("resolved_document_scope")
    if resolved_scope is not None and (
        not isinstance(resolved_scope, list) or any(not isinstance(item, str) for item in resolved_scope)
    ):
        errors.append("resolved_document_scope must be list[str] when present")

    answerability_result = debug_payload.get("answerability_result")
    if answerability_result is not None and not isinstance(answerability_result, Mapping):
        errors.append("answerability_result must be an object or null when present")

    if errors:
        return ContractCheckResult(
            check_name="no_malformed_debug_payload",
            passed=False,
            failure_code="debug_payload_malformed",
            message="Debug payload does not match expected high-level shape.",
            details={"errors": errors},
        )

    return ContractCheckResult(
        check_name="no_malformed_debug_payload",
        passed=True,
        failure_code=None,
        message="Debug payload shape is valid.",
    )


def _extract_selected_document_ids(parsed_debug: Mapping[str, Any] | None) -> set[str]:
    if not parsed_debug:
        return set()

    candidate_lists: list[Sequence[Any] | None] = []
    meta = parsed_debug.get("meta")
    if isinstance(meta, Mapping):
        candidate_lists.append(meta.get("selected_document_ids"))
    scope = parsed_debug.get("scope")
    if isinstance(scope, Mapping):
        candidate_lists.append(scope.get("selected_document_ids"))
    adapter_meta = parsed_debug.get("adapter_meta")
    if isinstance(adapter_meta, Mapping):
        candidate_lists.append(adapter_meta.get("selected_document_ids"))

    selected: set[str] = set()
    for values in candidate_lists:
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            selected.update(str(item) for item in values if isinstance(item, str) and item)
    return selected


def _extract_resolved_document_scope(parsed_debug: Mapping[str, Any] | None) -> set[str]:
    if not parsed_debug:
        return set()
    resolved = parsed_debug.get("resolved_document_scope")
    if not isinstance(resolved, list):
        return set()
    return {item for item in resolved if isinstance(item, str) and item}


def _duplicates(values: Sequence[Any]) -> list[str]:
    string_values = [item for item in values if isinstance(item, str)]
    counts = Counter(string_values)
    return sorted([value for value, count in counts.items() if count > 1])


def _skipped(check_name: EvaluatorCheckName, message: str) -> ContractCheckResult:
    return ContractCheckResult(
        check_name=check_name,
        passed=False,
        failure_code="skipped_due_to_invalid_contract",
        message=message,
    )


__all__ = [
    "ContractCheckResult",
    "ContractEvaluationResult",
    "evaluate_contract_checks",
]


def _citation_document_id(citation: Any) -> str | None:
    if isinstance(citation, Mapping):
        value = citation.get("document_id")
    else:
        value = getattr(citation, "document_id", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
