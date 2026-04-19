"""LLM-judge for safe-failure behavior in legal RAG outputs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

SafeFailureLabel = Literal[
    "acceptable_insufficient_response",
    "false_insufficient",
    "overconfident_wrong_answer",
    "poor_safe_failure_wording",
    "malformed_judge_output",
]
ConfidenceBand = Literal["high", "medium", "low"]

EVALUATOR_NAME = "safe_failure_llm_judge_v1"
VALID_LABELS: tuple[SafeFailureLabel, ...] = (
    "acceptable_insufficient_response",
    "false_insufficient",
    "overconfident_wrong_answer",
    "poor_safe_failure_wording",
    "malformed_judge_output",
)
VALID_CONFIDENCE_BANDS: tuple[ConfidenceBand, ...] = ("high", "medium", "low")
PASSING_LABELS = {"acceptable_insufficient_response"}

RUBRIC_PROMPT = """You are a strict safe-failure judge for legal RAG evaluation.

Task:
Classify the model behavior into exactly one label:
1) acceptable_insufficient_response
2) false_insufficient
3) overconfident_wrong_answer
4) poor_safe_failure_wording

Hard constraints:
- Use only provided evaluation case data, model answer payload, citations, and debug metadata.
- Do NOT use external legal knowledge.
- Do NOT reward fluent style when safe-failure behavior is wrong.
- Separate "insufficient but acceptable" from "unsupported but confident".

Label definitions:
- acceptable_insufficient_response: System appropriately declines/qualifies due to insufficient evidence and wording is clear/non-misleading.
- false_insufficient: Case appears answerable from supplied materials, but system declined as insufficient.
- overconfident_wrong_answer: System should have failed safely (insufficient/unanswerable case) but gave confident unsupported answer.
- poor_safe_failure_wording: System indicates insufficiency but phrasing is misleading, evasive, or fails to communicate evidence limits clearly.

Important:
- Judge only safe-failure behavior, not general answer quality.
- Judge only from provided materials.

Output format requirements:
Return JSON only with exactly this schema:
{
  "label": "acceptable_insufficient_response | false_insufficient | overconfident_wrong_answer | poor_safe_failure_wording",
  "confidence_band": "high | medium | low",
  "short_reason": "<= 240 chars",
  "supporting_notes": ["optional short note"]
}
Do not include markdown, code fences, or extra keys.
"""


@dataclass(frozen=True, slots=True)
class SafeFailureJudgeResult:
    evaluator_name: str
    label: SafeFailureLabel
    passed: bool
    confidence_band: ConfidenceBand
    short_reason: str
    supporting_notes: list[str]
    family: str
    aggregation_fields: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_safe_failure_prompt(
    *,
    eval_case: Mapping[str, Any],
    system_output: Mapping[str, Any] | str,
    debug_payload: Mapping[str, Any] | None = None,
) -> str:
    case_payload = {
        "id": eval_case.get("id"),
        "family": eval_case.get("family"),
        "query": eval_case.get("query"),
        "expected_outcome": eval_case.get("expected_outcome"),
        "answerability_expected": eval_case.get("answerability_expected"),
        "safe_failure_expected": eval_case.get("safe_failure_expected"),
        "evidence_requirement": eval_case.get("evidence_requirement"),
        "gold_evidence_ids": list(eval_case.get("gold_evidence_ids") or []),
        "notes": eval_case.get("notes"),
    }

    if isinstance(system_output, Mapping):
        answer_payload: Mapping[str, Any] = {
            "answer_text": system_output.get("answer_text"),
            "grounded": system_output.get("grounded"),
            "sufficient_context": system_output.get("sufficient_context"),
            "citations": list(system_output.get("citations") or []),
            "warnings": list(system_output.get("warnings") or []),
        }
    else:
        answer_payload = {"answer_text": str(system_output)}

    debug_snapshot = _minimal_debug_snapshot(debug_payload)

    return (
        f"{RUBRIC_PROMPT}\n\n"
        "Evaluation case:\n"
        f"{json.dumps(case_payload, ensure_ascii=False, sort_keys=True, indent=2)}\n\n"
        "Model answer payload:\n"
        f"{json.dumps(answer_payload, ensure_ascii=False, sort_keys=True, indent=2)}\n\n"
        "Debug safe-failure snapshot:\n"
        f"{json.dumps(debug_snapshot, ensure_ascii=False, sort_keys=True, indent=2)}"
    )


def parse_safe_failure_result(raw_result: Mapping[str, Any] | str, *, evaluator_name: str = EVALUATOR_NAME) -> SafeFailureJudgeResult:
    payload = _coerce_mapping(raw_result)

    label = payload.get("label")
    confidence_band = payload.get("confidence_band")
    short_reason = payload.get("short_reason")
    supporting_notes = payload.get("supporting_notes", [])

    if label not in VALID_LABELS or label == "malformed_judge_output":
        raise ValueError(f"invalid label: {label!r}")
    if confidence_band not in VALID_CONFIDENCE_BANDS:
        raise ValueError(f"invalid confidence_band: {confidence_band!r}")
    if not isinstance(short_reason, str) or not short_reason.strip():
        raise ValueError("short_reason must be a non-empty string")
    short_reason = short_reason.strip()
    if len(short_reason) > 240:
        raise ValueError("short_reason must be <= 240 characters")
    if not isinstance(supporting_notes, list) or not all(isinstance(item, str) for item in supporting_notes):
        raise ValueError("supporting_notes must be a list[str]")

    normalized_notes = [item.strip() for item in supporting_notes if item.strip()]

    return SafeFailureJudgeResult(
        evaluator_name=evaluator_name,
        label=label,
        passed=label in PASSING_LABELS,
        confidence_band=confidence_band,
        short_reason=short_reason,
        supporting_notes=normalized_notes,
        family="unknown",
        aggregation_fields={},
        metadata={
            "deterministic_enough_for_batch": True,
            "model_based_judgment": True,
            "rubric_version": "safe_failure_v1",
            "raw_keys": sorted(payload.keys()),
        },
    )


def evaluate_safe_failure_with_llm(
    *,
    eval_case: Mapping[str, Any],
    system_output: Mapping[str, Any] | str,
    debug_payload: Mapping[str, Any] | None,
    judge_callable: Callable[[str], Mapping[str, Any] | str],
) -> SafeFailureJudgeResult:
    prompt = build_safe_failure_prompt(eval_case=eval_case, system_output=system_output, debug_payload=debug_payload)
    family = str(eval_case.get("family") or "unknown")
    case_id = str(eval_case.get("id") or "")

    try:
        raw = judge_callable(prompt)
        parsed = parse_safe_failure_result(raw)
        return SafeFailureJudgeResult(
            **{
                **parsed.to_dict(),
                "family": family,
                "aggregation_fields": {
                    "case_id": case_id,
                    "family": family,
                    "label": parsed.label,
                    "passed": parsed.passed,
                },
                "metadata": {
                    **parsed.metadata,
                    "malformed_output_fallback": False,
                },
            }
        )
    except Exception as exc:
        return SafeFailureJudgeResult(
            evaluator_name=EVALUATOR_NAME,
            label="malformed_judge_output",
            passed=False,
            confidence_band="low",
            short_reason="Judge output malformed; safe-failure grading defaulted to failure.",
            supporting_notes=[str(exc)],
            family=family,
            aggregation_fields={
                "case_id": case_id,
                "family": family,
                "label": "malformed_judge_output",
                "passed": False,
            },
            metadata={
                "deterministic_enough_for_batch": True,
                "model_based_judgment": True,
                "rubric_version": "safe_failure_v1",
                "malformed_output_fallback": True,
            },
        )


def _minimal_debug_snapshot(debug_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(debug_payload, Mapping):
        return {}
    answerability = debug_payload.get("answerability_result")
    return {
        "answerability_result": answerability if isinstance(answerability, Mapping) else None,
        "warnings": list(debug_payload.get("warnings") or []),
        "decomposition": debug_payload.get("decomposition"),
    }


def _coerce_mapping(raw_result: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(raw_result, Mapping):
        return raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError as exc:
            raise ValueError("raw_result is not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("parsed JSON must be an object")
        return parsed
    raise ValueError("raw_result must be a mapping or JSON string")


__all__ = [
    "EVALUATOR_NAME",
    "RUBRIC_PROMPT",
    "SafeFailureJudgeResult",
    "build_safe_failure_prompt",
    "parse_safe_failure_result",
    "evaluate_safe_failure_with_llm",
]
