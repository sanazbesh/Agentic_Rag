"""LLM-judge for evidence-groundedness in legal RAG outputs.

This module stays intentionally narrow:
- rubric-based prompt construction
- strict structured-output parsing
- safe fallback for malformed judge output

It does not perform LLM transport or build a general judge framework.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

GroundednessLabel = Literal[
    "grounded_answer",
    "partially_grounded_answer",
    "unsupported_inference",
    "overconfident_wrong_answer",
    "malformed_judge_output",
]
ConfidenceBand = Literal["high", "medium", "low"]

EVALUATOR_NAME = "groundedness_llm_judge_v1"
VALID_LABELS: tuple[GroundednessLabel, ...] = (
    "grounded_answer",
    "partially_grounded_answer",
    "unsupported_inference",
    "overconfident_wrong_answer",
    "malformed_judge_output",
)
VALID_CONFIDENCE_BANDS: tuple[ConfidenceBand, ...] = ("high", "medium", "low")
PASSING_LABELS = {"grounded_answer", "partially_grounded_answer"}

RUBRIC_PROMPT = """You are a strict groundedness judge for legal RAG evaluation.

Task:
Classify the system answer into exactly one label:
1) grounded_answer
2) partially_grounded_answer
3) unsupported_inference
4) overconfident_wrong_answer

Hard constraints:
- Use only the supplied evaluation case, model answer payload, citations, and debug evidence metadata.
- Do NOT use external legal knowledge.
- Do NOT reward plausible legal wording without explicit support.
- Do NOT infer support from style, confidence, or tone.
- If support is missing, judge it as unsupported even when wording sounds plausible.

Label definitions (mutually exclusive):
- grounded_answer: Material claims are supported by provided evidence/citations and do not overreach.
- partially_grounded_answer: Core answer is supported, but at least one material extension is weak/missing support.
- unsupported_inference: Answer makes unsupported claims or extrapolations beyond provided evidence.
- overconfident_wrong_answer: Answer asserts a definitive claim that conflicts with supplied evidence/metadata or should have been uncertain but is presented confidently.

Important:
- Judge only groundedness against supplied materials (not general writing quality).
- Keep rationale concise and case-specific.

Output format requirements:
Return JSON only with exactly this schema:
{
  "label": "grounded_answer | partially_grounded_answer | unsupported_inference | overconfident_wrong_answer",
  "confidence_band": "high | medium | low",
  "short_reason": "<= 240 chars",
  "supporting_notes": ["optional short note"]
}
Do not include markdown, code fences, or extra keys.
"""


@dataclass(frozen=True, slots=True)
class GroundednessJudgeResult:
    evaluator_name: str
    label: GroundednessLabel
    passed: bool
    confidence_band: ConfidenceBand
    short_reason: str
    supporting_notes: list[str]
    family: str
    aggregation_fields: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_groundedness_prompt(
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
        "gold_citation_refs": list(eval_case.get("gold_citation_refs") or []),
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
        "Debug evidence snapshot:\n"
        f"{json.dumps(debug_snapshot, ensure_ascii=False, sort_keys=True, indent=2)}"
    )


def parse_groundedness_result(raw_result: Mapping[str, Any] | str, *, evaluator_name: str = EVALUATOR_NAME) -> GroundednessJudgeResult:
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

    return GroundednessJudgeResult(
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
            "rubric_version": "groundedness_v1",
            "raw_keys": sorted(payload.keys()),
        },
    )


def evaluate_groundedness_with_llm(
    *,
    eval_case: Mapping[str, Any],
    system_output: Mapping[str, Any] | str,
    debug_payload: Mapping[str, Any] | None,
    judge_callable: Callable[[str], Mapping[str, Any] | str],
) -> GroundednessJudgeResult:
    prompt = build_groundedness_prompt(eval_case=eval_case, system_output=system_output, debug_payload=debug_payload)
    family = str(eval_case.get("family") or "unknown")
    case_id = str(eval_case.get("id") or "")

    try:
        raw = judge_callable(prompt)
        parsed = parse_groundedness_result(raw)
        return GroundednessJudgeResult(
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
        return GroundednessJudgeResult(
            evaluator_name=EVALUATOR_NAME,
            label="malformed_judge_output",
            passed=False,
            confidence_band="low",
            short_reason="Judge output malformed; classified conservatively as failure.",
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
                "rubric_version": "groundedness_v1",
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
        "context_resolution": debug_payload.get("context_resolution"),
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
    "GroundednessJudgeResult",
    "build_groundedness_prompt",
    "parse_groundedness_result",
    "evaluate_groundedness_with_llm",
]
