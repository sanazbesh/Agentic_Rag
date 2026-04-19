"""LLM-judge prompt layer for semantic answer-correctness grading.

This module is intentionally narrow for offline/batch evaluation:
- prompt construction with a constrained legal-safe rubric
- machine-readable output contract
- strict parser for judge outputs

It does not implement LLM transport, orchestration, tracing, or dashboards.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

AnswerCorrectnessLabel = Literal["correct", "partially correct", "incorrect", "unsupported"]
ConfidenceBand = Literal["high", "medium", "low"]

EVALUATOR_NAME = "answer_correctness_llm_judge_v1"
VALID_LABELS: tuple[AnswerCorrectnessLabel, ...] = (
    "correct",
    "partially correct",
    "incorrect",
    "unsupported",
)
VALID_CONFIDENCE_BANDS: tuple[ConfidenceBand, ...] = ("high", "medium", "low")
PASSING_LABELS = {"correct", "partially correct"}

RUBRIC_PROMPT = """You are a strict legal RAG evaluation judge.

Task:
Classify the model answer into exactly one label:
1) correct
2) partially correct
3) incorrect
4) unsupported

Legal-safe grading rules (must follow):
- Grade only against the provided evaluation case inputs, expected outcome, and supplied evidence context.
- Do not make new legal conclusions.
- Do not assume missing facts.
- Do not reward fluent but unsupported legal-sounding text.
- Treat speculative or unsupported legal claims as unsupported or incorrect.
- Judge whether the answer addresses the asked question, not writing quality.
- Remain conservative when evidence is incomplete or ambiguous.

Rubric (mutually exclusive labels):
- correct: The answer correctly responds to the query, is supported by provided evidence/expected support structure, and has no material contradictions or unsupported additions.
- partially correct: The answer contains some correct relevant information but is incomplete, too narrow, or has a minor unsupported extension; it does not fully satisfy the expected result.
- incorrect: The answer is materially wrong for this case (e.g., misstates facts, answers a different question, contradicts expected outcome, or makes the wrong legal/factual claim based on provided inputs).
- unsupported: The answer presents claims as supported even though support is not available in provided case/evidence; it relies on speculation, missing facts, or inference beyond supplied evaluation context.

Important distinction:
- incorrect = wrong answer for the case.
- unsupported = not supportable from provided case/evidence.

Output format requirements:
- Return JSON only.
- Use exactly this schema:
{
  "label": "correct | partially correct | incorrect | unsupported",
  "confidence_band": "high | medium | low",
  "short_reason": "<= 240 characters, concise and case-grounded",
  "supporting_notes": ["optional short bullet", "optional short bullet"]
}
- Do not include markdown, code fences, or extra keys.
"""


@dataclass(frozen=True, slots=True)
class AnswerCorrectnessJudgeResult:
    """Parsed machine-readable LLM judge output for answer correctness."""

    evaluator_name: str
    label: AnswerCorrectnessLabel
    passed: bool
    confidence_band: ConfidenceBand
    short_reason: str
    supporting_notes: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_answer_correctness_prompt(
    *,
    eval_case: Mapping[str, Any],
    system_output: Mapping[str, Any] | str,
) -> str:
    """Build deterministic-enough grading prompt for one eval case + answer output."""

    case_payload = {
        "id": eval_case.get("id"),
        "family": eval_case.get("family"),
        "query": eval_case.get("query"),
        "expected_outcome": eval_case.get("expected_outcome"),
        "expected_answer_type": eval_case.get("expected_answer_type"),
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

    return (
        f"{RUBRIC_PROMPT}\n\n"
        "Evaluation case:\n"
        f"{json.dumps(case_payload, ensure_ascii=False, sort_keys=True, indent=2)}\n\n"
        "Model answer payload:\n"
        f"{json.dumps(answer_payload, ensure_ascii=False, sort_keys=True, indent=2)}"
    )


def build_batch_answer_correctness_prompts(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any] | str]],
) -> list[dict[str, Any]]:
    """Create prompts for batch flows without introducing LLM-calling orchestration."""

    prompts: list[dict[str, Any]] = []
    for eval_case, system_output in rows:
        prompts.append(
            {
                "case_id": str(eval_case.get("id") or ""),
                "evaluator_name": EVALUATOR_NAME,
                "prompt": build_answer_correctness_prompt(eval_case=eval_case, system_output=system_output),
            }
        )
    return prompts


def parse_answer_correctness_result(
    raw_result: Mapping[str, Any] | str,
    *,
    evaluator_name: str = EVALUATOR_NAME,
) -> AnswerCorrectnessJudgeResult:
    """Parse and validate structured LLM-judge output.

    Raises:
        ValueError: when payload is malformed or incomplete.
    """

    payload = _coerce_mapping(raw_result)

    label = payload.get("label")
    confidence_band = payload.get("confidence_band")
    short_reason = payload.get("short_reason")
    supporting_notes = payload.get("supporting_notes", [])

    if label not in VALID_LABELS:
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

    return AnswerCorrectnessJudgeResult(
        evaluator_name=evaluator_name,
        label=label,
        passed=label in PASSING_LABELS,
        confidence_band=confidence_band,
        short_reason=short_reason,
        supporting_notes=normalized_notes,
        metadata={
            "deterministic_enough_for_batch": True,
            "model_based_judgment": True,
            "rubric_version": "answer_correctness_v1",
            "raw_keys": sorted(payload.keys()),
        },
    )


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
