from __future__ import annotations

from evals.graders.llm_judges.groundedness import (
    RUBRIC_PROMPT as GROUNDEDNESS_RUBRIC,
    build_groundedness_prompt,
    evaluate_groundedness_with_llm,
)
from evals.graders.llm_judges.safe_failure import (
    RUBRIC_PROMPT as SAFE_FAILURE_RUBRIC,
    build_safe_failure_prompt,
    evaluate_safe_failure_with_llm,
)


def _base_case() -> dict[str, object]:
    return {
        "id": "case-ticket-14",
        "family": "party_role_verification",
        "query": "Who is the employer in the provided agreement?",
        "expected_outcome": "answered",
        "expected_answer_type": "fact_extraction",
        "answerability_expected": "answerable",
        "safe_failure_expected": False,
        "evidence_requirement": "required",
        "gold_evidence_ids": ["eu-1"],
        "gold_citation_refs": [{"document_id": "doc-1", "locator": "Section 2"}],
        "notes": "Employer is Acme Corp.",
    }


def _grounded_result() -> dict[str, object]:
    return {
        "answer_text": "The employer is Acme Corp.",
        "grounded": True,
        "sufficient_context": True,
        "citations": [{"document_id": "doc-1", "parent_chunk_id": "eu-1"}],
        "warnings": [],
    }


def _insufficient_result() -> dict[str, object]:
    return {
        "answer_text": "I do not have enough evidence in the provided documents to answer reliably.",
        "grounded": False,
        "sufficient_context": False,
        "citations": [],
        "warnings": ["insufficient_context:fact_not_found"],
    }


def test_groundedness_rubric_includes_required_guardrails() -> None:
    prompt = GROUNDEDNESS_RUBRIC.lower()
    assert "do not use external legal knowledge" in prompt
    assert "do not reward plausible legal wording" in prompt
    assert "do not infer support from style" in prompt


def test_safe_failure_rubric_includes_required_guardrails() -> None:
    prompt = SAFE_FAILURE_RUBRIC.lower()
    assert "do not use external legal knowledge" in prompt
    assert "separate \"insufficient but acceptable\" from \"unsupported but confident\"" in prompt


def test_groundedness_can_classify_grounded_answer() -> None:
    result = evaluate_groundedness_with_llm(
        eval_case=_base_case(),
        system_output=_grounded_result(),
        debug_payload={"answerability_result": {"sufficient_context": True, "should_answer": True}},
        judge_callable=lambda _: {
            "label": "grounded_answer",
            "confidence_band": "high",
            "short_reason": "Answer claims are directly supported by cited evidence.",
            "supporting_notes": ["Citation aligns with gold evidence."],
        },
    )

    assert result.label == "grounded_answer"
    assert result.passed is True
    assert result.aggregation_fields["family"] == "party_role_verification"


def test_groundedness_can_classify_unsupported_inference() -> None:
    unsupported = _grounded_result()
    unsupported["answer_text"] = "The employer is Acme Corp and termination is always without cause."

    result = evaluate_groundedness_with_llm(
        eval_case=_base_case(),
        system_output=unsupported,
        debug_payload={"answerability_result": {"sufficient_context": True, "should_answer": True}},
        judge_callable=lambda _: {
            "label": "unsupported_inference",
            "confidence_band": "medium",
            "short_reason": "The added termination claim is not supported by provided evidence.",
            "supporting_notes": [],
        },
    )

    assert result.label == "unsupported_inference"
    assert result.passed is False


def test_safe_failure_can_classify_acceptable_insufficient_response() -> None:
    case = _base_case()
    case["answerability_expected"] = "unanswerable"
    case["expected_outcome"] = "safe_failure_insufficient_evidence"
    case["safe_failure_expected"] = True

    result = evaluate_safe_failure_with_llm(
        eval_case=case,
        system_output=_insufficient_result(),
        debug_payload={"answerability_result": {"sufficient_context": False, "should_answer": False}},
        judge_callable=lambda _: {
            "label": "acceptable_insufficient_response",
            "confidence_band": "high",
            "short_reason": "System correctly declined due to insufficient evidence and communicated limits clearly.",
            "supporting_notes": ["No unsupported claim made."],
        },
    )

    assert result.label == "acceptable_insufficient_response"
    assert result.passed is True
    assert result.family == "party_role_verification"


def test_safe_failure_can_classify_overconfident_wrong_answer() -> None:
    case = _base_case()
    case["answerability_expected"] = "unanswerable"
    case["expected_outcome"] = "safe_failure_ambiguous"
    case["safe_failure_expected"] = True

    overconfident = _grounded_result()
    overconfident["answer_text"] = "The employer is definitely Beta LLC."

    result = evaluate_safe_failure_with_llm(
        eval_case=case,
        system_output=overconfident,
        debug_payload={"answerability_result": {"sufficient_context": False, "should_answer": False}},
        judge_callable=lambda _: {
            "label": "overconfident_wrong_answer",
            "confidence_band": "high",
            "short_reason": "Case expected safe failure, but answer gave a definitive unsupported claim.",
            "supporting_notes": [],
        },
    )

    assert result.label == "overconfident_wrong_answer"
    assert result.passed is False


def test_machine_readable_output_shape_is_stable() -> None:
    result = evaluate_groundedness_with_llm(
        eval_case=_base_case(),
        system_output=_grounded_result(),
        debug_payload={},
        judge_callable=lambda _: {
            "label": "partially_grounded_answer",
            "confidence_band": "low",
            "short_reason": "Core claim supported, but one extension lacks citation support.",
            "supporting_notes": [],
        },
    )

    payload = result.to_dict()
    assert payload["evaluator_name"] == "groundedness_llm_judge_v1"
    assert isinstance(payload["aggregation_fields"], dict)
    assert isinstance(payload["metadata"], dict)


def test_malformed_judge_output_is_handled_safely() -> None:
    result = evaluate_safe_failure_with_llm(
        eval_case=_base_case(),
        system_output=_insufficient_result(),
        debug_payload={},
        judge_callable=lambda _: "not-json",
    )

    assert result.label == "malformed_judge_output"
    assert result.passed is False
    assert result.metadata["malformed_output_fallback"] is True


def test_prompts_include_case_answer_and_debug_sections() -> None:
    grounded_prompt = build_groundedness_prompt(
        eval_case=_base_case(),
        system_output=_grounded_result(),
        debug_payload={"answerability_result": {"sufficient_context": True}},
    )
    safe_failure_prompt = build_safe_failure_prompt(
        eval_case=_base_case(),
        system_output=_insufficient_result(),
        debug_payload={"answerability_result": {"sufficient_context": False}},
    )

    assert "Evaluation case:" in grounded_prompt
    assert "Model answer payload:" in grounded_prompt
    assert "Debug evidence snapshot:" in grounded_prompt

    assert "Evaluation case:" in safe_failure_prompt
    assert "Model answer payload:" in safe_failure_prompt
    assert "Debug safe-failure snapshot:" in safe_failure_prompt
