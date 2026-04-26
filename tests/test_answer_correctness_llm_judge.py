from __future__ import annotations

import pytest

from evals.graders.llm_judges.answer_correctness import (
    RUBRIC_PROMPT,
    build_answer_correctness_prompt,
    build_batch_answer_correctness_prompts,
    parse_answer_correctness_result,
)


def _base_case() -> dict[str, object]:
    return {
        "id": "case-legal-13",
        "family": "party_role_verification",
        "query": "Who is identified as the employer in the provided documents?",
        "expected_outcome": "answered",
        "expected_answer_type": "fact_extraction",
        "answerability_expected": "answerable",
        "safe_failure_expected": False,
        "evidence_requirement": "required",
        "gold_evidence_ids": ["eu-1"],
        "gold_citation_refs": [{"document_id": "doc-1", "locator": "Section 2"}],
        "notes": "Employer is Acme Corp.",
    }


def _final_result() -> dict[str, object]:
    return {
        "answer_text": "The employer is Acme Corp.",
        "grounded": True,
        "sufficient_context": True,
        "citations": [{"document_id": "doc-1", "parent_chunk_id": "eu-1"}],
        "warnings": [],
    }


def test_rubric_prompt_includes_all_required_labels() -> None:
    lowered = RUBRIC_PROMPT.lower()
    assert "correct" in lowered
    assert "partially correct" in lowered
    assert "incorrect" in lowered
    assert "unsupported" in lowered


def test_rubric_prompt_distinguishes_incorrect_vs_unsupported() -> None:
    lowered = RUBRIC_PROMPT.lower()
    assert "incorrect = wrong answer for the case" in lowered
    assert "unsupported = not supportable from provided case/evidence" in lowered


def test_rubric_prompt_contains_legal_safe_instructions() -> None:
    prompt = RUBRIC_PROMPT.lower()
    assert "grade only against the provided evaluation case" in prompt
    assert "do not make new legal conclusions" in prompt
    assert "do not assume missing facts" in prompt
    assert "do not reward fluent but unsupported legal-sounding text" in prompt
    assert "remain conservative when evidence is incomplete or ambiguous" in prompt


def test_prompt_builder_includes_case_and_answer_payload() -> None:
    prompt = build_answer_correctness_prompt(eval_case=_base_case(), system_output=_final_result())

    assert "Evaluation case:" in prompt
    assert "Model answer payload:" in prompt
    assert '"query": "Who is identified as the employer in the provided documents?"' in prompt
    assert '"answer_text": "The employer is Acme Corp."' in prompt


def test_parser_accepts_valid_structured_output() -> None:
    result = parse_answer_correctness_result(
        {
            "label": "correct",
            "confidence_band": "high",
            "short_reason": "Matches expected employer and is supported by provided evidence.",
            "supporting_notes": ["Aligned with gold evidence eu-1."],
        }
    )

    as_dict = result.to_dict()
    assert as_dict["label"] == "correct"
    assert as_dict["passed"] is True
    assert as_dict["confidence_band"] == "high"
    assert isinstance(as_dict["supporting_notes"], list)


def test_parser_rejects_malformed_or_incomplete_output() -> None:
    with pytest.raises(ValueError):
        parse_answer_correctness_result({"label": "correct", "short_reason": "ok"})

    with pytest.raises(ValueError):
        parse_answer_correctness_result(
            {
                "label": "not-a-label",
                "confidence_band": "high",
                "short_reason": "bad",
                "supporting_notes": [],
            }
        )


def test_parser_supports_json_string_payload() -> None:
    payload = '{"label":"unsupported","confidence_band":"medium","short_reason":"Speculates beyond evidence.","supporting_notes":[]}'
    result = parse_answer_correctness_result(payload)

    assert result.label == "unsupported"
    assert result.passed is False


def test_batch_prompt_builder_supports_batch_style_flow() -> None:
    rows = [(_base_case(), _final_result()), ({**_base_case(), "id": "case-legal-13b"}, "Insufficient evidence.")]

    prompts = build_batch_answer_correctness_prompts(rows)

    assert len(prompts) == 2
    assert prompts[0]["case_id"] == "case-legal-13"
    assert prompts[1]["case_id"] == "case-legal-13b"
    assert "strict legal rag evaluation judge" in prompts[0]["prompt"].lower()


def test_entrypoints_are_directly_callable_without_llm_stack() -> None:
    prompt = build_answer_correctness_prompt(eval_case=_base_case(), system_output=_final_result())
    parsed = parse_answer_correctness_result(
        {
            "label": "partially correct",
            "confidence_band": "low",
            "short_reason": "Contains correct core answer but misses qualification.",
            "supporting_notes": [],
        }
    )

    assert isinstance(prompt, str)
    assert parsed.metadata["deterministic_enough_for_batch"] is True
