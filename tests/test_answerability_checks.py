from __future__ import annotations

from evals.graders.answerability_checks import (
    aggregate_answerability_results_by_family,
    evaluate_answerability_checks,
)


def _base_case() -> dict[str, object]:
    return {
        "id": "case-1",
        "family": "party_role_verification",
        "answerability_expected": "answerable",
        "expected_outcome": "answered",
        "safe_failure_expected": False,
        "expected_answer_type": "fact_extraction",
    }


def _supported_result() -> dict[str, object]:
    return {
        "answer_text": "The employer is Acme Corp.",
        "grounded": True,
        "sufficient_context": True,
        "citations": [{"document_id": "doc-1"}],
        "warnings": [],
    }


def _insufficient_result() -> dict[str, object]:
    return {
        "answer_text": "Insufficient evidence.",
        "grounded": False,
        "sufficient_context": False,
        "citations": [],
        "warnings": ["insufficient_context:no_retrieved_context"],
    }


def test_correct_supported_answer_classification() -> None:
    result = evaluate_answerability_checks(
        eval_case=_base_case(),
        final_result=_supported_result(),
        debug_payload={"answerability_result": {"sufficient_context": True, "should_answer": True}},
    )

    assert result.classification == "correct_supported_answer"
    assert result.passed is True


def test_correct_insufficient_answer_classification() -> None:
    case = _base_case()
    case["answerability_expected"] = "unanswerable"
    case["expected_outcome"] = "safe_failure_insufficient_evidence"
    case["safe_failure_expected"] = True

    result = evaluate_answerability_checks(
        eval_case=case,
        final_result=_insufficient_result(),
        debug_payload={"answerability_result": {"sufficient_context": False, "should_answer": False, "insufficiency_reason": "fact_not_found"}},
    )

    assert result.classification == "correct_insufficient_answer"
    assert result.passed is True


def test_false_positive_classification_when_safe_failure_expected_but_system_supports() -> None:
    case = _base_case()
    case["answerability_expected"] = "unanswerable"
    case["expected_outcome"] = "safe_failure_ambiguous"
    case["safe_failure_expected"] = True

    result = evaluate_answerability_checks(eval_case=case, final_result=_supported_result(), debug_payload={})

    assert result.classification == "false_positive"
    metric = next(item for item in result.metrics if item.metric_name == "false_confident_answer")
    assert metric.value == 1.0
    assert metric.passed is False


def test_false_negative_classification_when_answerable_case_marked_insufficient() -> None:
    result = evaluate_answerability_checks(
        eval_case=_base_case(),
        final_result=_insufficient_result(),
        debug_payload={"answerability_result": {"sufficient_context": False, "should_answer": False, "insufficiency_reason": "fact_not_found"}},
    )

    assert result.classification == "false_negative"
    metric = next(item for item in result.metrics if item.metric_name == "false_insufficient_answer")
    assert metric.value == 1.0
    assert metric.passed is False


def test_safe_failure_deterministic_checks_flag_invalid_supported_without_grounding_combo() -> None:
    case = _base_case()
    case["answerability_expected"] = "unanswerable"
    case["expected_outcome"] = "safe_failure_out_of_scope"
    case["safe_failure_expected"] = True

    invalid = _supported_result()
    invalid["grounded"] = False

    result = evaluate_answerability_checks(eval_case=case, final_result=invalid, debug_payload={})

    safe_failure = next(item for item in result.metrics if item.metric_name == "safe_failure_quality")
    assert safe_failure.passed is False
    assert "supported_without_grounded" in safe_failure.details["contradictions"]


def test_family_specific_sufficiency_correctness_metric_reflects_decision_quality() -> None:
    case = _base_case()
    case["family"] = "chronology_date_event"

    result = evaluate_answerability_checks(eval_case=case, final_result=_insufficient_result(), debug_payload={})

    metric = next(item for item in result.metrics if item.metric_name == "family_sufficiency_correct")
    assert metric.value == 0.0
    assert metric.passed is False
    assert metric.details["family"] == "chronology_date_event"


def test_per_case_output_shape_is_stable_and_machine_readable() -> None:
    case = _base_case()
    payload = _supported_result()

    first = evaluate_answerability_checks(eval_case=case, final_result=payload, debug_payload={})
    second = evaluate_answerability_checks(eval_case=case, final_result=payload, debug_payload={})

    assert first.to_dict() == second.to_dict()
    as_dict = first.to_dict()
    assert as_dict["evaluator_name"] == "answerability_checks_v1"
    assert as_dict["classification"] == "correct_supported_answer"
    assert isinstance(as_dict["metrics"], list)
    assert isinstance(as_dict["aggregation_fields"], dict)


def test_aggregation_by_family_computes_required_counts_and_rates() -> None:
    case_a = _base_case()
    result_a = evaluate_answerability_checks(eval_case=case_a, final_result=_supported_result(), debug_payload={})

    case_b = _base_case()
    case_b["id"] = "case-2"
    case_b["answerability_expected"] = "unanswerable"
    case_b["expected_outcome"] = "safe_failure_insufficient_evidence"
    case_b["safe_failure_expected"] = True
    result_b = evaluate_answerability_checks(eval_case=case_b, final_result=_supported_result(), debug_payload={})

    grouped = aggregate_answerability_results_by_family([result_a, result_b])
    family = grouped["party_role_verification"]
    assert family["case_count"] == 2
    assert family["false_positive_count"] == 1
    assert family["correct_supported_count"] == 1
    assert family["safe_failure_quality_pass_count"] == 1


def test_evaluator_handles_expected_answerable_and_unanswerable_consistently() -> None:
    answerable = _base_case()
    answerable_result = evaluate_answerability_checks(eval_case=answerable, final_result=_supported_result(), debug_payload={})

    unanswerable = _base_case()
    unanswerable["id"] = "case-3"
    unanswerable["answerability_expected"] = "unanswerable"
    unanswerable["expected_outcome"] = "safe_failure_out_of_scope"
    unanswerable["safe_failure_expected"] = True
    unanswerable_result = evaluate_answerability_checks(eval_case=unanswerable, final_result=_insufficient_result(), debug_payload={})

    assert answerable_result.classification == "correct_supported_answer"
    assert unanswerable_result.classification == "correct_insufficient_answer"


def test_answerability_evaluator_entrypoint_callable_for_offline_runner_integration() -> None:
    result = evaluate_answerability_checks(eval_case=_base_case(), final_result=_supported_result(), debug_payload={})

    assert result.metadata["deterministic"] is True
    assert result.metadata["model_based_judgment"] is False
