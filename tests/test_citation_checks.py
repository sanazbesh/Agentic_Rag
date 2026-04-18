from __future__ import annotations

from evals.graders.citation_checks import (
    aggregate_citation_results_by_family,
    evaluate_citation_checks,
)


def _base_case() -> dict[str, object]:
    return {
        "id": "case-citation-1",
        "family": "party_role_verification",
        "answerability_expected": "answerable",
        "expected_outcome": "answered",
        "evidence_requirement": "required",
        "selected_document_ids": ["doc-1"],
        "gold_evidence_ids": ["gold-eu-1"],
        "gold_citation_refs": [{"document_id": "doc-1", "locator": "Introduction"}],
    }


def _supported_result() -> dict[str, object]:
    return {
        "answer_text": "Employer is Acme Corp.",
        "grounded": True,
        "sufficient_context": True,
        "citations": [
            {
                "parent_chunk_id": "gold-eu-1",
                "document_id": "doc-1",
                "supporting_excerpt": "Acme Corp. (the Employer)",
            }
        ],
        "warnings": [],
    }


def _metric(result: object, name: str) -> object:
    return next(item for item in result.metrics if item.metric_name == name)


def test_grounded_answer_with_missing_citations_is_flagged() -> None:
    case = _base_case()
    final_result = _supported_result()
    final_result["citations"] = []

    result = evaluate_citation_checks(eval_case=case, final_result=final_result, debug_payload={})

    presence = _metric(result, "citation_presence")
    assert presence.passed is False
    assert presence.details["failure_code"] == "grounded_without_citations"


def test_supported_answer_with_matching_gold_evidence_passes_support_match() -> None:
    result = evaluate_citation_checks(eval_case=_base_case(), final_result=_supported_result(), debug_payload={})

    support = _metric(result, "citation_support_match")
    assert support.passed is True
    assert support.value == 1.0
    assert support.details["matched_gold_evidence_ids"] == ["gold-eu-1"]


def test_supported_answer_with_wrong_evidence_ids_fails_support_match() -> None:
    case = _base_case()
    final_result = _supported_result()
    final_result["citations"] = [{"parent_chunk_id": "wrong-eu", "document_id": "doc-1"}]

    result = evaluate_citation_checks(eval_case=case, final_result=final_result, debug_payload={})

    support = _metric(result, "citation_support_match")
    assert support.passed is False
    assert support.value == 0.0


def test_citation_outside_selected_scope_is_flagged_irrelevant() -> None:
    case = _base_case()
    final_result = _supported_result()
    final_result["citations"] = [{"parent_chunk_id": "gold-eu-1", "document_id": "doc-x"}]

    result = evaluate_citation_checks(eval_case=case, final_result=final_result, debug_payload={})

    relevance = _metric(result, "citation_relevance")
    assert relevance.passed is False
    assert relevance.details["irrelevant_indices"] == [0]


def test_extraneous_citation_counts_toward_unused_rate_with_strict_gold() -> None:
    case = _base_case()
    final_result = _supported_result()
    final_result["citations"] = [
        {"parent_chunk_id": "gold-eu-1", "document_id": "doc-1"},
        {"parent_chunk_id": "extra-eu", "document_id": "doc-1"},
    ]

    result = evaluate_citation_checks(eval_case=case, final_result=final_result, debug_payload={})

    unused = _metric(result, "unused_citation_rate")
    assert unused.value == 0.5
    assert unused.details["unused_count"] == 1


def test_unsupported_answer_with_no_citations_is_handled_cleanly() -> None:
    case = _base_case()
    case["answerability_expected"] = "unanswerable"
    case["expected_outcome"] = "safe_failure_insufficient_evidence"
    case["evidence_requirement"] = "none"
    case["gold_evidence_ids"] = []
    final_result = {
        "answer_text": "Insufficient evidence.",
        "grounded": False,
        "sufficient_context": False,
        "citations": [],
        "warnings": [],
    }

    result = evaluate_citation_checks(eval_case=case, final_result=final_result, debug_payload={})

    presence = _metric(result, "citation_presence")
    support = _metric(result, "citation_support_match")
    assert presence.passed is True
    assert support.passed is None


def test_per_case_output_shape_is_stable_and_machine_readable() -> None:
    case = _base_case()
    payload = _supported_result()

    first = evaluate_citation_checks(eval_case=case, final_result=payload, debug_payload={})
    second = evaluate_citation_checks(eval_case=case, final_result=payload, debug_payload={})

    assert first.to_dict() == second.to_dict()
    as_dict = first.to_dict()
    assert as_dict["evaluator_name"] == "citation_checks_v1"
    assert as_dict["case_family"] == "party_role_verification"
    assert isinstance(as_dict["metrics"], list)
    assert isinstance(as_dict["aggregation_fields"], dict)


def test_aggregation_by_family_computes_required_fields() -> None:
    case_a = _base_case()
    result_a = evaluate_citation_checks(eval_case=case_a, final_result=_supported_result(), debug_payload={})

    case_b = _base_case()
    case_b["id"] = "case-citation-2"
    final_b = _supported_result()
    final_b["citations"] = [{"parent_chunk_id": "wrong", "document_id": "doc-x"}]
    result_b = evaluate_citation_checks(eval_case=case_b, final_result=final_b, debug_payload={})

    grouped = aggregate_citation_results_by_family([result_a, result_b])
    family = grouped["party_role_verification"]
    assert family["case_count"] == 2
    assert family["citation_presence_pass_rate"] == 1.0
    assert family["citation_support_match_pass_rate"] == 0.5
    assert family["unused_citation_rate_avg"] == 0.5


def test_presence_failure_is_distinct_from_support_match_failure() -> None:
    case = _base_case()

    missing = _supported_result()
    missing["citations"] = []
    missing_result = evaluate_citation_checks(eval_case=case, final_result=missing, debug_payload={})
    assert _metric(missing_result, "citation_presence").passed is False
    assert _metric(missing_result, "citation_support_match").passed is None

    wrong = _supported_result()
    wrong["citations"] = [{"parent_chunk_id": "wrong-eu", "document_id": "doc-1"}]
    wrong_result = evaluate_citation_checks(eval_case=case, final_result=wrong, debug_payload={})
    assert _metric(wrong_result, "citation_presence").passed is True
    assert _metric(wrong_result, "citation_support_match").passed is False


def test_citation_evaluator_entrypoint_callable_for_offline_runner_integration() -> None:
    result = evaluate_citation_checks(eval_case=_base_case(), final_result=_supported_result(), debug_payload={})

    assert result.metadata["deterministic"] is True
    assert result.metadata["model_based_judgment"] is False
