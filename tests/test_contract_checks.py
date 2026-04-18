from __future__ import annotations

from evals.graders.contract_checks import evaluate_contract_checks


def _valid_final_result() -> dict[str, object]:
    return {
        "answer_text": "Supported answer.",
        "grounded": True,
        "sufficient_context": True,
        "citations": [
            {
                "parent_chunk_id": "p1",
                "document_id": "doc-1",
                "source_name": "Doc 1",
                "heading": "Section 1",
                "supporting_excerpt": "Excerpt",
            }
        ],
        "warnings": [],
    }


def test_valid_final_answer_contract_passes() -> None:
    result = evaluate_contract_checks(final_result=_valid_final_result())

    assert result.passed is True
    assert result.checks[0].check_name == "valid_final_answer_contract"
    assert result.checks[0].passed is True


def test_malformed_final_answer_contract_fails_clearly() -> None:
    result = evaluate_contract_checks(final_result={"answer_text": "missing fields"})

    assert result.passed is False
    contract = result.checks[0]
    assert contract.check_name == "valid_final_answer_contract"
    assert contract.passed is False
    assert contract.failure_code == "final_answer_contract_invalid"


def test_grounded_true_with_empty_citations_fails() -> None:
    payload = _valid_final_result()
    payload["citations"] = []
    result = evaluate_contract_checks(final_result=payload)

    check = next(item for item in result.checks if item.check_name == "citations_present_when_grounded")
    assert check.passed is False
    assert check.failure_code == "grounded_without_citations"


def test_unsupported_with_citations_fails() -> None:
    payload = _valid_final_result()
    payload["grounded"] = False
    result = evaluate_contract_checks(final_result=payload)

    check = next(item for item in result.checks if item.check_name == "citations_empty_when_unsupported")
    assert check.passed is False
    assert check.failure_code == "ungrounded_with_citations"


def test_duplicate_warnings_are_flagged() -> None:
    payload = _valid_final_result()
    payload["warnings"] = ["same", "same"]

    result = evaluate_contract_checks(final_result=payload)

    check = next(item for item in result.checks if item.check_name == "no_duplicate_warnings")
    assert check.passed is False
    assert check.failure_code == "duplicate_warnings_detected"
    assert check.details is not None
    assert check.details["duplicate_final_warnings"] == ["same"]


def test_malformed_debug_payload_is_flagged() -> None:
    result = evaluate_contract_checks(final_result=_valid_final_result(), debug_payload={"warnings": "bad"})

    check = next(item for item in result.checks if item.check_name == "no_malformed_debug_payload")
    assert check.passed is False
    assert check.failure_code == "debug_payload_malformed"


def test_selected_document_scope_violation_is_flagged() -> None:
    debug_payload = {
        "meta": {"selected_document_ids": ["doc-allowed"]},
        "resolved_document_scope": ["doc-allowed"],
    }
    result = evaluate_contract_checks(final_result=_valid_final_result(), debug_payload=debug_payload)

    check = next(item for item in result.checks if item.check_name == "selected_document_scope_respected")
    assert check.passed is False
    assert check.failure_code == "citation_outside_selected_document_scope"


def test_selected_document_resolved_scope_violation_is_flagged() -> None:
    debug_payload = {
        "meta": {"selected_document_ids": ["doc-1"]},
        "resolved_document_scope": ["doc-1", "doc-2"],
    }
    payload = _valid_final_result()
    payload["citations"] = []
    payload["grounded"] = False

    result = evaluate_contract_checks(final_result=payload, debug_payload=debug_payload)

    check = next(item for item in result.checks if item.check_name == "selected_document_scope_respected")
    assert check.passed is False
    assert check.failure_code == "resolved_scope_outside_selected_documents"


def test_evaluator_output_shape_is_stable_and_deterministic() -> None:
    final_result = _valid_final_result()
    debug_payload = {"warnings": ["a"], "meta": {"selected_document_ids": ["doc-1"]}}

    first = evaluate_contract_checks(final_result=final_result, debug_payload=debug_payload)
    second = evaluate_contract_checks(final_result=final_result, debug_payload=debug_payload)

    assert first.to_dict() == second.to_dict()
    as_dict = first.to_dict()
    assert as_dict["evaluator_name"] == "contract_checks_v1"
    assert isinstance(as_dict["checks"], list)
    assert len(as_dict["checks"]) == 6


def test_contract_evaluator_entrypoint_callable_for_future_offline_runner() -> None:
    result = evaluate_contract_checks(final_result=_valid_final_result(), debug_payload=None)

    assert result.metadata["deterministic"] is True
    assert result.metadata["model_based_judgment"] is False
    assert result.metadata["check_count"] == 6
