from evals.graders.family_routing import (
    aggregate_family_routing_results,
    build_family_confusion_matrix,
    evaluate_family_routing,
)


def _case(*, case_id: str = "case-1", family: str = "employment_lifecycle") -> dict[str, str]:
    return {"id": case_id, "family": family}


def test_family_routing_scores_expected_and_predicted_family_from_routing_notes() -> None:
    result = evaluate_family_routing(
        eval_case=_case(),
        system_output={
            "query_classification": {
                "routing_notes": [
                    "rewrite_recommended",
                    "legal_question_family:employment_contract_lifecycle",
                ]
            }
        },
    )

    assert result.expected_family == "employment_lifecycle"
    assert result.predicted_family == "employment_lifecycle"
    assert result.is_correct is True
    assert result.confusion_cell == {
        "expected_family": "employment_lifecycle",
        "predicted_family": "employment_lifecycle",
        "count": 1,
        "correct": True,
    }


def test_family_routing_prefers_direct_predicted_family_when_present() -> None:
    result = evaluate_family_routing(
        eval_case=_case(family="policy_issue_spotting"),
        system_output={
            "predicted_family": "policy_issue_spotting",
            "query_classification": {
                "routing_notes": ["legal_question_family:financial_entitlement"],
            },
        },
    )

    assert result.predicted_family == "policy_issue_spotting"
    assert result.is_correct is True
    assert result.metadata["predicted_family_resolution_source"] == "system_output.predicted_family"


def test_family_routing_records_mismatch_and_unresolved_predictions() -> None:
    mismatch = evaluate_family_routing(
        eval_case=_case(family="chronology_date_event"),
        system_output={"routing_notes": ["legal_question_family:party_role_entity"]},
    )
    unresolved = evaluate_family_routing(eval_case=_case(), system_output={})

    assert mismatch.predicted_family == "party_role_verification"
    assert mismatch.is_correct is False
    assert unresolved.predicted_family is None
    assert unresolved.is_correct is False
    assert unresolved.confusion_cell["predicted_family"] == "unresolved"


def test_family_routing_confusion_matrix_and_aggregation_are_explicit() -> None:
    results = [
        evaluate_family_routing(
            eval_case=_case(case_id="a", family="employment_lifecycle"),
            system_output={"query_classification": {"routing_notes": ["legal_question_family:employment_contract_lifecycle"]}},
        ),
        evaluate_family_routing(
            eval_case=_case(case_id="b", family="employment_lifecycle"),
            system_output={"query_classification": {"routing_notes": ["legal_question_family:financial_entitlement"]}},
        ),
        evaluate_family_routing(
            eval_case=_case(case_id="c", family="policy_issue_spotting"),
            system_output={},
        ),
    ]

    matrix = build_family_confusion_matrix(results)
    assert matrix == {
        "employment_lifecycle": {
            "employment_lifecycle": 1,
            "financial_entitlement": 1,
        },
        "policy_issue_spotting": {
            "unresolved": 1,
        },
    }

    aggregate = aggregate_family_routing_results(results)
    assert aggregate["case_count"] == 3
    assert aggregate["correct_count"] == 1
    assert aggregate["family_confusion_matrix"] == matrix
    assert aggregate["by_expected_family"]["employment_lifecycle"]["accuracy"] == 0.5
    assert aggregate["by_expected_family"]["policy_issue_spotting"]["confusion_row"] == {"unresolved": 1}
