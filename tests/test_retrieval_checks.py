from __future__ import annotations

from evals.graders.retrieval_checks import (
    aggregate_retrieval_results_by_family,
    evaluate_retrieval_checks,
)


def _base_case() -> dict[str, object]:
    return {
        "id": "case-1",
        "family": "chronology_date_event",
        "gold_evidence_ids": ["gold-eu-1", "gold-eu-2"],
        "gold_parent_chunk_ids": ["p-gold-1", "p-gold-2"],
    }


def test_gold_chunk_recall_at_k_single_query() -> None:
    result = evaluate_retrieval_checks(
        eval_case=_base_case(),
        retrieval_payload={
            "reranked_child_results": [
                {"child_chunk_id": "c1", "parent_chunk_id": "p1", "payload": {"evidence_unit_id": "x"}},
                {"child_chunk_id": "c2", "parent_chunk_id": "p2", "payload": {"evidence_unit_id": "gold-eu-1"}},
            ]
        },
    )

    recall5 = next(item for item in result.metrics if item.metric_name == "gold_chunk_recall_at_5")
    recall10 = next(item for item in result.metrics if item.metric_name == "gold_chunk_recall_at_10")
    assert recall5.value == 1.0
    assert recall10.value == 1.0
    assert recall5.details["matched_gold_ids"] == ["gold-eu-1"]


def test_gold_chunk_recall_at_k_decomposed_prefers_final_global_rerank_pool() -> None:
    result = evaluate_retrieval_checks(
        eval_case=_base_case(),
        retrieval_payload={
            "decomposition_plan": {"should_decompose": True},
            "merged_candidates": [{"hit": {"child_chunk_id": "x"}}],
            "reranked_child_results": [
                {"child_chunk_id": "root-1", "parent_chunk_id": "p-root", "payload": {"evidence_unit_id": "not-gold"}},
            ],
            "parent_expansion_child_results": [
                {"child_chunk_id": "merged-1", "parent_chunk_id": "p-gold-1", "payload": {"evidence_unit_id": "gold-eu-2"}},
            ],
        },
    )

    best_rank = next(item for item in result.metrics if item.metric_name == "best_rank_of_gold_evidence")
    assert best_rank.details["best_rank"] == 1


def test_gold_parent_recall_uses_parent_expansion_shapes() -> None:
    result = evaluate_retrieval_checks(
        eval_case=_base_case(),
        retrieval_payload={
            "parent_chunks": [{"parent_chunk_id": "p-gold-2"}],
            "parent_ids": ["p-gold-1"],
            "reranked_child_results": [{"child_chunk_id": "c1", "parent_chunk_id": "p-gold-1"}],
        },
    )

    metric = next(item for item in result.metrics if item.metric_name == "gold_parent_recall")
    assert metric.value == 1.0
    assert metric.passed is True


def test_best_rank_reports_missing_gold_evidence_cleanly() -> None:
    result = evaluate_retrieval_checks(
        eval_case=_base_case(),
        retrieval_payload={
            "reranked_child_results": [
                {"child_chunk_id": "c1", "parent_chunk_id": "p1", "payload": {"evidence_unit_id": "x"}},
                {"child_chunk_id": "c2", "parent_chunk_id": "p2", "payload": {"evidence_unit_id": "y"}},
            ]
        },
    )

    metric = next(item for item in result.metrics if item.metric_name == "best_rank_of_gold_evidence")
    assert metric.value is None
    assert metric.note == "No gold evidence appeared in the ranked retrieval list."


def test_wrong_family_usage_rate_is_deterministic_from_structural_metadata() -> None:
    case = _base_case()
    case["family"] = "party_role_verification"
    result = evaluate_retrieval_checks(
        eval_case=case,
        retrieval_payload={
            "reranked_child_results": [
                {"child_chunk_id": "c1", "parent_chunk_id": "p1", "payload": {"legal_family": "party_role_entity"}},
                {"child_chunk_id": "c2", "parent_chunk_id": "p2", "payload": {"legal_family": "chronology_date_event"}},
                {"child_chunk_id": "c3", "parent_chunk_id": "p3", "payload": {"legal_family": "chronology_date_event"}},
            ]
        },
    )

    metric = next(item for item in result.metrics if item.metric_name == "wrong_family_evidence_usage_rate")
    assert metric.value == 2 / 3
    assert metric.details["dominated_by_wrong_family"] is True


def test_per_case_output_shape_is_stable_and_machine_readable() -> None:
    case = _base_case()
    payload = {"reranked_child_results": [{"child_chunk_id": "c1", "parent_chunk_id": "p-gold-1", "payload": {"evidence_unit_id": "gold-eu-1"}}]}
    first = evaluate_retrieval_checks(eval_case=case, retrieval_payload=payload)
    second = evaluate_retrieval_checks(eval_case=case, retrieval_payload=payload)

    assert first.to_dict() == second.to_dict()
    as_dict = first.to_dict()
    assert as_dict["evaluator_name"] == "retrieval_checks_v1"
    assert as_dict["case_family"] == "chronology_date_event"
    assert isinstance(as_dict["metrics"], list)


def test_aggregation_by_family_computes_counts_recall_rank_and_wrong_family_rate() -> None:
    case_a = _base_case()
    result_a = evaluate_retrieval_checks(
        eval_case=case_a,
        retrieval_payload={"reranked_child_results": [{"child_chunk_id": "c1", "parent_chunk_id": "p-gold-1", "payload": {"evidence_unit_id": "gold-eu-1", "legal_family": "chronology_date_event"}}]},
    )

    case_b = _base_case()
    case_b["id"] = "case-2"
    result_b = evaluate_retrieval_checks(
        eval_case=case_b,
        retrieval_payload={"reranked_child_results": [{"child_chunk_id": "c9", "parent_chunk_id": "p9", "payload": {"evidence_unit_id": "miss", "legal_family": "financial_entitlement"}}]},
    )

    grouped = aggregate_retrieval_results_by_family([result_a, result_b])
    chrono = grouped["chronology_date_event"]
    assert chrono["case_count"] == 2
    assert "k_5" in chrono["recall_at_k"]
    assert chrono["best_rank"]["count"] == 1


def test_evaluator_handles_single_and_decomposed_inputs_via_one_interface() -> None:
    case = _base_case()
    single = evaluate_retrieval_checks(
        eval_case=case,
        retrieval_payload={"reranked_child_results": [{"child_chunk_id": "c1", "parent_chunk_id": "p1", "payload": {"evidence_unit_id": "gold-eu-1"}}]},
    )
    decomposed = evaluate_retrieval_checks(
        eval_case=case,
        retrieval_payload={
            "decomposition_plan": {"should_decompose": True},
            "merged_candidates": [{"hit": {"child_chunk_id": "c2"}}],
            "parent_expansion_child_results": [{"child_chunk_id": "c2", "parent_chunk_id": "p2", "payload": {"evidence_unit_id": "gold-eu-2"}}],
        },
    )

    assert single.metadata["retrieval_mode"] == "single_query"
    assert decomposed.metadata["retrieval_mode"] == "decomposed"


def test_retrieval_evaluator_entrypoint_callable_for_offline_runner_integration() -> None:
    result = evaluate_retrieval_checks(
        eval_case=_base_case(),
        retrieval_payload={"reranked_child_results": []},
    )

    assert result.metadata["deterministic"] is True
    assert result.metadata["model_based_judgment"] is False
