from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")
ValidationError = pydantic.ValidationError

from agentic_rag.orchestration.retrieval_graph import DecompositionPlan, SubQueryPlan


def test_subquery_plan_accepts_valid_values_and_dependency_ids_default_is_safe() -> None:
    first = SubQueryPlan(
        id="sq-1",
        question="What is the notice period under the termination clause?",
        purpose="Identify the baseline notice obligation for later comparison.",
        expected_answer_type="obligation",
    )
    second = SubQueryPlan(
        id="sq-2",
        question="How does the amendment change the notice period?",
        purpose="Capture the amended notice obligation.",
        expected_answer_type="comparison",
    )

    assert first.expected_answer_type == "obligation"
    assert first.dependency_ids == []
    assert second.dependency_ids == []
    assert first.dependency_ids is not second.dependency_ids


def test_decomposition_plan_accepts_non_decomposed_state_with_safe_defaults() -> None:
    first = DecompositionPlan(
        should_decompose=False,
        root_question="What is the confidentiality obligation in section 8?",
    )
    second = DecompositionPlan(
        should_decompose=False,
        root_question="What remedies apply for late payment?",
    )

    assert first.should_decompose is False
    assert first.subqueries == []
    assert first.planner_notes == []
    assert second.subqueries == []
    assert second.planner_notes == []
    assert first.subqueries is not second.subqueries
    assert first.planner_notes is not second.planner_notes


def test_decomposition_plan_accepts_decomposed_state_with_one_valid_subquery() -> None:
    subquery = SubQueryPlan(
        id="sq-1",
        question="Which clause defines cause for termination?",
        purpose="Locate the definition before applying exception analysis.",
        expected_answer_type="definition",
    )

    plan = DecompositionPlan(
        should_decompose=True,
        root_question="How do cause definitions and exceptions interact for termination?",
        strategy="exception_chain",
        subqueries=[subquery],
    )

    assert plan.should_decompose is True
    assert plan.strategy == "exception_chain"
    assert len(plan.subqueries) == 1
    assert plan.subqueries[0].id == "sq-1"


def test_subquery_plan_rejects_invalid_expected_answer_type() -> None:
    with pytest.raises(ValidationError) as exc_info:
        SubQueryPlan(
            id="sq-invalid",
            question="When does payment become overdue?",
            purpose="Identify timing threshold.",
            expected_answer_type="timeline",
        )

    assert any(error.get("loc") == ("expected_answer_type",) for error in exc_info.value.errors())


def test_decomposition_plan_rejects_invalid_strategy() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DecompositionPlan(
            should_decompose=True,
            root_question="How should obligations be compared across clauses?",
            strategy="multi_hop",
            subqueries=[],
        )

    assert any(error.get("loc") == ("strategy",) for error in exc_info.value.errors())
