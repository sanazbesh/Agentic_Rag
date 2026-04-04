from agentic_rag.orchestration.decomposition_gate import (
    CATEGORY_REGISTRY,
    CONSERVATIVE_REASONS,
    REASON_PRECEDENCE,
    STRONG_REASONS,
    decide_decomposition_need,
)
from agentic_rag.orchestration.query_understanding import understand_query


def test_reason_labels_are_centralized_and_stable() -> None:
    assert REASON_PRECEDENCE == (
        "comparison_query",
        "multi_intent_conjunction",
        "amendment_vs_base",
        "temporal_relationship",
        "exception_chain",
        "cross_clause_obligation_condition",
        "context_dependent_followup",
        "simple_single_clause_lookup",
    )


def test_category_registry_maps_expected_labels_and_strengths() -> None:
    by_label = {entry.label: entry.strong_by_default for entry in CATEGORY_REGISTRY}
    assert by_label == {
        "comparison_query": True,
        "multi_intent_conjunction": False,
        "amendment_vs_base": True,
        "temporal_relationship": False,
        "exception_chain": True,
        "cross_clause_obligation_condition": False,
        "context_dependent_followup": False,
    }
    assert STRONG_REASONS == {"comparison_query", "amendment_vs_base", "exception_chain"}
    assert CONSERVATIVE_REASONS == {
        "multi_intent_conjunction",
        "temporal_relationship",
        "cross_clause_obligation_condition",
        "context_dependent_followup",
    }


def test_simple_single_clause_lookup_is_conservative_false() -> None:
    decision = decide_decomposition_need(query="What is confidentiality?")
    assert decision.needs_decomposition is False
    assert decision.reasons == ["simple_single_clause_lookup"]


def test_comparison_true_positive_is_strong() -> None:
    decision = decide_decomposition_need(query="Compare governing law versus dispute resolution.")
    assert decision.needs_decomposition is True
    assert "comparison_query" in decision.reasons


def test_multi_intent_true_positive_but_conservative_only() -> None:
    decision = decide_decomposition_need(query="Who are the parties and what are their obligations?")
    assert decision.needs_decomposition is False
    assert decision.reasons == ["multi_intent_conjunction"]


def test_multi_intent_false_positive_protection_for_ordinary_and() -> None:
    decision = decide_decomposition_need(query="What is the title and date?")
    assert decision.needs_decomposition is False
    assert decision.reasons == ["simple_single_clause_lookup"]


def test_amendment_vs_base_true_positive_is_strong() -> None:
    decision = decide_decomposition_need(
        query="How did the original agreement change after the amendment?"
    )
    assert decision.needs_decomposition is True
    assert "amendment_vs_base" in decision.reasons


def test_temporal_relationship_true_positive_but_conservative_only() -> None:
    decision = decide_decomposition_need(query="How did notice obligations change over time?")
    assert decision.needs_decomposition is False
    assert decision.reasons == ["temporal_relationship"]


def test_temporal_relationship_false_positive_protection_for_pure_date() -> None:
    decision = decide_decomposition_need(query="What is the effective date in 2020?")
    assert decision.needs_decomposition is False
    assert "temporal_relationship" not in decision.reasons


def test_exception_chain_true_positive_is_strong() -> None:
    decision = decide_decomposition_need(
        query="What obligations apply except as otherwise provided in section 5?"
    )
    assert decision.needs_decomposition is True
    assert "exception_chain" in decision.reasons


def test_cross_clause_true_positive_but_conservative_only() -> None:
    decision = decide_decomposition_need(
        query="Which section covers notice and cure obligations in this agreement?"
    )
    assert decision.needs_decomposition is False
    assert decision.reasons == ["multi_intent_conjunction", "cross_clause_obligation_condition"]


def test_cross_clause_false_positive_protection_for_shallow_two_topic_phrase() -> None:
    decision = decide_decomposition_need(query="notice and cure")
    assert decision.needs_decomposition is False
    assert "cross_clause_obligation_condition" not in decision.reasons


def test_followup_alone_does_not_trigger_or_label() -> None:
    understanding = understand_query(
        "What about that clause?",
        conversation_summary="Prior turn discussed termination.",
        recent_messages=[{"role": "user", "content": "Earlier asked about termination notice."}],
    )
    decision = decide_decomposition_need(
        query="What about that clause?",
        query_understanding=understanding,
        query_context={"used_conversation_context": True, "unresolved_references": []},
    )
    assert decision.needs_decomposition is False
    assert "context_dependent_followup" not in decision.reasons


def test_followup_is_supporting_label_when_strong_structure_present() -> None:
    understanding = understand_query(
        "Compare governing law versus dispute resolution in that clause.",
        conversation_summary="Prior turn discussed governing law section.",
        recent_messages=[{"role": "assistant", "content": "Governing law is Delaware."}],
    )
    decision = decide_decomposition_need(
        query="Compare governing law versus dispute resolution in that clause.",
        query_understanding=understanding,
        query_context={"used_conversation_context": True, "unresolved_references": ["that clause"]},
    )
    assert decision.needs_decomposition is True
    assert decision.reasons == [
        "comparison_query",
        "cross_clause_obligation_condition",
        "context_dependent_followup",
    ]


def test_mixed_strong_and_weak_categories_follow_one_bool_rule() -> None:
    decision = decide_decomposition_need(
        query="Compare parties and obligations in this agreement before and after amendment unless provided that clause 3 applies.",
    )
    assert decision.needs_decomposition is True
    assert decision.reasons == [
        "comparison_query",
        "multi_intent_conjunction",
        "amendment_vs_base",
        "temporal_relationship",
        "exception_chain",
        "cross_clause_obligation_condition",
    ]


def test_repeated_identical_inputs_are_fully_deterministic() -> None:
    query = "Compare parties and obligations in this agreement unless provided that clause 3 applies."
    first = decide_decomposition_need(query=query)
    second = decide_decomposition_need(query=query)
    assert first.needs_decomposition == second.needs_decomposition
    assert first.reasons == second.reasons


def test_may_need_decomposition_hint_does_not_override_gate() -> None:
    second = understand_query("alpha and beta?")
    assert second.may_need_decomposition is True
    decision = decide_decomposition_need(
        query="alpha and beta?",
        query_understanding=second,
        query_context={"used_conversation_context": False, "unresolved_references": []},
    )
    assert decision.needs_decomposition is False
    assert decision.reasons == ["simple_single_clause_lookup"]
