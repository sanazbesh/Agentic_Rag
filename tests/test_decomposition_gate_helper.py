from agentic_rag.orchestration.decomposition_gate import (
    REASON_PRECEDENCE,
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


def test_simple_single_clause_lookup_is_conservative_false() -> None:
    decision = decide_decomposition_need(query="What is confidentiality?")
    assert decision.needs_decomposition is False
    assert decision.reasons == ["simple_single_clause_lookup"]


def test_multi_reason_order_is_deterministic_from_precedence() -> None:
    query = "Compare parties obligations and rights in this agreement unless provided that exceptions apply."
    decision = decide_decomposition_need(query=query)
    assert decision.needs_decomposition is True
    assert decision.reasons == [
        "comparison_query",
        "multi_intent_conjunction",
        "exception_chain",
        "cross_clause_obligation_condition",
    ]


def test_amendment_change_and_temporal_signals_trigger() -> None:
    decision = decide_decomposition_need(
        query="How did the amendment change confidentiality obligations over time?"
    )
    assert decision.needs_decomposition is True
    assert decision.reasons == ["amendment_vs_base", "temporal_relationship"]


def test_context_followup_requires_explicit_supporting_signal() -> None:
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


def test_followup_can_trigger_when_multi_intent_is_explicit() -> None:
    understanding = understand_query(
        "What about cure rights and notice in that clause?",
        conversation_summary="Prior turn discussed termination section.",
        recent_messages=[{"role": "assistant", "content": "Termination notice is 30 days."}],
    )
    decision = decide_decomposition_need(
        query="What about cure rights and notice in that clause?",
        query_understanding=understanding,
        query_context={"used_conversation_context": True, "unresolved_references": []},
    )
    assert decision.needs_decomposition is True
    assert decision.reasons == ["multi_intent_conjunction", "context_dependent_followup"]


def test_may_need_decomposition_hint_does_not_override_gate() -> None:
    understanding = understand_query("What is governing law?")
    assert understanding.may_need_decomposition is False
    second = understand_query("alpha and beta?")
    assert second.may_need_decomposition is True
    decision = decide_decomposition_need(
        query="alpha and beta?",
        query_understanding=second,
        query_context={"used_conversation_context": False, "unresolved_references": []},
    )
    assert decision.needs_decomposition is False
    assert decision.reasons == ["simple_single_clause_lookup"]
