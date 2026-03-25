from __future__ import annotations

from agentic_rag.tools import decompose_query, rewrite_query
from agentic_rag.tools.query_intelligence import QueryTransformationService


class _FakeLLM:
    def __init__(self, response: str, raises: bool = False) -> None:
        self.response = response
        self.raises = raises
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("llm failure")
        return self.response


def test_rewrite_query_returns_structured_result() -> None:
    result = rewrite_query("What is the negligence standard in California?")

    assert result.original_query == "What is the negligence standard in California?"
    assert isinstance(result.rewritten_query, str)
    assert isinstance(result.used_conversation_context, bool)
    assert result.rewrite_notes


def test_rewrite_query_preserves_legal_meaning() -> None:
    query = "Under New York law, what are the elements of breach of contract in 2024?"

    result = rewrite_query(query)

    assert result.rewritten_query == query
    assert "New York" in result.rewritten_query
    assert "2024" in result.rewritten_query


def test_rewrite_query_handles_empty_input_safely() -> None:
    assert rewrite_query("").rewritten_query == ""
    assert rewrite_query("   ").rewritten_query == ""


def test_rewrite_query_resolves_references_when_context_is_provided() -> None:
    summary = "We discussed Section 2.1 of the MSA termination rights."

    result = rewrite_query("How is that clause enforced?", conversation_summary=summary)

    assert result.used_conversation_context is True
    assert "Section 2.1" in result.rewritten_query


def test_rewrite_query_does_not_inject_context_when_unnecessary() -> None:
    query = "Interpret Section 10 indemnity language under Delaware law."

    result = rewrite_query(
        query,
        conversation_summary="Earlier we reviewed Section 2.1 in a different contract.",
        recent_messages=["Please keep discussing Section 2.1."],
    )

    assert result.rewritten_query == query
    assert result.used_conversation_context is False


def test_decompose_query_returns_structured_result() -> None:
    result = decompose_query("What is consideration in contract law?")

    assert result.original_query == "What is consideration in contract law?"
    assert isinstance(result.sub_queries, tuple)
    assert isinstance(result.used_conversation_context, bool)
    assert result.decomposition_notes


def test_decompose_query_returns_single_subquery_for_simple_query() -> None:
    result = decompose_query("What is promissory estoppel?")

    assert result.sub_queries == ("What is promissory estoppel?",)


def test_decompose_query_returns_multiple_for_complex_queries() -> None:
    query = "What is the rule and exception for hearsay in New York; definition and remedy for violation"

    result = decompose_query(query)

    assert len(result.sub_queries) > 1
    assert any("rule" in part.lower() for part in result.sub_queries)
    assert any("exception" in part.lower() for part in result.sub_queries)


def test_decompose_query_preserves_deterministic_ordering() -> None:
    query = "definition and enforcement and remedy of liquidated damages"

    first = decompose_query(query)
    second = decompose_query(query)

    assert first.sub_queries == second.sub_queries


def test_both_tools_correctly_use_or_ignore_context() -> None:
    ambiguous = rewrite_query(
        "Does this case discuss duty?",
        conversation_summary="We are analyzing Smith v. Jones from the California Court of Appeal.",
    )
    self_contained = decompose_query(
        "Compare procedural and substantive unconscionability in California.",
        conversation_summary="Older topic about hearsay exceptions.",
        recent_messages=["Prior case was unrelated."],
    )

    assert ambiguous.used_conversation_context is True
    assert "Smith v. Jones" in ambiguous.rewritten_query
    assert self_contained.used_conversation_context is False


def test_llm_is_used_only_when_needed() -> None:
    llm = _FakeLLM('{\"rewritten_query\": \"How is Section 3 enforced?\"}')
    service = QueryTransformationService(llm_client=llm)

    clear = service.rewrite_query("Interpret Section 10 under Delaware law.")
    ambiguous = service.rewrite_query(
        "How is that clause enforced?",
        conversation_summary="We discussed Section 3 in the MSA.",
    )

    assert clear.rewritten_query == "Interpret Section 10 under Delaware law."
    assert ambiguous.rewritten_query == "How is Section 3 enforced?"
    assert llm.calls == 1


def test_safe_fallbacks_on_llm_failure() -> None:
    service = QueryTransformationService(llm_client=_FakeLLM("{}", raises=True))

    rewrite = service.rewrite_query(
        "How is that clause enforced?",
        conversation_summary="We discussed Section 3 in the MSA.",
    )
    decompose = service.decompose_query("definition and remedy for unconscionability")

    assert rewrite.rewritten_query == "How is that clause enforced?"
    assert rewrite.rewrite_notes == "llm_failure_fallback_original_query"
    assert decompose.sub_queries == ("definition and remedy for unconscionability",)
    assert decompose.decomposition_notes == "llm_failure_fallback_original_query"
