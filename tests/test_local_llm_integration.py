from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.llm import LocalLLMConfig, build_local_prompt_llm, local_llm_config_from_env
from agentic_rag.orchestration.retrieval_graph import llm_assisted_decomposition_plan
from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult
from agentic_rag.retrieval import ParentChunkResult
from agentic_rag.tools.answer_generation import LegalAnswerSynthesizer
from agentic_rag.tools.query_intelligence import QueryTransformationService


@dataclass
class _FakePromptClient:
    response: str
    should_raise: bool = False

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if self.should_raise:
            raise RuntimeError("boom")
        return self.response


def _decision() -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        original_query="q",
        normalized_query="q",
        question_type="comparison_query",
        is_followup=False,
        is_context_dependent=False,
        use_conversation_context=False,
        is_document_scoped=True,
        refers_to_prior_document_scope=False,
        refers_to_prior_clause_or_topic=False,
        should_rewrite=True,
        should_extract_entities=True,
        should_retrieve=True,
        may_need_decomposition=True,
        answerability_expectation="comparison",
        resolved_document_hints=[],
        resolved_topic_hints=[],
        resolved_clause_hints=[],
        ambiguity_notes=[],
        routing_notes=[],
        warnings=[],
    )


def _parent() -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id="p-1",
        document_id="doc-1",
        text="This Agreement is governed by Delaware law.",
        source="s3://doc.md",
        source_name="doc.md",
        heading_path=("MSA", "Governing Law"),
        heading_text="Governing Law",
        parent_order=0,
        part_number=1,
        total_parts=1,
        metadata={},
    )


def test_local_llm_config_loads_from_env() -> None:
    cfg = local_llm_config_from_env(
        {
            "AGENTIC_RAG_LOCAL_LLM_ENABLED": "true",
            "AGENTIC_RAG_LOCAL_LLM_PROVIDER": "ollama",
            "AGENTIC_RAG_LOCAL_LLM_MODEL": "llama3.1:8b-instruct-q4_0",
            "AGENTIC_RAG_LOCAL_LLM_BASE_URL": "http://localhost:11434",
            "AGENTIC_RAG_LOCAL_LLM_TEMPERATURE": "0.2",
            "AGENTIC_RAG_LOCAL_LLM_TIMEOUT_SECONDS": "12",
        }
    )

    assert cfg.enabled is True
    assert cfg.provider == "ollama"
    assert cfg.model == "llama3.1:8b-instruct-q4_0"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.temperature == 0.2
    assert cfg.timeout_seconds == 12


def test_build_local_prompt_llm_returns_none_when_disabled() -> None:
    client = build_local_prompt_llm(LocalLLMConfig(enabled=False))
    assert client is None


def test_query_rewrite_uses_provider_when_enabled() -> None:
    service = QueryTransformationService(llm_client=_FakePromptClient('{"rewritten_query":"How is Section 9 enforced?"}'))

    result = service.rewrite_query("How is that clause enforced?", conversation_summary="We discussed Section 9.")

    assert result.rewritten_query == "How is Section 9 enforced?"
    assert result.rewrite_notes.startswith("resolved_reference_with_llm")


def test_decomposition_planning_falls_back_when_provider_unavailable(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as rg

    monkeypatch.setattr(rg, "build_local_prompt_llm_from_env", lambda: None)
    monkeypatch.setattr(
        rg,
        "local_llm_config_from_env",
        lambda: LocalLLMConfig(enabled=True, provider="ollama", model="mistral:7b"),
    )

    plan = llm_assisted_decomposition_plan(
        query="Compare indemnity and liability clauses.",
        needs_decomposition=True,
        reasons=["comparison_query"],
        query_classification=_decision(),
        context_resolution=None,
    )

    assert plan is not None
    assert any(note.startswith("planner_path:deterministic_fallback") for note in plan.planner_notes)


def test_decomposition_planning_uses_provider_when_enabled(monkeypatch) -> None:
    import agentic_rag.orchestration.retrieval_graph as rg

    payload = (
        '{"strategy":"comparison","subqueries":['
        '{"id":"sq-1","question":"Find indemnity clause text.","purpose":"Find indemnity evidence.","required":true,"expected_answer_type":"cross_reference","dependency_ids":[]},'
        '{"id":"sq-2","question":"Find limitation clause text.","purpose":"Find liability evidence.","required":true,"expected_answer_type":"cross_reference","dependency_ids":[]}'
        "]}"
    )
    monkeypatch.setattr(rg, "build_local_prompt_llm_from_env", lambda: _FakePromptClient(payload))
    monkeypatch.setattr(
        rg,
        "local_llm_config_from_env",
        lambda: LocalLLMConfig(enabled=True, provider="ollama", model="llama3.1:8b"),
    )

    plan = llm_assisted_decomposition_plan(
        query="Compare indemnity and liability clauses.",
        needs_decomposition=True,
        reasons=["comparison_query"],
        query_classification=_decision(),
        context_resolution=None,
    )

    assert plan is not None
    assert len(plan.subqueries) == 2
    assert any(note.startswith("planner_path:llm") for note in plan.planner_notes)


def test_final_synthesis_uses_provider_with_deterministic_citations() -> None:
    synth = LegalAnswerSynthesizer(
        llm_client=_FakePromptClient("Grounded draft from provided evidence only."),
        llm_provider_label="ollama:llama3.1:8b",
    )

    result = synth.generate([_parent()], "What law governs the agreement?")

    assert result.sufficient_context is True
    assert result.citations and result.citations[0].parent_chunk_id == "p-1"
    assert any(warning.startswith("answer_synthesis_path:llm") for warning in result.warnings)


def test_final_synthesis_has_deterministic_fallback_when_provider_fails() -> None:
    synth = LegalAnswerSynthesizer(
        llm_client=_FakePromptClient("", should_raise=True),
        llm_provider_label="ollama:llama3.1:8b",
    )

    result = synth.generate([_parent()], "What law governs the agreement?")

    assert result.sufficient_context is True
    assert result.citations
    assert any(warning.startswith("answer_synthesis_path:deterministic_fallback") for warning in result.warnings)
