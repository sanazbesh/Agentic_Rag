from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, heading: str, text: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-policy-issue",
        text=text,
        source="test",
        source_name="dispute_policy_issue_file.md",
        heading_path=("Dispute File", heading),
        heading_text=heading,
    )


def _policy_issue_context() -> list[ParentChunkResult]:
    return [
        _parent(
            "pi-1",
            "Workplace Policies",
            (
                "Employee Handbook policy references anti-harassment policy, progressive discipline policy, "
                "and complaint reporting procedure as relevant to the dispute allegations."
            ),
        ),
        _parent(
            "pi-2",
            "Issue Summary",
            (
                "Key legal issues raised include wrongful dismissal, alleged ESA violation for unpaid wages, "
                "and alleged reprisal after protected complaint activity."
            ),
        ),
        _parent(
            "pi-3",
            "Claim Framing",
            (
                "Nature of the claim is wrongful dismissal and statutory violation; the pleading seeks damages "
                "for unpaid wages and declaratory relief for breach of statutory rights."
            ),
        ),
        _parent(
            "pi-4",
            "Contract Clauses",
            (
                "Termination clause and discipline procedure clause are identified as clauses related to the dispute, "
                "along with handbook complaint policy provisions."
            ),
        ),
        _parent(
            "pi-5",
            "Procedural Status",
            "Procedural status note links filed claim allegations to outstanding statutory and policy issues in dispute.",
        ),
        _parent(
            "pi-6",
            "Confidentiality",
            (
                "Employee must maintain confidentiality of proprietary information and must not disclose business records "
                "except as required by law."
            ),
        ),
    ]


def test_policy_or_issue_question_recognized_as_legal_issue_spotting_query() -> None:
    result = understand_query("What legal issues are raised?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:policy_issue_spotting" in result.routing_notes


def test_relevant_policy_question_uses_policy_responsive_evidence_when_present() -> None:
    query = "What policies are relevant?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _policy_issue_context())
    answer = generate_answer(_policy_issue_context(), query)

    assert result.sufficient_context is True
    assert "policy_issue_relevant_policy_evidence_detected" in result.evidence_notes
    assert "policy" in answer.answer_text.lower() or "handbook" in answer.answer_text.lower()


def test_legal_issue_question_uses_issue_framing_evidence_when_present() -> None:
    query = "What legal issues are raised?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _policy_issue_context())
    answer = generate_answer(_policy_issue_context(), query)

    assert result.sufficient_context is True
    assert "policy_issue_legal_issue_framing_evidence_detected" in result.evidence_notes
    assert "wrongful dismissal" in answer.answer_text.lower() or "issue" in answer.answer_text.lower()


def test_dispute_clause_question_uses_clause_or_policy_evidence_when_present() -> None:
    query = "What clauses or policies relate to this dispute?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _policy_issue_context())
    answer = generate_answer(_policy_issue_context(), query)

    assert result.sufficient_context is True
    assert "policy_issue_dispute_clause_or_policy_evidence_detected" in result.evidence_notes
    assert "clause" in answer.answer_text.lower() or "policy" in answer.answer_text.lower()


def test_nature_of_claim_question_uses_responsive_claim_framing_when_present() -> None:
    query = "What is the nature of the claim?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _policy_issue_context())
    answer = generate_answer(_policy_issue_context(), query)

    assert result.sufficient_context is True
    assert "policy_issue_nature_of_claim_evidence_detected" in result.evidence_notes
    assert "claim" in answer.answer_text.lower() or "wrongful dismissal" in answer.answer_text.lower()


def test_key_issues_question_uses_issue_responsive_evidence_when_present() -> None:
    query = "What are the key issues in the file?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _policy_issue_context())
    answer = generate_answer(_policy_issue_context(), query)

    assert result.sufficient_context is True
    assert "policy_issue_key_issues_evidence_detected" in result.evidence_notes
    assert "issue" in answer.answer_text.lower() or "esa" in answer.answer_text.lower()


def test_missing_or_ambiguous_policy_or_issue_evidence_fails_safely() -> None:
    query = "What legal issues are raised?"
    understanding = understand_query(query)
    context = [
        _parent(
            "pi-x",
            "Confidentiality",
            "Employee must maintain confidentiality and return proprietary records at termination.",
        )
    ]

    result = assess_answerability(query, understanding, context)
    answer = generate_answer(context, query)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"
    assert answer.sufficient_context is False


def test_non_policy_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _policy_issue_context())

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
