from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answer_generation import generate_answer
from agentic_rag.tools.answerability import assess_answerability


def _parent(pid: str, heading: str, text: str) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-financial",
        text=text,
        source="test",
        source_name="employment_financial_records.md",
        heading_path=("Employment Financial File", heading),
        heading_text=heading,
    )


def _financial_context() -> list[ParentChunkResult]:
    return [
        _parent(
            "f-1",
            "Compensation",
            "Promised compensation includes a base salary of $95,000 per year payable bi-weekly, plus standard benefits.",
        ),
        _parent(
            "f-2",
            "Unpaid Amounts Claimed",
            "Demand letter lists unpaid wages of $12,500 and unpaid vacation pay of $2,300 as outstanding amounts.",
        ),
        _parent(
            "f-3",
            "Bonus and Vacation Pay",
            "Employee is entitled to a prorated annual bonus and accrued vacation pay for unused days at termination.",
        ),
        _parent(
            "f-4",
            "Expense Reimbursements",
            "Expense reimbursement records show mileage and travel receipts submitted, with $1,180 still unreimbursed.",
        ),
        _parent(
            "f-5",
            "Severance",
            "Severance payable is eight (8) weeks of base salary, subject to statutory minimums.",
        ),
        _parent(
            "f-6",
            "Payroll Records",
            "Payroll records and pay stubs for each pay period support the wage claim and identify missing payments.",
        ),
        _parent(
            "f-7",
            "Confidentiality",
            (
                "Employee must maintain confidentiality of proprietary information during and after employment, "
                "must not disclose confidential business information, and must follow all confidentiality "
                "controls listed in the agreement unless disclosure is required by law."
            ),
        ),
    ]


def test_financial_question_recognized_as_entitlement_query() -> None:
    result = understand_query("What compensation was promised?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:financial_entitlement" in result.routing_notes


def test_compensation_question_uses_compensation_evidence_when_present() -> None:
    query = "What compensation was promised?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _financial_context())
    answer = generate_answer(_financial_context(), query)

    assert result.sufficient_context is True
    assert "financial_entitlement_compensation_evidence_detected" in result.evidence_notes
    assert "$95,000" in answer.answer_text


def test_unpaid_amount_question_uses_financial_claim_evidence_when_present() -> None:
    query = "What amounts are unpaid?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _financial_context())
    answer = generate_answer(_financial_context(), query)

    assert result.sufficient_context is True
    assert "financial_entitlement_unpaid_amount_evidence_detected" in result.evidence_notes
    assert "unpaid wages" in answer.answer_text.lower()


def test_bonus_or_vacation_pay_question_uses_responsive_entitlement_evidence_when_present() -> None:
    query = "What bonus or vacation pay is claimed?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _financial_context())
    answer = generate_answer(_financial_context(), query)

    assert result.sufficient_context is True
    assert "financial_entitlement_bonus_or_vacation_evidence_detected" in result.evidence_notes
    assert "bonus" in answer.answer_text.lower() or "vacation" in answer.answer_text.lower()


def test_reimbursement_question_uses_expense_or_reimbursement_evidence_when_present() -> None:
    query = "What reimbursements are at issue?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _financial_context())
    answer = generate_answer(_financial_context(), query)

    assert result.sufficient_context is True
    assert "financial_entitlement_reimbursement_evidence_detected" in result.evidence_notes
    assert "reimbursement" in answer.answer_text.lower() or "expense" in answer.answer_text.lower()


def test_financial_records_support_question_uses_pay_stub_or_record_evidence_when_present() -> None:
    query = "What financial records support the claim?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _financial_context())
    answer = generate_answer(_financial_context(), query)

    assert result.sufficient_context is True
    assert "financial_entitlement_records_evidence_detected" in result.evidence_notes
    assert "pay stub" in answer.answer_text.lower() or "payroll" in answer.answer_text.lower()


def test_missing_or_ambiguous_financial_evidence_fails_safely() -> None:
    query = "What amounts are unpaid?"
    understanding = understand_query(query)
    context = [
        _parent(
            "f-x",
            "Confidentiality",
            "Employee must maintain confidentiality and return proprietary information on request.",
        )
    ]

    result = assess_answerability(query, understanding, context)
    answer = generate_answer(context, query)

    assert result.sufficient_context is False
    assert result.insufficiency_reason == "fact_not_found"
    assert answer.sufficient_context is False


def test_non_financial_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)
    result = assess_answerability(query, understanding, _financial_context())

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
