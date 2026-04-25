from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.retrieval import ParentChunkResult
from agentic_rag.tools import LegalAnswerSynthesizer, generate_answer


@dataclass
class _CompressedLike:
    parent_chunk_id: str
    document_id: str
    source_name: str
    heading_text: str
    compressed_text: str


def _parent(
    parent_id: str,
    heading: str,
    text: str,
    document_id: str = "doc-1",
    metadata: dict[str, str] | None = None,
) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=parent_id,
        document_id=document_id,
        text=text,
        source=f"s3://bucket/{parent_id}.md",
        source_name=f"{parent_id}.md",
        heading_path=("MSA", heading),
        heading_text=heading,
        parent_order=0,
        part_number=1,
        total_parts=1,
        metadata=metadata or {},
    )


def test_generate_answer_grounded_with_valid_citations() -> None:
    context = [
        _parent(
            "p-1",
            "Termination",
            "Either party may terminate for material breach if notice is provided and a 30-day cure period expires.",
        )
    ]

    result = generate_answer(context, "When can a party terminate for breach?")

    assert result.grounded is True
    assert result.sufficient_context is True
    assert result.citations
    assert result.citations[0].parent_chunk_id == "p-1"


def test_generate_answer_insufficient_context_safe_response() -> None:
    context = [_parent("p-1", "Definitions", "This section defines capitalized terms only.")]

    result = generate_answer(context, "What are the indemnification remedies?")

    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert "does not provide enough information" in result.answer_text.lower()


def test_generate_answer_empty_context_fallback() -> None:
    result = generate_answer([], "What is the governing law?")

    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert any("insufficient_context" in warning for warning in result.warnings)


def test_generate_answer_no_hallucination_for_missing_query_fact() -> None:
    context = [_parent("p-1", "Confidentiality", "Confidential information must be protected from disclosure.")]

    result = generate_answer(context, "What is the arbitration seat in Singapore?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert "singapore" not in result.answer_text.lower()


def test_generate_answer_citation_integrity_matches_used_context() -> None:
    context = [
        _parent("p-1", "Termination", "Termination is allowed for uncured material breach."),
        _parent("p-2", "Governing Law", "This Agreement is governed by Delaware law."),
    ]

    result = generate_answer(context, "What law governs the agreement?")

    cited_ids = {citation.parent_chunk_id for citation in result.citations}
    assert cited_ids == {"p-2"}


def test_generate_answer_preserves_qualifier_meaning() -> None:
    text = (
        "Supplier shall deliver services within 10 days. "
        "Except as required by law, Supplier may not disclose Confidential Information."
    )
    result = generate_answer([_parent("p-1", "Confidentiality", text)], "Can supplier disclose confidential information?")

    assert result.grounded is True
    assert "except as required by law" in result.answer_text.lower()


def test_generate_answer_handles_multiple_parent_chunks() -> None:
    context = [
        _parent("p-1", "Payment", "Customer must pay undisputed fees within 30 days."),
        _parent("p-2", "Late Fees", "Late fees apply only to overdue undisputed amounts."),
    ]

    result = generate_answer(context, "What are the payment timing and late fee rules?")

    assert len(result.citations) >= 2
    assert {c.parent_chunk_id for c in result.citations[:2]} == {"p-1", "p-2"}


def test_generate_answer_is_deterministic() -> None:
    context = [_parent("p-1", "Termination", "Either party may terminate for material breach.")]
    first = generate_answer(context, "When can the contract be terminated?")
    second = generate_answer(context, "When can the contract be terminated?")

    assert first == second


def test_generate_answer_failure_fallback() -> None:
    class _BrokenSynthesizer(LegalAnswerSynthesizer):
        def _rank_relevant_chunks(self, context, query):  # type: ignore[override]
            raise RuntimeError("boom")

    broken = _BrokenSynthesizer()
    result = broken.generate([_parent("p-1", "X", "Y")], "Q")

    assert result.grounded is False
    assert result.sufficient_context is False
    assert result.citations == []
    assert any(warning.startswith("failure:") for warning in result.warnings)


def test_generate_answer_query_not_in_context_no_fabrication() -> None:
    context = [
        _CompressedLike(
            parent_chunk_id="p-c1",
            document_id="doc-7",
            source_name="doc-7.md",
            heading_text="Scope",
            compressed_text="The agreement defines service scope and acceptance criteria only.",
        )
    ]

    result = generate_answer(context, "What is the limitation of liability cap?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert "limitation of liability cap" not in result.answer_text.lower()


def test_agreement_intro_role_assignment_identifies_employer_and_employee() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made effective January 1, 2025, by and between Acme Corp, Inc. and Jane Smith.",
        )
    ]

    employer = generate_answer(context, "Who is the employer?")
    employee = generate_answer(context, "Who is the employee?")

    assert "acme corp" in employer.answer_text.lower()
    assert "jane smith" in employee.answer_text.lower()
    assert employer.sufficient_context is True
    assert employee.sufficient_context is True


def test_who_is_the_employer_returns_company_side_party_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Parties",
            'This Employment Agreement is between Acme Holdings LLC ("Employer") and Jane Smith ("Employee").',
        )
    ]

    result = generate_answer(context, "Who is the employer?")

    assert result.sufficient_context is True
    assert "acme holdings llc" in result.answer_text.lower()


def test_who_is_the_employee_returns_individual_side_party_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Parties",
            'This Employment Agreement is between Acme Holdings LLC ("Employer") and Jane Smith ("Employee").',
        )
    ]

    result = generate_answer(context, "Who is the employee?")

    assert result.sufficient_context is True
    assert "jane smith" in result.answer_text.lower()
    assert "acme holdings llc" not in result.answer_text.lower().split("direct answer:")[1].split("\n")[0]


def test_intro_pattern_between_and_with_explicit_role_labels_resolves_roles() -> None:
    context = [
        _parent(
            "p-1",
            "Parties",
            "BETWEEN:\nAcme Holdings LLC (the “Employer”)\nAND:\nJane Smith (the “Employee”)\n",
        )
    ]

    employer = generate_answer(context, "Who is the employer?")
    employee = generate_answer(context, "Who is the employee?")

    assert employer.sufficient_context is True
    assert employee.sufficient_context is True
    assert "acme holdings llc" in employer.answer_text.lower()
    assert "jane smith" in employee.answer_text.lower()
    assert employer.citations
    assert employee.citations


def test_role_labeled_intro_with_location_returns_party_name_not_location_and_keeps_citation_excerpt() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            (
                "This Employment Agreement is made by and between "
                "Aurora Data Systems Inc., Toronto, Ontario (the “Employer”) and "
                "Daniel Reza Mohammadi, Toronto, Ontario (the “Employee”)."
            ),
        )
    ]

    employer = generate_answer(context, "Who is the employer?")
    employee = generate_answer(context, "Who is the employee?")
    parties = generate_answer(context, "Who are the parties?")
    between = generate_answer(context, "Is this agreement between Aurora Data Systems Inc. and Daniel Reza Mohammadi?")

    employer_direct_line = employer.answer_text.splitlines()[0]
    employee_direct_line = employee.answer_text.splitlines()[0]
    parties_direct_line = parties.answer_text.splitlines()[0]

    assert employer.sufficient_context is True
    assert "Aurora Data Systems Inc." in employer.answer_text
    assert "Toronto, Ontario" not in employer_direct_line
    assert employer.citations
    assert "Aurora Data Systems Inc., Toronto, Ontario" in employer.citations[0].supporting_excerpt

    assert employee.sufficient_context is True
    assert "Daniel Reza Mohammadi" in employee.answer_text
    assert "Toronto, Ontario" not in employee_direct_line

    assert parties.sufficient_context is True
    assert "Aurora Data Systems Inc." in parties.answer_text
    assert "Daniel Reza Mohammadi" in parties.answer_text
    assert "Toronto, Ontario" not in parties_direct_line

    assert between.sufficient_context is True
    assert "yes" in between.answer_text.lower()


def test_intro_pattern_as_role_format_resolves_parties() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "Acme Holdings LLC as Employer and Jane Smith as Employee agree to the following terms.",
        )
    ]

    result = generate_answer(context, "Who are the parties?")

    assert result.sufficient_context is True
    assert "acme holdings llc" in result.answer_text.lower()
    assert "jane smith" in result.answer_text.lower()


def test_who_are_the_parties_returns_both_parties_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Who are the parties?")

    assert result.sufficient_context is True
    assert "acme corp" in result.answer_text.lower()
    assert "jane smith" in result.answer_text.lower()


def test_party_set_variants_return_both_parties_with_citations_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]
    for query in (
        "Who are the parties involved in this document?",
        "Identify the parties in this agreement",
    ):
        result = generate_answer(context, query)
        assert result.sufficient_context is True
        assert "acme corp" in result.answer_text.lower()
        assert "jane smith" in result.answer_text.lower()
        assert result.citations


def test_agreement_between_x_and_y_checks_both_extracted_parties() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    yes_result = generate_answer(context, "Is this agreement between Acme Corp and Jane Smith?")
    no_result = generate_answer(context, "Is this agreement between Acme Corp and John Roe?")

    assert "yes" in yes_result.answer_text.lower()
    assert "no" in no_result.answer_text.lower()


def test_agreement_between_query_returns_supported_verification_answer_when_both_parties_match() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Is this agreement with Jane Smith?")

    assert result.sufficient_context is True
    assert result.grounded is True
    assert "yes" in result.answer_text.lower()


def test_agreement_between_query_fails_safely_when_party_set_is_ambiguous_or_incomplete() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made between the Company and the Employee.",
        )
    ]

    result = generate_answer(context, "Is this agreement between Acme Corp and Jane Smith?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert any("party_role_assignment_unresolved" in warning for warning in result.warnings)


def test_ambiguous_or_incomplete_party_intro_fails_safely() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made between the Company and the Employee.",
        )
    ]

    result = generate_answer(context, "Who is the employee?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert any("party_role_assignment_unresolved" in warning for warning in result.warnings)


def test_employer_question_returns_company_side_party_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Who is the employer?")

    assert result.sufficient_context is True
    assert "acme corp" in result.answer_text.lower()


def test_employee_question_returns_individual_side_party_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Who is the employee?")

    assert result.sufficient_context is True
    assert "jane smith" in result.answer_text.lower()


def test_parties_question_returns_both_parties_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Who are the parties?")

    assert result.sufficient_context is True
    assert "acme corp" in result.answer_text.lower()
    assert "jane smith" in result.answer_text.lower()


def test_agreement_between_x_and_y_uses_extracted_party_set_when_supported() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Is this agreement between Acme Corp and Jane Smith?")

    assert result.sufficient_context is True
    assert "yes" in result.answer_text.lower()


def test_ambiguous_or_missing_role_resolution_still_fails_safely() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made between the Company and the Employee.",
        )
    ]

    result = generate_answer(context, "Who is the employer?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert any("party_role_assignment_unresolved" in warning for warning in result.warnings)


def test_hiring_company_and_company_side_variants_use_intro_party_resolution() -> None:
    context = [
        _parent(
            "p-1",
            "Parties",
            'This Employment Agreement is between Acme Holdings LLC ("Employer") and Jane Smith ("Employee").',
        )
    ]

    hiring_company = generate_answer(context, "Who is the hiring company?")
    company_side = generate_answer(context, "Which party is the company side?")
    individual_side = generate_answer(context, "Which party is the individual side?")

    assert hiring_company.sufficient_context is True
    assert company_side.sufficient_context is True
    assert individual_side.sufficient_context is True
    assert "acme holdings llc" in hiring_company.answer_text.lower()
    assert "acme holdings llc" in company_side.answer_text.lower()
    assert "jane smith" in individual_side.answer_text.lower()




def test_agreement_between_query_returns_supported_verification_answer_when_punctuation_differs() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Aurora Data Systems Inc and Daniel Reza Mohammadi.",
        )
    ]

    result = generate_answer(context, "Is this agreement between Aurora Data Systems Inc. and Daniel Reza Mohammadi?")

    assert result.sufficient_context is True
    assert result.grounded is True
    assert "yes" in result.answer_text.lower()


def test_agreement_between_query_fails_safely_when_query_entity_set_is_incomplete_or_ambiguous() -> None:
    context = [
        _parent(
            "p-1",
            "Introduction",
            "This Employment Agreement is made by and between Acme Corp and Jane Smith.",
        )
    ]

    result = generate_answer(context, "Is this agreement between Acme Corp and the employee?")

    assert result.sufficient_context is False
    assert result.grounded is False
    assert any("party_role_assignment_unresolved" in warning for warning in result.warnings)
def test_non_party_clause_lookup_behavior_remains_unchanged() -> None:
    context = [
        _parent(
            "p-1",
            "Confidentiality",
            (
                "Confidentiality obligations apply to all Confidential Information. "
                "The receiving party shall keep all Confidential Information strictly confidential."
            ),
        )
    ]

    result = generate_answer(context, "What does the document say about confidentiality?")

    assert result.grounded is True
    assert result.citations
    assert "confidential" in result.answer_text.lower()


def test_matter_metadata_file_number_answer_uses_metadata_responsive_evidence() -> None:
    context = [
        _parent(
            "p-1",
            "Matter Information",
            "File Number: CV-2025-0192",
            metadata={"file_number": "CV-2025-0192"},
        )
    ]

    result = generate_answer(context, "What is the file number?")

    assert result.sufficient_context is True
    assert "CV-2025-0192" in result.answer_text


def test_matter_metadata_question_does_not_use_unrelated_clause_text() -> None:
    context = [
        _parent(
            "p-1",
            "Termination",
            "Either party may terminate this Agreement with thirty days written notice.",
        )
    ]

    result = generate_answer(context, "What is the file number?")

    assert result.sufficient_context is False
    assert result.grounded is False
