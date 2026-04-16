from __future__ import annotations

from agentic_rag.orchestration.query_understanding import understand_query
from agentic_rag.retrieval.parent_child import ParentChunkResult
from agentic_rag.tools.answerability import assess_answerability


def _parent(
    pid: str,
    heading: str,
    text: str,
    *,
    metadata: dict[str, str] | None = None,
) -> ParentChunkResult:
    return ParentChunkResult(
        parent_chunk_id=pid,
        document_id="doc-matter",
        text=text,
        source="test",
        source_name="matter_info.md",
        heading_path=("Matter Information", heading),
        heading_text=heading,
        metadata=metadata or {},
    )


def test_metadata_question_recognized_as_legal_matter_info_query() -> None:
    result = understand_query("What is the file number?")

    assert result.question_type == "extractive_fact_query"
    assert result.answerability_expectation == "fact_extraction"
    assert "legal_question_family:matter_document_metadata" in result.routing_notes


def test_file_number_question_uses_metadata_responsive_evidence_when_present() -> None:
    query = "What is the file number?"
    understanding = understand_query(query)
    context = [
        _parent("p-1", "Confidentiality", "Either party may terminate for breach."),
        _parent("p-2", "Matter Information", "File Number: CV-2025-0192"),
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert "matter_document_metadata_responsive_evidence_detected" in result.evidence_notes


def test_jurisdiction_question_uses_metadata_or_header_evidence_when_present() -> None:
    query = "What jurisdiction applies?"
    understanding = understand_query(query)
    context = [
        _parent("p-1", "Matter Information", "Jurisdiction: Ontario"),
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_court_question_uses_caption_or_header_evidence_when_present() -> None:
    query = "What court is involved?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-1",
            "Caption",
            "IN THE SUPREME COURT OF BRITISH COLUMBIA",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_client_question_uses_matter_info_or_intro_evidence_when_present() -> None:
    query = "Who is the client?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-1",
            "Matter Information",
            "Client: Acme Holdings Inc.",
            metadata={"client_name": "Acme Holdings Inc."},
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_case_name_question_uses_caption_or_matter_name_evidence_when_present() -> None:
    query = "What is the case name?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-1",
            "Caption",
            "Case Name: Acme Holdings Inc. v. Doe",
            metadata={"case_name": "Acme Holdings Inc. v. Doe"},
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is True
    assert result.should_answer is True


def test_missing_or_ambiguous_metadata_evidence_fails_safely() -> None:
    query = "What is the file number?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-1",
            "Termination",
            "Either party may terminate this Agreement with thirty days written notice.",
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert result.sufficient_context is False
    assert result.should_answer is False
    assert result.insufficiency_reason == "fact_not_found"


def test_non_metadata_clause_lookup_behavior_remains_unchanged() -> None:
    query = "What does the document say about confidentiality?"
    understanding = understand_query(query)
    context = [
        _parent(
            "p-1",
            "Confidentiality",
            (
                "The receiving party shall keep Confidential Information confidential and may disclose it only when "
                "required by law with prompt written notice where legally permitted."
            ),
        )
    ]

    result = assess_answerability(query, understanding, context)

    assert understanding.answerability_expectation == "clause_lookup"
    assert result.sufficient_context is True
