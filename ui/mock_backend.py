"""Mock backend responses for the Streamlit legal RAG inspection UI.

These responses mirror the strict final answer contract expected from the real
backend runner and include a stable, optional debug payload.
"""

from __future__ import annotations

from typing import Any


MOCK_DOCUMENTS: list[dict[str, str]] = [
    {"id": "doc-nda-001", "name": "Mutual NDA (2024)"},
    {"id": "doc-msa-017", "name": "Master Services Agreement"},
    {"id": "doc-employment-003", "name": "Employment Agreement"},
    {"id": "doc-dpa-009", "name": "Data Processing Addendum"},
]


def get_mock_documents() -> list[dict[str, str]]:
    """Return mock documents for sidebar selection."""

    return MOCK_DOCUMENTS


def run_mock_backend_query(
    *,
    query: str,
    conversation_summary: str | None,
    recent_messages: list[dict[str, Any]] | None,
    selected_documents: list[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a mock final result and debug payload using the real schema."""

    normalized_query = query.strip().lower()
    insufficient = any(token in normalized_query for token in ["not enough", "insufficient", "unknown"])

    if insufficient:
        final_result = {
            "answer_text": (
                "I cannot answer this reliably with the retrieved legal context. "
                "Please provide additional governing documents or narrower scope."
            ),
            "grounded": False,
            "sufficient_context": False,
            "citations": [],
            "warnings": [
                "insufficient_context:mock",
                "No supporting parent chunks met grounding threshold.",
            ],
        }
    else:
        final_result = {
            "answer_text": (
                "The MSA appears to cap liability at fees paid in the prior 12 months, "
                "with carve-outs for confidentiality and IP infringement."
            ),
            "grounded": True,
            "sufficient_context": True,
            "citations": [
                {
                    "parent_chunk_id": "parent_45",
                    "document_id": "doc-msa-017",
                    "source_name": "Master Services Agreement",
                    "heading": "Section 12 - Limitation of Liability",
                    "supporting_excerpt": (
                        "...aggregate liability shall not exceed fees paid in the "
                        "twelve (12) months preceding the claim..."
                    ),
                },
                {
                    "parent_chunk_id": "parent_51",
                    "document_id": "doc-msa-017",
                    "source_name": "Master Services Agreement",
                    "heading": "Section 9 - Confidentiality",
                    "supporting_excerpt": "...limitations do not apply to breaches of confidentiality obligations...",
                },
            ],
            "warnings": [],
        }

    debug_payload: dict[str, Any] = {
        "rewritten_query": "Summarize liability cap and carve-outs in selected legal agreements.",
        "extracted_entities": ["liability cap", "carve-outs", "MSA", "confidentiality"],
        "filters": {
            "selected_documents": selected_documents or [],
            "jurisdiction": "US",
            "document_type": ["contract"],
        },
        "hybrid_search_results": [
            {"child_chunk_id": "child_183", "score": 0.91, "document_id": "doc-msa-017"},
            {"child_chunk_id": "child_311", "score": 0.83, "document_id": "doc-msa-017"},
        ],
        "reranked_child_results": [
            {"child_chunk_id": "child_183", "rerank_score": 0.97},
            {"child_chunk_id": "child_311", "rerank_score": 0.76},
        ],
        "parent_chunks": [
            {
                "parent_chunk_id": "parent_45",
                "heading": "Section 12 - Limitation of Liability",
                "document_id": "doc-msa-017",
            },
            {
                "parent_chunk_id": "parent_51",
                "heading": "Section 9 - Confidentiality",
                "document_id": "doc-msa-017",
            },
        ],
        "compressed_context": [
            "Liability is capped at fees paid in prior 12 months.",
            "Cap does not apply to confidentiality and IP carve-outs.",
        ],
        "warnings": list(final_result["warnings"]),
        "meta": {
            "mode": "mock",
            "recent_messages_count": len(recent_messages or []),
            "conversation_summary_present": bool(conversation_summary),
        },
    }

    return final_result, debug_payload
