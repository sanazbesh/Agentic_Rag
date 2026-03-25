from __future__ import annotations

from agentic_rag.retrieval import (
    ChunkReranker,
    ChildChunkSearcher,
    HybridSearchResult,
    InMemoryChildChunkRepository,
    InMemoryKeywordChunkRepository,
    InMemoryParentChunkRepository,
    KeywordSearchService,
    ParentChildRetrievalTools,
    ParentChunkStore,
    RerankedChunkResult,
)


def _toolkit() -> ParentChildRetrievalTools:
    child_records = [
        {
            "id": "child-1",
            "text": "Breach of contract elements include duty and damages.",
            "payload": {
                "parent_chunk_id": "parent-1",
                "document_id": "doc-1",
                "jurisdiction": "NY",
                "court": "Supreme Court",
                "document_type": "opinion",
                "date": "2024-01-15",
                "text": "Breach of contract elements include duty and damages.",
            },
        },
        {
            "id": "child-2",
            "text": "Negligence requires duty breach causation and damages.",
            "payload": {
                "parent_chunk_id": "parent-2",
                "document_id": "doc-2",
                "jurisdiction": "CA",
                "court": "Court of Appeal",
                "document_type": "opinion",
                "date": "2023-09-01",
                "text": "Negligence requires duty breach causation and damages.",
            },
        },
    ]

    parent_lookup = {
        "parent-1": {
            "parent_chunk_id": "parent-1",
            "document_id": "doc-1",
            "text": "Full parent context for contract analysis.",
            "source": "s3://bucket/doc1.md",
            "source_name": "doc1.md",
            "heading_path": ["Contracts", "Elements"],
            "heading_text": "Elements",
            "parent_order": 2,
            "part_number": 1,
            "total_parts": 1,
        },
        "parent-2": {
            "parent_chunk_id": "parent-2",
            "document_id": "doc-2",
            "text": "Full parent context for negligence analysis.",
            "source": "s3://bucket/doc2.md",
            "source_name": "doc2.md",
            "heading_path": ["Torts", "Negligence"],
            "heading_text": "Negligence",
            "parent_order": 1,
            "part_number": 1,
            "total_parts": 1,
        },
    }

    return ParentChildRetrievalTools(
        child_searcher=ChildChunkSearcher(repository=InMemoryChildChunkRepository(child_records), default_limit=5),
        parent_store=ParentChunkStore(repository=InMemoryParentChunkRepository(parent_lookup)),
        keyword_search_service=KeywordSearchService(
            repository=InMemoryKeywordChunkRepository(child_records),
            default_limit=5,
        ),
        chunk_reranker=ChunkReranker(),
    )


def test_search_child_chunks_returns_structured_results() -> None:
    toolkit = _toolkit()

    results = toolkit.search_child_chunks("breach damages")

    assert results
    first = results[0]
    assert first.child_chunk_id
    assert first.parent_chunk_id
    assert first.document_id
    assert first.text
    assert isinstance(first.score, float)
    assert isinstance(first.payload, dict)


def test_search_child_chunks_includes_parent_chunk_id() -> None:
    toolkit = _toolkit()

    results = toolkit.search_child_chunks("contract")

    assert [item.parent_chunk_id for item in results] == ["parent-1"]


def test_search_child_chunks_handles_filters_none() -> None:
    toolkit = _toolkit()

    no_filters = toolkit.search_child_chunks("duty")
    explicit_none = toolkit.search_child_chunks("duty", filters=None)

    assert no_filters == explicit_none


def test_search_child_chunks_handles_empty_and_no_results_safely() -> None:
    toolkit = _toolkit()

    assert toolkit.search_child_chunks("") == []
    assert toolkit.search_child_chunks("   ") == []
    assert toolkit.search_child_chunks("res ipsa loquitur") == []


def test_retrieve_parent_chunks_fetches_valid_parent_chunks() -> None:
    toolkit = _toolkit()

    parents = toolkit.retrieve_parent_chunks(["parent-1"])

    assert len(parents) == 1
    parent = parents[0]
    assert parent.parent_chunk_id == "parent-1"
    assert parent.document_id == "doc-1"
    assert parent.source == "s3://bucket/doc1.md"
    assert parent.source_name == "doc1.md"
    assert parent.heading_path == ("Contracts", "Elements")
    assert parent.parent_order == 2


def test_retrieve_parent_chunks_handles_missing_ids_safely() -> None:
    toolkit = _toolkit()

    parents = toolkit.retrieve_parent_chunks(["missing", "parent-1"])

    assert [parent.parent_chunk_id for parent in parents] == ["parent-1"]


def test_retrieve_parent_chunks_preserves_order_and_deduplicates_ids() -> None:
    toolkit = _toolkit()

    parents = toolkit.retrieve_parent_chunks(["parent-2", "parent-1", "parent-2"])

    assert [parent.parent_chunk_id for parent in parents] == ["parent-2", "parent-1"]


def test_tools_are_deterministic_for_same_inputs() -> None:
    toolkit = _toolkit()

    first_search = toolkit.search_child_chunks("duty damages")
    second_search = toolkit.search_child_chunks("duty damages")
    assert first_search == second_search

    first_parent_fetch = toolkit.retrieve_parent_chunks(["parent-2", "parent-1"])
    second_parent_fetch = toolkit.retrieve_parent_chunks(["parent-2", "parent-1"])
    assert first_parent_fetch == second_parent_fetch


def test_search_child_chunks_supports_extensible_legal_filters() -> None:
    toolkit = _toolkit()

    results = toolkit.search_child_chunks("duty", filters={"jurisdiction": "NY", "court": "Supreme Court"})

    assert [item.child_chunk_id for item in results] == ["child-1"]


def test_hybrid_search_returns_structured_results() -> None:
    toolkit = _toolkit()

    results = toolkit.hybrid_search("breach duty damages")

    assert results
    first = results[0]
    assert isinstance(first, HybridSearchResult)
    assert first.child_chunk_id
    assert first.parent_chunk_id
    assert first.document_id
    assert isinstance(first.combined_score, float)
    assert isinstance(first.vector_score, float)
    assert isinstance(first.keyword_score, float)
    assert isinstance(first.payload, dict)


def test_hybrid_search_combines_vector_and_keyword_results() -> None:
    toolkit = _toolkit()

    results = toolkit.hybrid_search("contract")

    assert results
    top = results[0]
    assert top.vector_score > 0.0
    assert top.keyword_score > 0.0
    assert top.combined_score > 0.0


def test_hybrid_search_deduplicates_overlapping_results() -> None:
    toolkit = _toolkit()

    results = toolkit.hybrid_search("duty damages")

    ids = [item.child_chunk_id for item in results]
    assert len(ids) == len(set(ids))


def test_hybrid_search_respects_filters_none() -> None:
    toolkit = _toolkit()

    no_filters = toolkit.hybrid_search("duty")
    explicit_none = toolkit.hybrid_search("duty", filters=None)

    assert no_filters == explicit_none


def test_hybrid_search_handles_empty_and_no_result_cases() -> None:
    toolkit = _toolkit()

    assert toolkit.hybrid_search("") == []
    assert toolkit.hybrid_search("   ") == []
    assert toolkit.hybrid_search("res ipsa loquitur") == []


def test_rerank_chunks_returns_sorted_results() -> None:
    toolkit = _toolkit()

    chunks = toolkit.hybrid_search("breach duty damages")
    reranked = toolkit.rerank_chunks(chunks, "breach duty damages")

    assert reranked
    assert isinstance(reranked[0], RerankedChunkResult)
    assert reranked == sorted(
        reranked,
        key=lambda item: (-item.rerank_score, -item.original_score, item.child_chunk_id),
    )


def test_rerank_chunks_preserves_metadata_and_parent_linkage() -> None:
    toolkit = _toolkit()

    chunks = toolkit.hybrid_search("breach contract")
    reranked = toolkit.rerank_chunks(chunks, "breach contract")

    assert reranked
    first = reranked[0]
    assert first.parent_chunk_id
    assert first.child_chunk_id
    assert first.document_id
    assert isinstance(first.payload, dict)


def test_rerank_chunks_assigns_scores_and_is_deterministic() -> None:
    toolkit = _toolkit()
    chunks = toolkit.hybrid_search("duty damages")

    first = toolkit.rerank_chunks(chunks, "duty damages")
    second = toolkit.rerank_chunks(chunks, "duty damages")

    assert all(item.rerank_score >= 0.0 for item in first)
    assert all(item.original_score >= 0.0 for item in first)
    assert first == second


def test_hybrid_then_rerank_integration_preserves_parent_chunk_ids() -> None:
    toolkit = _toolkit()

    hybrid_results = toolkit.hybrid_search("negligence causation duty")
    reranked = toolkit.rerank_chunks(hybrid_results, "negligence causation duty")

    assert hybrid_results
    assert reranked
    assert all(item.parent_chunk_id for item in reranked)
