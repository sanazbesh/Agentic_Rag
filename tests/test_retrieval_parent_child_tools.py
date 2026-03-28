from __future__ import annotations

from agentic_rag.retrieval import (
    ChunkReranker,
    DenseChildSearchService,
    ChildChunkSearcher,
    HybridSearchResult,
    HybridSearchService,
    InMemoryChildChunkRepository,
    InMemoryKeywordChunkRepository,
    InMemoryParentChunkRepository,
    KeywordChunkRepository,
    KeywordSearchService,
    ParentChildRetrievalTools,
    ParentChunkStore,
    RRFFuser,
    RerankedChunkResult,
    SparseChildSearchService,
    VectorSearchService,
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
    assert isinstance(first.hybrid_score, float)
    assert isinstance(first.matched_in_dense, bool)
    assert isinstance(first.matched_in_sparse, bool)
    assert (first.dense_score is None) or isinstance(first.dense_score, float)
    assert (first.sparse_score is None) or isinstance(first.sparse_score, float)
    assert isinstance(first.payload, dict)


def test_hybrid_search_combines_dense_and_sparse_results() -> None:
    toolkit = _toolkit()

    results = toolkit.hybrid_search("contract")

    assert results
    top = results[0]
    assert top.dense_score is not None and top.dense_score > 0.0
    assert top.sparse_score is not None and top.sparse_score > 0.0
    assert top.hybrid_score > 0.0


def test_hybrid_search_rrf_scoring_correctness() -> None:
    toolkit = _toolkit()

    results = toolkit.hybrid_search("duty damages", top_k=10)

    by_id = {item.child_chunk_id: item for item in results}
    assert "child-1" in by_id and "child-2" in by_id

    expected_child_1 = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    expected_child_2 = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
    assert abs(by_id["child-1"].hybrid_score - expected_child_1) < 1e-12
    assert abs(by_id["child-2"].hybrid_score - expected_child_2) < 1e-12


def test_hybrid_search_deduplicates_overlapping_results() -> None:
    toolkit = _toolkit()

    results = toolkit.hybrid_search("duty damages")

    ids = [item.child_chunk_id for item in results]
    assert len(ids) == len(set(ids))


def test_hybrid_search_deduplicates_duplicates_within_source_list() -> None:
    records = [
        {
            "id": "dup",
            "text": "Duty appears once.",
            "payload": {"parent_chunk_id": "p1", "document_id": "d1", "jurisdiction": "NY"},
        },
        {
            "id": "dup",
            "text": "Duty appears duplicated.",
            "payload": {"parent_chunk_id": "p1b", "document_id": "d1b", "jurisdiction": "NY"},
        },
    ]
    service = HybridSearchService(
        dense_service=DenseChildSearchService(
            vector_service=VectorSearchService(
                child_searcher=ChildChunkSearcher(repository=InMemoryChildChunkRepository(records), default_limit=10)
            )
        ),
        sparse_service=SparseChildSearchService(
            keyword_service=KeywordSearchService(
                repository=InMemoryKeywordChunkRepository(records),
                default_limit=10,
            )
        ),
        fuser=RRFFuser(rrf_k=60),
    )
    results = service.search("duty", top_k=10)
    assert [item.child_chunk_id for item in results] == ["dup"]
    assert results[0].parent_chunk_id == "p1"


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


def test_hybrid_search_dense_only_fallback_when_sparse_fails() -> None:
    class FailingKeywordRepository(KeywordChunkRepository):
        def search_keyword(self, query: str, *, filters=None, limit: int = 10):  # type: ignore[override]
            raise RuntimeError("sparse unavailable")

    records = [
        {"id": "child-1", "text": "duty damages", "payload": {"parent_chunk_id": "p1", "document_id": "d1"}}
    ]
    service = HybridSearchService(
        dense_service=DenseChildSearchService(
            vector_service=VectorSearchService(
                child_searcher=ChildChunkSearcher(repository=InMemoryChildChunkRepository(records), default_limit=5)
            )
        ),
        sparse_service=SparseChildSearchService(
            keyword_service=KeywordSearchService(repository=FailingKeywordRepository(), default_limit=5)
        ),
    )
    results = service.search("duty", top_k=5)
    assert len(results) == 1
    assert results[0].matched_in_dense is True
    assert results[0].matched_in_sparse is False


def test_hybrid_search_sparse_only_fallback_when_dense_fails() -> None:
    class FailingChildRepository(InMemoryChildChunkRepository):
        def search(self, query: str, *, filters=None, limit: int = 10):  # type: ignore[override]
            raise RuntimeError("dense unavailable")

    records = [
        {"id": "child-2", "text": "negligence duty", "payload": {"parent_chunk_id": "p2", "document_id": "d2"}}
    ]
    service = HybridSearchService(
        dense_service=DenseChildSearchService(
            vector_service=VectorSearchService(
                child_searcher=ChildChunkSearcher(repository=FailingChildRepository(records), default_limit=5)
            )
        ),
        sparse_service=SparseChildSearchService(
            keyword_service=KeywordSearchService(
                repository=InMemoryKeywordChunkRepository(records),
                default_limit=5,
            )
        ),
    )
    results = service.search("duty", top_k=5)
    assert len(results) == 1
    assert results[0].matched_in_dense is False
    assert results[0].matched_in_sparse is True


def test_hybrid_search_both_empty_returns_empty() -> None:
    toolkit = _toolkit()
    assert toolkit.hybrid_search("no overlap phrase", top_k=5) == []


def test_hybrid_search_deterministic_tie_break() -> None:
    records = [
        {"id": "a", "text": "alpha", "payload": {"parent_chunk_id": "p-a", "document_id": "d-a"}},
        {"id": "b", "text": "beta", "payload": {"parent_chunk_id": "p-b", "document_id": "d-b"}},
    ]
    service = HybridSearchService(
        dense_service=DenseChildSearchService(
            vector_service=VectorSearchService(
                child_searcher=ChildChunkSearcher(repository=InMemoryChildChunkRepository(records), default_limit=2)
            )
        ),
        sparse_service=SparseChildSearchService(
            keyword_service=KeywordSearchService(repository=InMemoryKeywordChunkRepository(records), default_limit=2)
        ),
    )
    # query matches neither term; each source returns [] so force tie via direct fuser impossible here.
    # Use direct fuser inputs instead for deterministic tie by child id.
    dense_hits = [
        service.dense_service.search("alpha", limit=2),
        service.dense_service.search("beta", limit=2),
    ]
    tied = RRFFuser(rrf_k=60).fuse(
        dense_results=[dense_hits[0][0]],
        sparse_results=[dense_hits[1][0]],
        top_k=10,
    )
    assert [item.child_chunk_id for item in tied] == ["a", "b"]


def test_hybrid_search_filter_behavior() -> None:
    toolkit = _toolkit()
    results = toolkit.hybrid_search("duty", filters={"jurisdiction": "NY"}, top_k=10)
    assert [item.child_chunk_id for item in results] == ["child-1"]


def test_hybrid_search_top_k_enforced() -> None:
    toolkit = _toolkit()
    results = toolkit.hybrid_search("duty damages", top_k=1)
    assert len(results) == 1


def test_hybrid_search_preserves_metadata_for_parent_lookup() -> None:
    toolkit = _toolkit()
    results = toolkit.hybrid_search("breach duty damages", top_k=10)
    assert results
    first = results[0]
    assert first.parent_chunk_id
    assert first.document_id
    assert first.text
    assert "jurisdiction" in first.metadata


def test_hybrid_search_non_llm_deterministic_infrastructure() -> None:
    toolkit = _toolkit()
    first = toolkit.hybrid_search("  duty damages  ", top_k=10)
    second = toolkit.hybrid_search("duty damages", top_k=10)
    assert first == second


def test_hybrid_search_returns_empty_for_non_positive_top_k() -> None:
    toolkit = _toolkit()
    assert toolkit.hybrid_search("duty", top_k=0) == []


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
