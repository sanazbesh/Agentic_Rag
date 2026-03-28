from __future__ import annotations

from agentic_rag.chunking.models import ChildChunk
from agentic_rag.indexing import BM25Index, LegalSparseTokenizer
from agentic_rag.retrieval import SparseSearchService, search_child_chunks_sparse


def _chunks() -> list[dict[str, object]]:
    return [
        {
            "child_chunk_id": "child-1",
            "parent_chunk_id": "parent-1",
            "document_id": "doc-1",
            "text": "UCC § 2-207 governs additional terms in acceptance.",
            "metadata": {
                "jurisdiction": "NY",
                "document_type": "statute",
                "court": "N/A",
                "date": "2024-01-01",
            },
        },
        {
            "child_chunk_id": "child-2",
            "parent_chunk_id": "parent-2",
            "document_id": "doc-2",
            "text": "Rule 12(b)(6) addresses dismissal for failure to state a claim.",
            "metadata": {
                "jurisdiction": "CA",
                "document_type": "opinion",
                "court": "Court of Appeal",
                "date": "2023-06-10",
            },
        },
        {
            "child_chunk_id": "child-3",
            "parent_chunk_id": "parent-3",
            "document_id": "doc-3",
            "text": "Article 2 of the UCC applies to sale of goods.",
            "metadata": {
                "jurisdiction": "NY",
                "document_type": "treatise",
                "court": "N/A",
                "date": "2022-01-05",
            },
        },
    ]


def test_basic_indexing_and_retrieval() -> None:
    index = BM25Index()
    result = index.index_child_chunks(_chunks())
    service = SparseSearchService(index)

    hits = service.search_child_chunks_sparse("ucc additional terms")

    assert result.total_chunks_indexed == 3
    assert not result.failed_chunk_ids
    assert not result.skipped_chunk_ids
    assert hits
    assert hits[0].child_chunk_id == "child-1"
    assert hits[0].parent_chunk_id == "parent-1"
    assert hits[0].document_id == "doc-1"
    assert isinstance(hits[0].sparse_score, float)


def test_deterministic_tokenization() -> None:
    tokenizer = LegalSparseTokenizer()
    assert tokenizer.tokenize(" Rule   12(b)(6)  ") == tokenizer.tokenize("rule 12(b)(6)")


def test_correct_ranking_for_keyword_matches() -> None:
    index = BM25Index()
    index.index_child_chunks(_chunks())
    service = SparseSearchService(index)

    hits = service.search_child_chunks_sparse("ucc ucc article")

    assert [item.child_chunk_id for item in hits[:2]] == ["child-3", "child-1"]


def test_filter_correctness_post_lexical_matching() -> None:
    index = BM25Index()
    index.index_child_chunks(_chunks())
    service = SparseSearchService(index)

    hits = service.search_child_chunks_sparse("ucc", filters={"jurisdiction": "NY", "document_type": "statute"})

    assert [item.child_chunk_id for item in hits] == ["child-1"]


def test_empty_input_handling() -> None:
    index = BM25Index()

    result = index.index_child_chunks([])

    assert result.total_chunks_indexed == 0
    assert result.failed_chunk_ids == []
    assert result.skipped_chunk_ids == []


def test_empty_query_result() -> None:
    index = BM25Index()
    index.index_child_chunks(_chunks())
    service = SparseSearchService(index)

    assert service.search_child_chunks_sparse("   ") == []
    assert service.search_child_chunks_sparse("nonexistent latin maxim") == []


def test_duplicate_id_behavior_last_write_wins() -> None:
    index = BM25Index()
    index.index_child_chunks(_chunks())

    update = {
        "child_chunk_id": "child-1",
        "parent_chunk_id": "parent-1b",
        "document_id": "doc-1b",
        "text": "Updated text referencing Article 2 and UCC.",
        "metadata": {"jurisdiction": "TX", "document_type": "opinion", "court": "Supreme Court", "date": "2025-01-01"},
    }
    index.index_child_chunks([update])
    service = SparseSearchService(index)
    hits = service.search_child_chunks_sparse("updated article")

    assert hits and hits[0].child_chunk_id == "child-1"
    assert hits[0].parent_chunk_id == "parent-1b"
    assert hits[0].document_id == "doc-1b"


def test_idempotent_indexing() -> None:
    index = BM25Index()
    chunks = _chunks()
    index.index_child_chunks(chunks)
    first = SparseSearchService(index).search_child_chunks_sparse("ucc", top_k=5)

    index.index_child_chunks(chunks)
    second = SparseSearchService(index).search_child_chunks_sparse("ucc", top_k=5)

    assert first == second


def test_edge_cases_with_legal_tokens_preserved() -> None:
    index = BM25Index()
    index.index_child_chunks(_chunks())

    direct_hits = search_child_chunks_sparse("§ 2-207", index, top_k=5)
    rule_hits = search_child_chunks_sparse("12(b)(6)", index, top_k=5)

    assert direct_hits and direct_hits[0].child_chunk_id == "child-1"
    assert rule_hits and rule_hits[0].child_chunk_id == "child-2"


def test_partial_failure_and_skips_handling() -> None:
    index = BM25Index()
    bad_inputs: list[dict[str, object]] = [
        {
            "child_chunk_id": "",
            "parent_chunk_id": "parent-x",
            "document_id": "doc-x",
            "text": "valid text",
            "metadata": {},
        },
        {
            "child_chunk_id": "child-whitespace",
            "parent_chunk_id": "parent-y",
            "document_id": "doc-y",
            "text": "   ",
            "metadata": {},
        },
        {
            "child_chunk_id": "child-ok",
            "parent_chunk_id": "parent-z",
            "document_id": "doc-z",
            "text": "Article 2 survives indexing.",
            "metadata": {},
        },
    ]

    result = index.index_child_chunks(bad_inputs)
    hits = SparseSearchService(index).search_child_chunks_sparse("article 2")

    assert result.total_chunks_indexed == 1
    assert "child-whitespace" in result.skipped_chunk_ids
    assert hits and hits[0].child_chunk_id == "child-ok"


def test_top_k_and_deterministic_tiebreak() -> None:
    index = BM25Index()
    equal_score_chunks = [
        ChildChunk(
            child_chunk_id="a-child",
            parent_chunk_id="p1",
            document_id="d1",
            source="s",
            source_name="n",
            text="contract breach",
            child_order=0,
            token_count=2,
        ),
        ChildChunk(
            child_chunk_id="b-child",
            parent_chunk_id="p2",
            document_id="d2",
            source="s",
            source_name="n",
            text="contract breach",
            child_order=1,
            token_count=2,
        ),
    ]
    index.index_child_chunks(equal_score_chunks)

    hits = SparseSearchService(index).search_child_chunks_sparse("contract breach", top_k=1)

    assert len(hits) == 1
    assert hits[0].child_chunk_id == "a-child"
