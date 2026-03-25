from __future__ import annotations

from agentic_rag.chunking import MarkdownParentChildChunker
from agentic_rag.ingestion import MarkdownDocumentIngestor
from agentic_rag.retrieval import (
    ChildChunkSearcher,
    InMemoryChildChunkRepository,
    InMemoryParentChunkRepository,
    ParentChildRetrievalTools,
    ParentChunkStore,
)
from agentic_rag.tools import rewrite_query


LEGAL_FIXTURE_MARKDOWN = """# Master Services Agreement

## 1. Termination
Either party may terminate this Agreement for material breach if the breaching party
fails to cure within thirty (30) days after written notice.

## 2. Confidentiality
Each party must keep Confidential Information strictly confidential and may use it
only to perform obligations under this Agreement.

## 3. Governing Law
This Agreement is governed by the laws of the State of New York, without regard
to conflict-of-laws principles.
"""


def test_end_to_end_legal_retrieval_flow() -> None:
    # 1-2) Ingest deterministic markdown legal fixture through current ingestion path.
    ingested_docs = MarkdownDocumentIngestor().ingest(
        [
            {
                "source": "fixtures/legal_msa.md",
                "source_name": "legal_msa.md",
                "text": LEGAL_FIXTURE_MARKDOWN,
            }
        ]
    )
    assert len(ingested_docs) == 1
    document = ingested_docs[0]

    # 3) Run parent-child chunking over the ingested document.
    chunking_result = MarkdownParentChildChunker().chunk(document)
    assert chunking_result.child_chunks
    assert chunking_result.parent_chunks

    # 4) Prepare retrieval backends via repository-native in-memory abstractions.
    toolkit = ParentChildRetrievalTools(
        child_searcher=ChildChunkSearcher(
            repository=InMemoryChildChunkRepository(chunking_result.child_qdrant_records()),
            default_limit=5,
        ),
        parent_store=ParentChunkStore(
            repository=InMemoryParentChunkRepository(chunking_result.parent_lookup()),
        ),
    )

    # 5) Rewrite realistic user query before retrieval.
    rewrite = rewrite_query("What does the agreement say about termination breach")

    # 6-7) Search child chunks using rewritten query.
    child_hits = toolkit.search_child_chunks(rewrite.rewritten_query)

    # 8-9) Collect parent ids from child hits and retrieve parent context.
    parent_ids = [hit.parent_chunk_id for hit in child_hits]
    parent_hits = toolkit.retrieve_parent_chunks(parent_ids)

    # 10) Assert the expected parent/legal section is included in returned context.
    assert rewrite.rewritten_query
    assert child_hits
    assert all(hit.parent_chunk_id for hit in child_hits)
    assert parent_hits

    assert any("material breach" in parent.text.lower() for parent in parent_hits)
    assert any("termination" in parent.text.lower() for parent in parent_hits)
    assert all(parent.document_id == document.id for parent in parent_hits)
