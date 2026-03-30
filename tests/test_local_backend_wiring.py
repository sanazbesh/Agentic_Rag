from __future__ import annotations

from pathlib import Path

from agentic_rag.orchestration.legal_rag_graph import run_legal_rag_turn
from ui.local_backend import build_local_backend_dependencies


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_local_backend_builds_from_uploaded_markdown(tmp_path: Path) -> None:
    doc_path = tmp_path / "msa.md"
    _write_text_file(
        doc_path,
        """# Service Agreement\n\n## Termination\nEither party may terminate with 30 days notice.\n""",
    )

    descriptor = {
        "id": "uploaded:msa.md",
        "name": "msa.md",
        "path": str(doc_path),
        "type": "md",
        "source": "uploaded",
    }

    build = build_local_backend_dependencies([descriptor])
    result = run_legal_rag_turn(
        query="termination notice",
        dependencies=build.dependencies,
    )

    assert build.scope_meta["loaded_document_count"] == 1
    assert build.scope_meta["parent_chunk_count"] >= 1
    assert "terminate" in result.answer_text.lower()


def test_local_backend_hybrid_filter_supports_selected_document_ids(tmp_path: Path) -> None:
    doc_a = tmp_path / "a.md"
    doc_b = tmp_path / "b.md"
    _write_text_file(doc_a, "# A\n\n## Rule\nContract A says notice is ten days.\n")
    _write_text_file(doc_b, "# B\n\n## Rule\nContract B says notice is ninety days.\n")

    descriptor_a = {
        "id": "uploaded:a.md",
        "name": "a.md",
        "path": str(doc_a),
        "type": "md",
        "source": "uploaded",
    }
    descriptor_b = {
        "id": "uploaded:b.md",
        "name": "b.md",
        "path": str(doc_b),
        "type": "md",
        "source": "uploaded",
    }

    build = build_local_backend_dependencies([descriptor_a, descriptor_b])
    hits = build.dependencies.retrieval.hybrid_search(
        "notice days",
        filters={"selected_document_ids": ["uploaded:a.md"]},
        top_k=10,
    )

    assert hits
    assert all(hit.document_id == "uploaded:a.md" for hit in hits)
