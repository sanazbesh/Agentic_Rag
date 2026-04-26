from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

_OFFLINE_FIXTURE_REGISTRY: dict[str, dict[str, str]] = {
    "doc-employment-master": {
        "name": "Employment Agreement (Master)",
        "relative_path": "evals/fixtures/offline_documents/doc-employment-master.md",
    },
    "doc-chronology": {
        "name": "Chronology Summary",
        "relative_path": "evals/fixtures/offline_documents/doc-chronology.md",
    },
    "doc-employment-amendment": {
        "name": "Employment Agreement Amendment",
        "relative_path": "evals/fixtures/offline_documents/doc-employment-amendment.md",
    },
    "doc-benefits-booklet": {
        "name": "Benefits Booklet",
        "relative_path": "evals/fixtures/offline_documents/doc-benefits-booklet.md",
    },
    "doc-onboarding-checklist": {
        "name": "Onboarding Checklist",
        "relative_path": "evals/fixtures/offline_documents/doc-onboarding-checklist.md",
    },
    "doc-employment": {
        "name": "Employment Agreement (Short)",
        "relative_path": "evals/fixtures/offline_documents/doc-employment.md",
    },
    "doc-employee-handbook": {
        "name": "Employee Handbook",
        "relative_path": "evals/fixtures/offline_documents/doc-employee-handbook.md",
    },
    "doc-termination-letter": {
        "name": "Termination Letter",
        "relative_path": "evals/fixtures/offline_documents/doc-termination-letter.md",
    },
    "doc-litigation-letter": {
        "name": "Litigation Letter",
        "relative_path": "evals/fixtures/offline_documents/doc-litigation-letter.md",
    },
}


def resolve_offline_eval_selected_documents(
    eval_case: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve dataset selected document IDs into local descriptors consumable by local backend."""

    selected_documents_raw = eval_case.get("selected_documents")
    selected_documents: list[dict[str, Any]] = []
    if isinstance(selected_documents_raw, list):
        for item in selected_documents_raw:
            if isinstance(item, Mapping):
                selected_documents.append(dict(item))

    selected_ids = [str(item) for item in list(eval_case.get("selected_document_ids") or []) if str(item).strip()]
    selected_paths = [str(item) for item in list(eval_case.get("selected_document_paths") or []) if str(item).strip()]

    existing_by_id = {str(doc.get("id")): doc for doc in selected_documents if doc.get("id")}
    existing_paths = {str(doc.get("path")) for doc in selected_documents if doc.get("path")}

    root = repo_root or Path(__file__).resolve().parents[2]
    for index, selected_id in enumerate(selected_ids):
        if selected_id in existing_by_id:
            continue

        descriptor: dict[str, Any] = {"id": selected_id}
        if index < len(selected_paths):
            descriptor["path"] = selected_paths[index]
        else:
            fixture_meta = _OFFLINE_FIXTURE_REGISTRY.get(selected_id)
            if fixture_meta is None:
                raise ValueError(f"offline_fixture_not_found:{selected_id}")

            fixture_path = root / fixture_meta["relative_path"]
            if not fixture_path.is_file():
                raise FileNotFoundError(f"offline_fixture_not_found:{selected_id}:{fixture_path}")

            descriptor.update(
                {
                    "path": str(fixture_path),
                    "name": fixture_meta["name"],
                    "type": fixture_path.suffix.lstrip(".") or "md",
                    "source": "offline_fixture",
                }
            )

        if descriptor.get("path"):
            existing_paths.add(str(descriptor["path"]))
        selected_documents.append(descriptor)

    for selected_path in selected_paths:
        if selected_path in existing_paths:
            continue
        selected_documents.append({"path": selected_path})

    return selected_documents


def known_offline_fixture_ids() -> set[str]:
    return set(_OFFLINE_FIXTURE_REGISTRY)
