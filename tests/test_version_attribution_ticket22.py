from __future__ import annotations

from agentic_rag.versioning import (
    MODEL_VERSION,
    UNKNOWN_MODEL_VERSION,
    get_version_attribution,
    normalize_model_version,
)


def test_central_version_source_exposes_all_required_identifiers() -> None:
    versions = get_version_attribution()
    assert versions == {
        "retrieval_version": "retrieval.v1",
        "answerability_version": "answerability.v1",
        "generation_version": "generation.v1",
        "prompt_bundle_version": "prompt_bundle.v1",
        "model_version": MODEL_VERSION,
    }


def test_model_version_normalization_uses_explicit_stable_fallback() -> None:
    assert normalize_model_version(None) == UNKNOWN_MODEL_VERSION
    assert normalize_model_version(" ") == UNKNOWN_MODEL_VERSION
    assert normalize_model_version("gpt-4.1-mini") == "gpt-4.1-mini"
