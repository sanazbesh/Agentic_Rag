"""Central, stable version attribution for legal RAG traces and eval outputs."""

from __future__ import annotations

from typing import Any

RETRIEVAL_VERSION = "retrieval.v1"
ANSWERABILITY_VERSION = "answerability.v1"
GENERATION_VERSION = "generation.v1"
PROMPT_BUNDLE_VERSION = "prompt_bundle.v1"
MODEL_VERSION = "local_deterministic_legal_rag_model.v1"
UNKNOWN_MODEL_VERSION = "unknown_model_version"


VersionAttribution = dict[str, str]


def normalize_model_version(value: Any, *, fallback: str = UNKNOWN_MODEL_VERSION) -> str:
    """Return a stable model version string with explicit fallback behavior."""

    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return fallback


def get_version_attribution(*, model_version: Any = MODEL_VERSION) -> VersionAttribution:
    """Return the canonical version attribution payload for one request/run."""

    return {
        "retrieval_version": RETRIEVAL_VERSION,
        "answerability_version": ANSWERABILITY_VERSION,
        "generation_version": GENERATION_VERSION,
        "prompt_bundle_version": PROMPT_BUNDLE_VERSION,
        "model_version": normalize_model_version(model_version),
    }
