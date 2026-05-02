"""Utilities for normalizing lifecycle status values returned by persistence layers."""

from __future__ import annotations

from typing import Any


def normalize_status(value: Any) -> str | None:
    """Return a plain status string for enum-like, string, or null values."""

    if value is None:
        return None

    normalized = getattr(value, "value", value)
    if normalized is None:
        return None
    if isinstance(normalized, str):
        return normalized
    return str(normalized)
