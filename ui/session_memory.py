"""Session-level conversation memory helpers for Streamlit + backend orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def summarize_conversation(history: Sequence[Mapping[str, Any]], *, max_turns: int = 6) -> str | None:
    """Build a compact deterministic summary from recent turns for rewrite/classification context."""

    recent_turns = list(history)[-max_turns:]
    if not recent_turns:
        return None
    lines: list[str] = []
    for turn in recent_turns:
        query = str(turn.get("query", "")).strip()
        answer = str(turn.get("answer_text", "")).strip()
        if query:
            lines.append(f"User: {query}")
        if answer:
            lines.append(f"Assistant: {answer[:240]}")
    if not lines:
        return None
    return "\n".join(lines)


def serialize_history_as_messages(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert stored turn history into backend-friendly recent_messages payload."""

    messages: list[dict[str, Any]] = []
    for turn in history:
        query = str(turn.get("query", "")).strip()
        answer = str(turn.get("answer_text", "")).strip()
        metadata = turn.get("metadata") if isinstance(turn.get("metadata"), Mapping) else {}
        if query:
            messages.append({"role": "user", "content": query})
        if answer:
            messages.append({"role": "assistant", "content": answer, "metadata": dict(metadata)})
    return messages


def append_conversation_turn(
    *,
    history: Sequence[Mapping[str, Any]],
    query: str,
    answer_text: str,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append turn immutably for safe session mutation and deterministic tests."""

    updated = [dict(item) for item in history]
    updated.append(
        {
            "query": query,
            "answer_text": answer_text,
            "metadata": dict(metadata or {}),
        }
    )
    return updated


def clear_conversation() -> list[dict[str, Any]]:
    """Return an empty conversation history payload for explicit reset flows."""

    return []
