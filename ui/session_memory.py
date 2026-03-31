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


def build_backend_context(
    *,
    history: Sequence[Mapping[str, Any]],
    conversation_summary_input: str | None,
    recent_messages_override: list[dict[str, Any]] | None,
) -> tuple[str | None, list[dict[str, Any]], bool]:
    """Build backend context from session history unless override is explicitly provided.

    Returns:
      (conversation_summary, recent_messages, used_recent_messages_override)
    """

    default_recent_messages = serialize_history_as_messages(history)
    default_summary = summarize_conversation(history)

    override_used = recent_messages_override is not None
    recent_messages = recent_messages_override if override_used else default_recent_messages
    summary_input = (conversation_summary_input or "").strip()
    conversation_summary = summary_input or default_summary

    return conversation_summary, recent_messages, override_used
