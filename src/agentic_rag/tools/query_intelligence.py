"""Query-intelligence tools for retrieval-focused legal RAG pipelines.

These utilities transform user queries before retrieval without coupling to any
vector store, retriever, or answer-generation logic.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


class QueryTransformationLLM(Protocol):
    """Minimal prompt-based LLM abstraction for query transformation."""

    def complete(self, prompt: str) -> str:
        """Return raw model text for a prompt."""


@dataclass(slots=True, frozen=True)
class QueryRewriteResult:
    """Structured output for a retrieval-oriented query rewrite.

    Rewriting improves retrieval by clarifying ambiguous phrasing while
    preserving legal meaning (jurisdiction, parties, time scope, and cited
    sections/cases) from the original query and optional conversation context.
    """

    original_query: str
    rewritten_query: str
    used_conversation_context: bool
    rewrite_notes: str = ""


@dataclass(slots=True, frozen=True)
class QueryDecompositionResult:
    """Structured output for decomposition of legal queries into sub-queries.

    Decomposition improves legal retrieval coverage by splitting multi-issue
    questions into focused retrieval targets (e.g., rule vs. exception) while
    preserving deterministic ordering and avoiding invented legal issues.
    """

    original_query: str
    sub_queries: tuple[str, ...]
    used_conversation_context: bool
    decomposition_notes: str = ""


@dataclass(slots=True)
class QueryTransformationService:
    """Shared service for deterministic legal query rewriting and decomposition.

    Conversation context is optional. It is only applied when the incoming query
    appears referential/ambiguous and needs disambiguation for retrieval.
    """

    _AMBIGUOUS_REFERENCE_PATTERN = re.compile(
        r"\b(that|this|those|these|previous|prior)\s+(clause|section|case|example|one)\b|\b(it|that one|this one)\b",
        flags=re.IGNORECASE,
    )

    _LEGAL_REFERENCE_PATTERNS = (
        re.compile(r"\bSection\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\bClause\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\bArticle\s+[\w.\-()]+", flags=re.IGNORECASE),
        re.compile(r"\b[A-Z][a-zA-Z]+\s+v\.\s+[A-Z][a-zA-Z]+\b"),
        re.compile(r"\b[A-Z][A-Za-z\s]+\s+Act\b"),
    )
    llm_client: QueryTransformationLLM | None = None

    def rewrite_query(
        self,
        query: str,
        conversation_summary: str | None = None,
        recent_messages: Sequence[Any] | None = None,
    ) -> QueryRewriteResult:
        """Return one retrieval-optimized query string with optional context use."""

        original_query = query
        normalized_query = (query or "").strip()
        if not normalized_query:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query="",
                used_conversation_context=False,
                rewrite_notes="empty_input",
            )

        context_blob = _build_context_blob(conversation_summary, recent_messages)
        needs_context = bool(self._AMBIGUOUS_REFERENCE_PATTERN.search(normalized_query))

        if not needs_context:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=normalized_query,
                used_conversation_context=False,
                rewrite_notes="query_already_clear",
            )

        if self.llm_client is not None and context_blob:
            llm_ok, llm_result = self._llm_rewrite_query(normalized_query, context_blob)
            if llm_ok and llm_result is not None:
                return QueryRewriteResult(
                    original_query=original_query,
                    rewritten_query=llm_result,
                    used_conversation_context=True,
                    rewrite_notes="resolved_reference_with_llm",
                )
            if not llm_ok:
                return QueryRewriteResult(
                    original_query=original_query,
                    rewritten_query=normalized_query,
                    used_conversation_context=False,
                    rewrite_notes="llm_failure_fallback_original_query",
                )

        referent = self._extract_reference_target(context_blob) if context_blob else None
        if not referent:
            return QueryRewriteResult(
                original_query=original_query,
                rewritten_query=normalized_query,
                used_conversation_context=False,
                rewrite_notes="ambiguous_but_no_context_reference_found",
            )

        rewritten = self._replace_ambiguous_reference(normalized_query, referent)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten,
            used_conversation_context=True,
            rewrite_notes="resolved_reference_from_context",
        )

    def decompose_query(
        self,
        query: str,
        conversation_summary: str | None = None,
        recent_messages: Sequence[Any] | None = None,
    ) -> QueryDecompositionResult:
        """Split legal queries into retrieval-oriented sub-queries."""

        rewrite_result = self.rewrite_query(
            query=query,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
        )
        rewritten_query = rewrite_result.rewritten_query
        if not rewritten_query:
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=(),
                used_conversation_context=rewrite_result.used_conversation_context,
                decomposition_notes="empty_input",
            )

        if not _is_complex_query(rewritten_query):
            return QueryDecompositionResult(
                original_query=query,
                sub_queries=(rewritten_query,),
                used_conversation_context=rewrite_result.used_conversation_context,
                decomposition_notes="single_query",
            )

        parts: list[str]
        if self.llm_client is not None:
            llm_ok, llm_parts = self._llm_decompose_query(rewritten_query, conversation_summary, recent_messages)
            if not llm_ok:
                safe_original = (query or "").strip()
                safe_parts = (safe_original,) if safe_original else ()
                return QueryDecompositionResult(
                    original_query=query,
                    sub_queries=safe_parts,
                    used_conversation_context=False,
                    decomposition_notes="llm_failure_fallback_original_query",
                )
            parts = llm_parts if llm_parts else self._split_into_sub_queries(rewritten_query)
        else:
            parts = self._split_into_sub_queries(rewritten_query)
        if not parts:
            parts = [rewritten_query]

        return QueryDecompositionResult(
            original_query=query,
            sub_queries=tuple(parts),
            used_conversation_context=rewrite_result.used_conversation_context,
            decomposition_notes="single_query" if len(parts) == 1 else "multi_issue_query",
        )

    def _extract_reference_target(self, context_blob: str) -> str | None:
        for pattern in self._LEGAL_REFERENCE_PATTERNS:
            matches = list(pattern.finditer(context_blob))
            if matches:
                return matches[-1].group(0)
        return None

    def _replace_ambiguous_reference(self, query: str, referent: str) -> str:
        replacements = (
            r"\b(that|this|those|these|previous|prior)\s+clause\b",
            r"\b(that|this|those|these|previous|prior)\s+section\b",
            r"\b(that|this|those|these|previous|prior)\s+case\b",
            r"\b(that|this|those|these|previous|prior)\s+example\b",
            r"\b(that one|this one|it)\b",
        )
        rewritten = query
        for pattern in replacements:
            rewritten = re.sub(pattern, referent, rewritten, flags=re.IGNORECASE)
        return rewritten

    def _llm_rewrite_query(self, query: str, context_blob: str) -> tuple[bool, str | None]:
        prompt = (
            "Rewrite the legal retrieval query by resolving ambiguous references only from context. "
            "Preserve jurisdiction, dates, parties, and legal scope. "
            "Return strict JSON: {\"rewritten_query\": \"...\"}.\n\n"
            f"CONTEXT:\n{context_blob}\n\nQUERY:\n{query}"
        )
        try:
            raw = self.llm_client.complete(prompt)  # type: ignore[union-attr]
            parsed = json.loads(raw)
            rewritten = str(parsed.get("rewritten_query", "")).strip()
            return True, rewritten or None
        except Exception:
            logger.exception("LLM rewrite failed; using heuristic/safe fallback.")
            return False, None

    def _llm_decompose_query(
        self,
        query: str,
        conversation_summary: str | None,
        recent_messages: Sequence[Any] | None,
    ) -> tuple[bool, list[str] | None]:
        context_blob = _build_context_blob(conversation_summary, recent_messages)
        prompt = (
            "Decompose the legal query into minimal non-overlapping retrieval sub-queries. "
            "Do not invent legal issues. Preserve deterministic ordering. "
            "Return strict JSON: {\"sub_queries\": [\"...\"]}.\n\n"
            f"CONTEXT:\n{context_blob}\n\nQUERY:\n{query}"
        )
        try:
            raw = self.llm_client.complete(prompt)  # type: ignore[union-attr]
            parsed = json.loads(raw)
            items = parsed.get("sub_queries")
            if not isinstance(items, list):
                return True, None
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            return True, list(dict.fromkeys(cleaned)) or None
        except Exception:
            logger.exception("LLM decomposition failed; using heuristic/safe fallback.")
            return False, None

    def _split_into_sub_queries(self, query: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []

        base_parts = [part.strip(" .") for part in re.split(r"\s*;\s*", normalized) if part.strip()]
        output: list[str] = []
        for part in base_parts:
            if _is_rule_exception_pattern(part):
                split_rule = re.split(r"\band\b", part, maxsplit=1, flags=re.IGNORECASE)
                output.extend(chunk.strip(" .") for chunk in split_rule if chunk.strip())
                continue

            if _should_split_on_and(part):
                split_parts = re.split(r"\band\b", part, flags=re.IGNORECASE)
                output.extend(chunk.strip(" .") for chunk in split_parts if chunk.strip())
                continue

            output.append(part)

        deduped = list(dict.fromkeys(output))
        return deduped


def _build_context_blob(conversation_summary: str | None, recent_messages: Sequence[Any] | None) -> str:
    summary_text = (conversation_summary or "").strip()
    message_texts: list[str] = []
    for message in recent_messages or ():
        if isinstance(message, str):
            text = message.strip()
        elif isinstance(message, Mapping):
            text = str(message.get("content") or message.get("text") or message.get("message") or "").strip()
        else:
            text = str(message).strip()
        if text:
            message_texts.append(text)
    return "\n".join([summary_text, *message_texts]).strip()


def _is_rule_exception_pattern(part: str) -> bool:
    lowered = part.lower()
    return "rule" in lowered and "exception" in lowered and " and " in lowered


def _should_split_on_and(part: str) -> bool:
    lowered = part.lower()
    if " and " not in lowered:
        return False

    anchor_keywords = (
        "what is",
        "what are",
        "compare",
        "difference",
        "definition",
        "enforcement",
        "remedy",
        "elements",
        "defenses",
        "jurisdiction",
        "procedural",
        "substantive",
    )
    matched_keywords = sum(1 for keyword in anchor_keywords if keyword in lowered)
    return matched_keywords >= 2


def _is_complex_query(query: str) -> bool:
    lowered = query.lower()
    if ";" in query:
        return True
    if " and " in lowered and ("rule" in lowered or "definition" in lowered or "compare" in lowered):
        return True
    return False


_DEFAULT_QUERY_TRANSFORMATION_SERVICE = QueryTransformationService()


def rewrite_query(
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Any] | None = None,
) -> QueryRewriteResult:
    """Rewrite a user query into a retrieval-optimized legal query.

    The function is deterministic and optionally uses conversation context only
    to resolve ambiguous references in follow-up turns.
    """

    return _DEFAULT_QUERY_TRANSFORMATION_SERVICE.rewrite_query(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )


def decompose_query(
    query: str,
    conversation_summary: str | None = None,
    recent_messages: Sequence[Any] | None = None,
) -> QueryDecompositionResult:
    """Break a legal query into retrieval-focused sub-queries.

    Decomposition is conservative: simple queries return a single sub-query,
    while clearly multi-part legal requests are split deterministically.
    """

    return _DEFAULT_QUERY_TRANSFORMATION_SERVICE.decompose_query(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
    )
