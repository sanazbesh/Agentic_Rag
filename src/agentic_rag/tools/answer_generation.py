"""Grounded legal answer generation from retrieved parent-chunk context.

This module implements a thin `generate_answer(context, query)` tool that:
- synthesizes answers strictly from supplied context (no retrieval, no memory)
- preserves legal qualifiers/limitations through extractive evidence use
- returns typed, traceable citations for downstream auditability
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class AnswerCitation:
    """Traceable citation to a single parent chunk supporting an answer claim."""

    parent_chunk_id: str
    document_id: str | None
    source_name: str | None
    heading: str | None
    supporting_excerpt: str | None


@dataclass(slots=True, frozen=True)
class GenerateAnswerResult:
    """Structured grounded-answer output.

    The answer is document-grounded, explicitly marks insufficiency, and keeps
    citations separate from prose for reliable downstream traceability.
    """

    answer_text: str
    grounded: bool
    sufficient_context: bool
    citations: list[AnswerCitation]
    warnings: list[str]


@dataclass(slots=True)
class LegalAnswerSynthesizer:
    """Deterministic extractive legal answer synthesizer.

    Grounded answering policy:
    - only uses provided parent/chunk context
    - forms claims from extracted sentences, not external knowledge
    - marks insufficiency when context cannot fully answer the query
    """

    min_query_overlap_for_relevance: int = 1
    max_support_points: int = 4
    qualifier_terms: tuple[str, ...] = (
        "unless",
        "except",
        "subject to",
        "provided that",
        "notwithstanding",
        "only if",
    )

    def generate(self, context: Sequence[object], query: str) -> GenerateAnswerResult:
        """Generate a grounded legal answer from retrieved context only."""

        try:
            normalized_query = (query or "").strip()
            normalized_context = [_normalize_context_item(item) for item in context]

            if not normalized_context:
                return GenerateAnswerResult(
                    answer_text=(
                        "Direct answer: No relevant information was retrieved for this question.\n\n"
                        "Supporting points:\n"
                        "- No parent chunks were provided to support an answer.\n\n"
                        "Caveats / limitations:\n"
                        "- The question cannot be answered from the available context."
                    ),
                    grounded=False,
                    sufficient_context=False,
                    citations=[],
                    warnings=["insufficient_context: no retrieved parent chunks"],
                )

            ranked = self._rank_relevant_chunks(normalized_context, normalized_query)
            if not ranked:
                return self._insufficient_response(
                    reason="insufficient_context: retrieved chunks do not address the query"
                )

            selected = ranked[: self.max_support_points]
            citations: list[AnswerCitation] = []
            supporting_lines: list[str] = []

            for item in selected:
                excerpt = self._best_excerpt(item["text"], normalized_query)
                if not excerpt:
                    continue
                citations.append(
                    AnswerCitation(
                        parent_chunk_id=item["parent_chunk_id"],
                        document_id=item.get("document_id"),
                        source_name=item.get("source_name"),
                        heading=item.get("heading"),
                        supporting_excerpt=excerpt,
                    )
                )
                heading = item.get("heading") or "(no heading)"
                supporting_lines.append(f"- {heading}: {excerpt}")

            if not citations:
                return self._insufficient_response(
                    reason="insufficient_context: no supported claims could be extracted"
                )

            direct = self._direct_answer(citations)
            caveats: list[str] = []

            sufficient_context = self._is_fully_answerable(ranked, normalized_query)
            if not sufficient_context:
                caveats.append("- The retrieved context appears partial for the full question.")

            qualifier_warning = self._check_qualifier_coverage(selected, citations)
            grounded = bool(citations) and qualifier_warning is None
            warnings: list[str] = []
            if qualifier_warning is not None:
                warnings.append(qualifier_warning)
                caveats.append("- Material legal qualifiers may be incompletely represented in the extracted support.")

            answer_text = (
                f"Direct answer: {direct}\n\n"
                "Supporting points:\n"
                + "\n".join(supporting_lines)
                + "\n\nCaveats / limitations:\n"
                + ("\n".join(caveats) if caveats else "- None identified within the retrieved context.")
            )

            if not citations:
                grounded = False

            return GenerateAnswerResult(
                answer_text=answer_text,
                grounded=grounded,
                sufficient_context=sufficient_context,
                citations=citations,
                warnings=warnings,
            )
        except Exception as exc:  # pragma: no cover - exercised in explicit fallback test
            return self._failure_response(str(exc))

    def _rank_relevant_chunks(self, context: Sequence[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        query_terms = _query_terms(query)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for idx, item in enumerate(context):
            text = item["text"].lower()
            overlap = sum(1 for term in query_terms if term in text)
            if overlap >= self.min_query_overlap_for_relevance:
                scored.append((overlap, idx, item))
        scored.sort(key=lambda row: (-row[0], row[1], row[2]["parent_chunk_id"]))
        return [row[2] for row in scored]

    def _best_excerpt(self, text: str, query: str) -> str | None:
        sentences = _split_sentences(text)
        if not sentences:
            return None

        query_terms = _query_terms(query)
        ranked = sorted(
            sentences,
            key=lambda s: (
                -sum(1 for term in query_terms if term in s.lower()),
                -int(any(token in s.lower() for token in self.qualifier_terms)),
                len(s),
            ),
        )
        best = ranked[0].strip()
        if not best:
            return None

        qualifier_sentence = next(
            (s.strip() for s in sentences if any(token in s.lower() for token in self.qualifier_terms)),
            None,
        )
        if qualifier_sentence and qualifier_sentence not in best:
            return f"{best} {qualifier_sentence}".strip()
        return best

    def _direct_answer(self, citations: Sequence[AnswerCitation]) -> str:
        first = citations[0].supporting_excerpt or "The retrieved context contains relevant provisions."
        return first

    def _is_fully_answerable(self, ranked: Sequence[dict[str, Any]], query: str) -> bool:
        query_terms = _query_terms(query)
        if not query_terms:
            return bool(ranked)
        normalized_terms = {_normalize_term(term) for term in query_terms}
        covered = {
            term
            for term in normalized_terms
            if any(term and term in _normalize_term(item["text"].lower()) for item in ranked)
        }
        if normalized_terms and covered == normalized_terms:
            return True

        top_text = _normalize_term(ranked[0]["text"].lower()) if ranked else ""
        top_overlap = sum(1 for term in normalized_terms if term and term in top_text)
        return top_overlap >= 2

    def _check_qualifier_coverage(
        self,
        selected: Sequence[dict[str, Any]],
        citations: Sequence[AnswerCitation],
    ) -> str | None:
        selected_text = " ".join(item["text"].lower() for item in selected)
        cited_text = " ".join((citation.supporting_excerpt or "").lower() for citation in citations)
        for qualifier in self.qualifier_terms:
            if qualifier in selected_text and qualifier not in cited_text:
                return "grounding_risk: qualifier_or_limitation_may_be_omitted"
        return None

    def _insufficient_response(self, reason: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=(
                "Direct answer: The retrieved context does not provide enough information to answer this question fully.\n\n"
                "Supporting points:\n"
                "- Available chunks do not contain sufficient directly responsive content.\n\n"
                "Caveats / limitations:\n"
                "- A reliable answer cannot be produced without additional relevant context."
            ),
            grounded=False,
            sufficient_context=False,
            citations=[],
            warnings=[reason],
        )

    def _failure_response(self, reason: str) -> GenerateAnswerResult:
        return GenerateAnswerResult(
            answer_text=(
                "Direct answer: The answer generation step failed safely.\n\n"
                "Supporting points:\n"
                "- No grounded claims were produced.\n\n"
                "Caveats / limitations:\n"
                "- Please retry with the same context after resolving the failure."
            ),
            grounded=False,
            sufficient_context=False,
            citations=[],
            warnings=[f"failure: {reason}"],
        )


_DEFAULT_ANSWER_SYNTHESIZER = LegalAnswerSynthesizer()


def generate_answer(context: Sequence[object], query: str) -> GenerateAnswerResult:
    """Generate a grounded legal answer from parent-chunk context.

    This tool is intentionally narrow: it does not retrieve data, use external
    knowledge, or access conversation memory. It reports insufficiency instead
    of guessing and returns structured citations for every supported claim.
    """

    return _DEFAULT_ANSWER_SYNTHESIZER.generate(context=context, query=query)


def _normalize_context_item(item: object) -> dict[str, Any]:
    text = _field(item, "compressed_text") or _field(item, "text") or ""
    return {
        "parent_chunk_id": str(_field(item, "parent_chunk_id") or ""),
        "document_id": _as_optional_str(_field(item, "document_id")),
        "source_name": _as_optional_str(_field(item, "source_name")),
        "heading": _as_optional_str(_field(item, "heading") or _field(item, "heading_text") or _field(item, "section")),
        "text": str(text),
    }


def _field(item: object, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _query_terms(query: str) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "what",
        "how",
        "when",
        "where",
        "why",
        "under",
        "for",
        "to",
        "of",
        "in",
        "on",
        "and",
        "or",
    }
    terms = [term for term in re.findall(r"[a-zA-Z0-9]+", (query or "").lower()) if term not in stop]
    return list(dict.fromkeys(terms))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _normalize_term(value: str) -> str:
    token = re.sub(r"[^a-z0-9\s]", " ", value.lower()).strip()
    token = re.sub(r"\b([a-z]{3,})s\b", r"\1", token)
    return re.sub(r"\s+", " ", token)
