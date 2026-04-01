"""Deterministic answerability assessment before final legal answer generation.

Answerability checking is a dedicated safety gate between retrieval and final
answer synthesis: it verifies whether retrieved evidence is merely related or
actually sufficient for the query's answerability expectation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult

QuestionType = Literal[
    "meta_query",
    "definition_query",
    "document_content_query",
    "document_summary_query",
    "extractive_fact_query",
    "comparison_query",
    "followup_query",
    "ambiguous_query",
    "other_query",
]
AnswerabilityExpectation = Literal[
    "definition_required",
    "clause_lookup",
    "summary",
    "fact_extraction",
    "comparison",
    "meta_response",
    "general_grounded_response",
    "clarification_needed",
]

SupportLevel = Literal["none", "weak", "partial", "sufficient"]
InsufficiencyReason = Literal[
    "no_relevant_context",
    "only_title_or_heading_match",
    "topic_match_but_not_answer",
    "partial_evidence_only",
    "definition_not_supported",
    "summary_not_supported",
    "comparison_not_supported",
    "fact_not_found",
    "ambiguity_requires_clarification",
    "other",
]


class AnswerabilityAssessment(BaseModel):
    """Strict answerability contract for routing and safe failure behavior.

    Sufficiency is stricter than relevance: related heading/topic matches can
    mark `has_relevant_context=True` while still requiring
    `sufficient_context=False` when answer-bearing language is missing.
    Definition and summary requests are intentionally stricter to prevent
    overconfident responses from thin evidence slices.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str
    question_type: QuestionType | str
    answerability_expectation: AnswerabilityExpectation | str

    has_relevant_context: bool
    sufficient_context: bool
    partially_supported: bool
    should_answer: bool

    support_level: SupportLevel
    insufficiency_reason: InsufficiencyReason | None

    matched_parent_chunk_ids: list[str] = Field(default_factory=list)
    matched_headings: list[str] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class AnswerabilityAssessor:
    """Deterministic evaluator for evidence coverage versus query expectation."""

    min_substantive_chars: int = 40
    summary_min_chunks: int = 2

    def assess(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
        retrieved_context: Sequence[object],
    ) -> AnswerabilityAssessment:
        original_query = (query or "").strip()
        expectation = query_understanding.answerability_expectation

        if expectation == "clarification_needed":
            return AnswerabilityAssessment(
                original_query=original_query,
                question_type=query_understanding.question_type,
                answerability_expectation=expectation,
                has_relevant_context=False,
                sufficient_context=False,
                partially_supported=False,
                should_answer=False,
                support_level="none",
                insufficiency_reason="ambiguity_requires_clarification",
                warnings=["clarification_needed_from_query_understanding"],
            )

        if expectation == "meta_response":
            return AnswerabilityAssessment(
                original_query=original_query,
                question_type=query_understanding.question_type,
                answerability_expectation=expectation,
                has_relevant_context=False,
                sufficient_context=False,
                partially_supported=False,
                should_answer=False,
                support_level="none",
                insufficiency_reason="other",
                warnings=["meta_response_handled_outside_retrieval_answerability"],
            )

        normalized_context = [self._normalize_context_item(item) for item in list(retrieved_context or [])]
        if not normalized_context:
            return AnswerabilityAssessment(
                original_query=original_query,
                question_type=query_understanding.question_type,
                answerability_expectation=expectation,
                has_relevant_context=False,
                sufficient_context=False,
                partially_supported=False,
                should_answer=False,
                support_level="none",
                insufficiency_reason="no_relevant_context",
                warnings=["empty_retrieved_context"],
            )

        query_terms = self._query_terms(original_query)
        matched = [item for item in normalized_context if self._is_relevant(item, query_terms)]
        if not matched and query_understanding.question_type in {
            "definition_query",
            "document_content_query",
            "document_summary_query",
            "extractive_fact_query",
            "comparison_query",
        }:
            matched = list(normalized_context)
        if not matched and expectation == "general_grounded_response":
            matched = list(normalized_context)
        heading_only = matched and all(self._is_heading_or_title_only(item) for item in matched)

        matched_ids = [item["parent_chunk_id"] for item in matched if item["parent_chunk_id"]]
        matched_headings = [item["heading"] for item in matched if item["heading"]]

        if not matched:
            return AnswerabilityAssessment(
                original_query=original_query,
                question_type=query_understanding.question_type,
                answerability_expectation=expectation,
                has_relevant_context=False,
                sufficient_context=False,
                partially_supported=False,
                should_answer=False,
                support_level="none",
                insufficiency_reason="no_relevant_context",
                warnings=["retrieved_context_not_relevant_to_query"],
            )

        if heading_only and expectation != "general_grounded_response":
            return AnswerabilityAssessment(
                original_query=original_query,
                question_type=query_understanding.question_type,
                answerability_expectation=expectation,
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=True,
                should_answer=False,
                support_level="weak",
                insufficiency_reason="only_title_or_heading_match",
                matched_parent_chunk_ids=matched_ids,
                matched_headings=matched_headings,
                evidence_notes=["relevant_heading_or_title_matches_without_substantive_body"],
            )

        return self._assess_by_expectation(
            original_query=original_query,
            question_type=query_understanding.question_type,
            expectation=expectation,
            matched=matched,
            matched_ids=matched_ids,
            matched_headings=matched_headings,
        )

    def _assess_by_expectation(
        self,
        *,
        original_query: str,
        question_type: QuestionType | str,
        expectation: AnswerabilityExpectation | str,
        matched: list[dict[str, str]],
        matched_ids: list[str],
        matched_headings: list[str],
    ) -> AnswerabilityAssessment:
        body_text = "\n".join(item["text"].lower() for item in matched)
        notes: list[str] = []

        def build(
            *,
            has_relevant_context: bool,
            sufficient_context: bool,
            partially_supported: bool,
            should_answer: bool,
            support_level: SupportLevel,
            insufficiency_reason: InsufficiencyReason | None,
            evidence_notes: list[str] | None = None,
            warnings: list[str] | None = None,
        ) -> AnswerabilityAssessment:
            return AnswerabilityAssessment(
                original_query=original_query,
                question_type=question_type,
                answerability_expectation=expectation,
                has_relevant_context=has_relevant_context,
                sufficient_context=sufficient_context,
                partially_supported=partially_supported,
                should_answer=should_answer,
                support_level=support_level,
                insufficiency_reason=insufficiency_reason,
                matched_parent_chunk_ids=matched_ids,
                matched_headings=matched_headings,
                evidence_notes=evidence_notes or [],
                warnings=warnings or [],
            )

        if expectation == "definition_required":
            has_def_lang = bool(re.search(r"\b(is|means|refers to|defined as|definition of)\b", body_text))
            if has_def_lang:
                notes.append("definitional_language_detected")
                return build(
                    has_relevant_context=True,
                    sufficient_context=True,
                    partially_supported=False,
                    should_answer=True,
                    support_level="sufficient",
                    insufficiency_reason=None,
                    evidence_notes=notes,
                )
            return build(
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=True,
                should_answer=False,
                support_level="partial",
                insufficiency_reason="definition_not_supported",
                evidence_notes=["topic_present_without_definitional_statement"],
            )

        if expectation == "clause_lookup":
            substantive = [item for item in matched if len(item["text"].strip()) >= self.min_substantive_chars]
            if substantive:
                return build(
                    has_relevant_context=True,
                    sufficient_context=True,
                    partially_supported=False,
                    should_answer=True,
                    support_level="sufficient",
                    insufficiency_reason=None,
                    evidence_notes=["substantive_clause_text_present"],
                )
            return build(
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=True,
                should_answer=False,
                support_level="weak",
                insufficiency_reason="topic_match_but_not_answer",
            )

        if expectation == "summary":
            distinct_chunks = {item["parent_chunk_id"] for item in matched if item["parent_chunk_id"]}
            if len(distinct_chunks) >= self.summary_min_chunks:
                return build(
                    has_relevant_context=True,
                    sufficient_context=True,
                    partially_supported=False,
                    should_answer=True,
                    support_level="sufficient",
                    insufficiency_reason=None,
                    evidence_notes=["multiple_substantive_sections_available_for_summary"],
                )
            return build(
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=True,
                should_answer=False,
                support_level="partial",
                insufficiency_reason="summary_not_supported",
                evidence_notes=["single_clause_or_section_only"],
            )

        if expectation == "fact_extraction":
            fact_markers = self._fact_markers(original_query)
            found = any(marker in body_text for marker in fact_markers)
            if found:
                return build(
                    has_relevant_context=True,
                    sufficient_context=True,
                    partially_supported=False,
                    should_answer=True,
                    support_level="sufficient",
                    insufficiency_reason=None,
                    evidence_notes=["explicit_fact_statement_detected"],
                )
            return build(
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=False,
                should_answer=False,
                support_level="weak",
                insufficiency_reason="fact_not_found",
            )

        if expectation == "comparison":
            if self._has_two_sides_signal(original_query, body_text):
                return build(
                    has_relevant_context=True,
                    sufficient_context=True,
                    partially_supported=False,
                    should_answer=True,
                    support_level="sufficient",
                    insufficiency_reason=None,
                    evidence_notes=["comparison_signals_for_both_sides_present"],
                )
            return build(
                has_relevant_context=True,
                sufficient_context=False,
                partially_supported=True,
                should_answer=False,
                support_level="partial",
                insufficiency_reason="comparison_not_supported",
                evidence_notes=["one_sided_or_incomplete_comparison_evidence"],
            )

        # general_grounded_response fallback
        if matched:
            return build(
                has_relevant_context=True,
                sufficient_context=True,
                partially_supported=False,
                should_answer=True,
                support_level="sufficient",
                insufficiency_reason=None,
            )
        return build(
            has_relevant_context=False,
            sufficient_context=False,
            partially_supported=False,
            should_answer=False,
            support_level="none",
            insufficiency_reason="no_relevant_context",
        )

    def _normalize_context_item(self, item: object) -> dict[str, str]:
        text = self._field(item, "compressed_text") or self._field(item, "text") or ""
        heading = self._field(item, "heading") or self._field(item, "heading_text") or self._field(item, "section") or ""
        return {
            "parent_chunk_id": str(self._field(item, "parent_chunk_id") or "").strip(),
            "heading": str(heading or "").strip(),
            "text": str(text or ""),
            "source_name": str(self._field(item, "source_name") or "").strip(),
        }

    def _is_relevant(self, item: Mapping[str, str], query_terms: set[str]) -> bool:
        haystack = f"{item.get('heading', '').lower()} {item.get('text', '').lower()}"
        tokens = set(re.findall(r"[a-z0-9]+", haystack))
        return any(term in tokens for term in query_terms)

    def _is_heading_or_title_only(self, item: Mapping[str, str]) -> bool:
        body = (item.get("text") or "").strip()
        heading = (item.get("heading") or "").strip()
        if not body:
            return True
        if heading and body.lower() == heading.lower():
            return True
        compact = " ".join(body.split())
        if len(compact.split()) <= 6 and not re.search(r"[.;:]", compact):
            return True
        return False

    def _query_terms(self, query: str) -> set[str]:
        stop = {"the", "a", "an", "what", "is", "are", "does", "say", "about", "this", "that", "which", "who", "how"}
        terms = [term for term in re.findall(r"[a-z0-9]+", (query or "").lower()) if term not in stop]
        return set(terms)

    def _fact_markers(self, query: str) -> tuple[str, ...]:
        lowered = query.lower()
        if "which law" in lowered or "govern" in lowered:
            return ("governed by", "laws of", "governing law")
        if "parties" in lowered or "who are the parties" in lowered:
            return ("between", "party", "parties", "by and between")
        if "notice period" in lowered:
            return ("notice period", "days notice", "written notice")
        return tuple(self._query_terms(query))

    def _has_two_sides_signal(self, query: str, context_text: str) -> bool:
        normalized_query = query.lower()
        comparator_markers = ("compare", "differ", "difference", "versus", "vs")
        if not any(marker in normalized_query for marker in comparator_markers):
            return False
        sides = re.split(r"\b(?:with|versus|vs)\b", normalized_query)
        side_terms = [set(self._query_terms(side)) for side in sides if side.strip()]
        side_terms = [terms for terms in side_terms if terms]
        if len(side_terms) < 2:
            return False
        return all(any(term in context_text for term in terms) for terms in side_terms[:2])

    def _field(self, item: object, key: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(key)
        return getattr(item, key, None)


_DEFAULT_ANSWERABILITY_ASSESSOR = AnswerabilityAssessor()


def assess_answerability(
    query: str,
    query_understanding: QueryUnderstandingResult,
    retrieved_context: Sequence[object],
) -> AnswerabilityAssessment:
    """Assess whether retrieved evidence is sufficient to answer safely.

    This function is a thin wrapper used by orchestration nodes; it never
    performs retrieval or final answer generation.
    """

    return _DEFAULT_ANSWERABILITY_ASSESSOR.assess(
        query=query,
        query_understanding=query_understanding,
        retrieved_context=retrieved_context,
    )
