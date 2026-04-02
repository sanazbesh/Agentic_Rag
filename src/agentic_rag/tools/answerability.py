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
CoverageStatus = Literal["none", "weak", "partial", "sufficient"]
CoverageReason = Literal[
    "no_context",
    "no_relevant_support",
    "definition_not_supported",
    "clause_supported",
    "summary_not_supported",
    "summary_partially_supported",
    "fact_not_found",
    "fact_supported",
    "comparison_not_supported",
    "comparison_partially_supported",
    "comparison_supported",
    "clarification_needed",
    "general_support",
    "other",
]
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


class CoverageEvaluation(BaseModel):
    """Strict evidence-coverage output contract for answerability evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str
    answerability_expectation: AnswerabilityExpectation
    coverage_status: CoverageStatus

    has_any_coverage: bool
    sufficient_coverage: bool
    partial_coverage: bool

    coverage_reason: CoverageReason | None

    matched_parent_chunk_ids: list[str] = Field(default_factory=list)
    matched_headings: list[str] = Field(default_factory=list)
    supporting_signals: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
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
        coverage = self.evaluate_coverage(
            query=query,
            query_understanding=query_understanding,
            context=retrieved_context,
        )
        return self._coverage_to_assessment(query_understanding=query_understanding, coverage=coverage)

    def evaluate_coverage(
        self,
        *,
        query: str,
        query_understanding: QueryUnderstandingResult,
        context: Sequence[object],
    ) -> CoverageEvaluation:
        original_query = (query or "").strip()
        expectation = query_understanding.answerability_expectation

        if expectation == "meta_response":
            return CoverageEvaluation(
                original_query=original_query,
                answerability_expectation=expectation,
                coverage_status="none",
                has_any_coverage=False,
                sufficient_coverage=False,
                partial_coverage=False,
                coverage_reason="no_context",
                warnings=["meta_response_no_retrieval_coverage_required"],
            )

        if expectation == "clarification_needed":
            return CoverageEvaluation(
                original_query=original_query,
                answerability_expectation=expectation,
                coverage_status="weak",
                has_any_coverage=False,
                sufficient_coverage=False,
                partial_coverage=False,
                coverage_reason="clarification_needed",
                missing_requirements=["query_requires_clarification_before_evidence_coverage"],
            )

        normalized_context = [self._normalize_context_item(item) for item in list(context or [])]
        if not normalized_context:
            return CoverageEvaluation(
                original_query=original_query,
                answerability_expectation=expectation,
                coverage_status="none",
                has_any_coverage=False,
                sufficient_coverage=False,
                partial_coverage=False,
                coverage_reason="no_context",
                warnings=["empty_retrieved_context"],
            )

        matched = list(normalized_context)
        heading_only = matched and all(self._is_heading_or_title_only(item) for item in matched)
        substantive = [item for item in matched if not self._is_heading_or_title_only(item)]

        matched_ids = [item["parent_chunk_id"] for item in matched if item["parent_chunk_id"]]
        matched_headings = [item["heading"] for item in matched if item["heading"]]
        distinct_sections = {item["heading"].lower() for item in substantive if item["heading"]}
        distinct_parents = {item["parent_chunk_id"] for item in substantive if item["parent_chunk_id"]}
        body_text = "\n".join(item["text"].lower() for item in substantive)

        if not matched:
            return CoverageEvaluation(
                original_query=original_query,
                answerability_expectation=expectation,
                coverage_status="none",
                has_any_coverage=False,
                sufficient_coverage=False,
                partial_coverage=False,
                coverage_reason="no_context",
            )

        def build(
            *,
            status: CoverageStatus,
            has_any_coverage: bool,
            sufficient_coverage: bool,
            partial_coverage: bool,
            reason: CoverageReason | None,
            supporting_signals: list[str] | None = None,
            missing_requirements: list[str] | None = None,
            warnings: list[str] | None = None,
        ) -> CoverageEvaluation:
            return CoverageEvaluation(
                original_query=original_query,
                answerability_expectation=expectation,
                coverage_status=status,
                has_any_coverage=has_any_coverage,
                sufficient_coverage=sufficient_coverage,
                partial_coverage=partial_coverage,
                coverage_reason=reason,
                matched_parent_chunk_ids=matched_ids,
                matched_headings=matched_headings,
                supporting_signals=supporting_signals or [],
                missing_requirements=missing_requirements or [],
                warnings=warnings or [],
            )

        if expectation == "definition_required":
            if heading_only:
                return build(
                    status="weak",
                    has_any_coverage=True,
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="definition_not_supported",
                    missing_requirements=["explicit_definitional_language"],
                    warnings=["heading_only_context"],
                )
            has_def_lang = bool(re.search(r"\b(is|means|refers to|defined as|definition of)\b", body_text))
            if has_def_lang:
                return build(
                    status="sufficient",
                    has_any_coverage=True,
                    sufficient_coverage=True,
                    partial_coverage=False,
                    reason="general_support",
                    supporting_signals=["definitional_language_detected"],
                )
            return build(
                status="weak",
                has_any_coverage=bool(substantive),
                sufficient_coverage=False,
                partial_coverage=False,
                reason="definition_not_supported",
                missing_requirements=["explicit_definitional_language"],
            )

        if expectation == "clause_lookup":
            if substantive:
                return build(
                    status="sufficient",
                    has_any_coverage=True,
                    sufficient_coverage=True,
                    partial_coverage=False,
                    reason="clause_supported",
                    supporting_signals=["substantive_clause_text_present"],
                )
            return build(
                status="weak",
                has_any_coverage=bool(matched),
                sufficient_coverage=False,
                partial_coverage=False,
                reason="no_relevant_support",
                missing_requirements=["substantive_clause_body_text"],
                warnings=["heading_only_context"] if heading_only else [],
            )

        if expectation == "summary":
            if len(distinct_sections) >= self.summary_min_chunks or len(distinct_parents) >= self.summary_min_chunks:
                return build(
                    status="sufficient",
                    has_any_coverage=True,
                    sufficient_coverage=True,
                    partial_coverage=False,
                    reason="general_support",
                    supporting_signals=["multi_section_context_detected"],
                )
            if substantive:
                return build(
                    status="partial",
                    has_any_coverage=True,
                    sufficient_coverage=False,
                    partial_coverage=True,
                    reason="summary_partially_supported",
                    supporting_signals=["single_section_context_only"],
                    missing_requirements=["multi_section_or_multi_parent_coverage"],
                )
            return build(
                status="weak",
                has_any_coverage=bool(matched),
                sufficient_coverage=False,
                partial_coverage=False,
                reason="summary_not_supported",
                missing_requirements=["substantive_multi_section_content"],
                warnings=["heading_only_context"] if heading_only else [],
            )

        if expectation == "fact_extraction":
            fact_markers = self._fact_markers(original_query)
            found = bool(substantive) and any(marker in body_text for marker in fact_markers)
            if found:
                return build(
                    status="sufficient",
                    has_any_coverage=True,
                    sufficient_coverage=True,
                    partial_coverage=False,
                    reason="fact_supported",
                    supporting_signals=["explicit_fact_statement_detected"],
                )
            return build(
                status="weak",
                has_any_coverage=bool(matched),
                sufficient_coverage=False,
                partial_coverage=False,
                reason="fact_not_found",
                missing_requirements=["explicit_fact_statement_in_context"],
                warnings=["heading_only_context"] if heading_only else [],
            )

        if expectation == "comparison":
            sides_present = self._comparison_side_hits(original_query, body_text)
            if sides_present >= 2:
                return build(
                    status="sufficient",
                    has_any_coverage=True,
                    sufficient_coverage=True,
                    partial_coverage=False,
                    reason="comparison_supported",
                    supporting_signals=["comparison_signals_for_both_sides_present"],
                )
            if sides_present == 1:
                return build(
                    status="partial",
                    has_any_coverage=True,
                    sufficient_coverage=False,
                    partial_coverage=True,
                    reason="comparison_partially_supported",
                    missing_requirements=["evidence_for_second_comparison_side"],
                )
            return build(
                status="weak",
                has_any_coverage=bool(substantive),
                sufficient_coverage=False,
                partial_coverage=False,
                reason="comparison_not_supported",
                missing_requirements=["evidence_for_both_comparison_sides"],
            )

        if expectation == "general_grounded_response" and substantive:
            return build(
                status="sufficient",
                has_any_coverage=True,
                sufficient_coverage=True,
                partial_coverage=False,
                reason="general_support",
            )
        return build(
            status="weak",
            has_any_coverage=bool(matched),
            sufficient_coverage=False,
            partial_coverage=False,
            reason="no_relevant_support",
            warnings=["heading_only_context"] if heading_only else [],
        )

    def _coverage_to_assessment(
        self,
        *,
        query_understanding: QueryUnderstandingResult,
        coverage: CoverageEvaluation,
    ) -> AnswerabilityAssessment:
        should_answer = bool(
            coverage.sufficient_coverage
            and query_understanding.answerability_expectation not in {"meta_response", "clarification_needed"}
        )
        insufficiency_reason = self._map_coverage_reason(coverage)
        return AnswerabilityAssessment(
            original_query=coverage.original_query,
            question_type=query_understanding.question_type,
            answerability_expectation=coverage.answerability_expectation,
            has_relevant_context=coverage.has_any_coverage,
            sufficient_context=coverage.sufficient_coverage,
            partially_supported=coverage.partial_coverage,
            should_answer=should_answer,
            support_level=coverage.coverage_status,
            insufficiency_reason=insufficiency_reason,
            matched_parent_chunk_ids=coverage.matched_parent_chunk_ids,
            matched_headings=coverage.matched_headings,
            evidence_notes=list(coverage.supporting_signals),
            warnings=list(coverage.warnings),
        )

    def _map_coverage_reason(self, coverage: CoverageEvaluation) -> InsufficiencyReason | None:
        if coverage.sufficient_coverage:
            return None
        if coverage.coverage_reason == "no_context":
            return "no_relevant_context"
        if coverage.coverage_reason == "definition_not_supported":
            if "heading_only_context" in coverage.warnings:
                return "only_title_or_heading_match"
            return "definition_not_supported"
        if coverage.coverage_reason == "summary_not_supported":
            return "summary_not_supported"
        if coverage.coverage_reason == "summary_partially_supported":
            return "partial_evidence_only"
        if coverage.coverage_reason == "fact_not_found":
            return "fact_not_found"
        if coverage.coverage_reason == "comparison_not_supported":
            return "comparison_not_supported"
        if coverage.coverage_reason == "comparison_partially_supported":
            return "partial_evidence_only"
        if coverage.coverage_reason == "clarification_needed":
            return "ambiguity_requires_clarification"
        if coverage.coverage_reason == "no_relevant_support":
            if "heading_only_context" in coverage.warnings:
                return "only_title_or_heading_match"
            return "topic_match_but_not_answer"
        return "other"

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

    def _comparison_side_hits(self, query: str, context_text: str) -> int:
        normalized_query = query.lower()
        comparator_markers = ("compare", "differ", "difference", "versus", "vs")
        if not any(marker in normalized_query for marker in comparator_markers):
            return 0
        sides = re.split(r"\b(?:with|versus|vs)\b", normalized_query)
        side_terms = [set(self._query_terms(side)) for side in sides if side.strip()]
        side_terms = [terms for terms in side_terms if terms]
        if len(side_terms) < 2:
            return 0
        return sum(1 for terms in side_terms[:2] if any(term in context_text for term in terms))

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


def evaluate_coverage(
    query: str,
    query_understanding: QueryUnderstandingResult,
    context: Sequence[object],
) -> CoverageEvaluation:
    """Return deterministic coverage evaluation for retrieved answer context."""

    return _DEFAULT_ANSWERABILITY_ASSESSOR.evaluate_coverage(
        query=query,
        query_understanding=query_understanding,
        context=context,
    )
