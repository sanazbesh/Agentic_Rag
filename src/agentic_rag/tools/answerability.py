"""Deterministic answerability assessment before final legal answer generation.

Answerability checking is a dedicated safety gate between retrieval and final
answer synthesis: it verifies whether retrieved evidence is merely related or
actually sufficient for the query's answerability expectation.
"""

from __future__ import annotations

import re
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult


logger = logging.getLogger(__name__)

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
EvidenceStrength = Literal["none", "weak", "moderate", "strong"]
EvidenceStrengthReason = Literal[
    "no_context",
    "title_only_match",
    "heading_only_match",
    "thin_single_clause",
    "single_substantive_clause",
    "multiple_substantive_sections",
    "broad_multi_section_support",
    "other",
]
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


class EvidenceStrengthEvaluation(BaseModel):
    """Strict structural evidence-strength contract for answerability composition.

    Strength is intentionally evaluated separately from expectation-based coverage:
    heading/title matches can be topically related while still weak evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str
    evidence_strength: EvidenceStrength

    has_title_only_match: bool
    has_heading_only_match: bool
    has_substantive_clause_text: bool
    has_multiple_substantive_sections: bool

    distinct_parent_chunk_count: int
    distinct_heading_count: int
    approximate_text_span_count: int

    strength_reason: EvidenceStrengthReason | None

    supporting_signals: list[str] = Field(default_factory=list)
    weakness_signals: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class EvidenceStrengthThresholds:
    """Centralized deterministic thresholds for evidence-strength classification."""

    thin_clause_char_threshold: int = 80
    substantive_clause_char_threshold: int = 120
    multiple_substantive_sections_threshold: int = 2
    broad_parent_chunk_threshold: int = 2


@dataclass(slots=True)
class AnswerabilityAssessor:
    """Deterministic evaluator for evidence coverage versus query expectation."""

    min_substantive_chars: int = 40
    summary_min_chunks: int = 2
    evidence_strength_thresholds: EvidenceStrengthThresholds = EvidenceStrengthThresholds()

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
                logger.debug(
                    "evaluate_coverage_definition branch=heading_only sufficient_coverage=false matched_items=%s",
                    len(matched),
                )
                return build(
                    status="weak",
                    has_any_coverage=True,
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="definition_not_supported",
                    missing_requirements=["explicit_definitional_language"],
                    warnings=["heading_only_context"],
                )
            has_def_lang = self._has_definition_support(original_query, substantive)
            logger.debug(
                "evaluate_coverage_definition branch=substantive has_definitional_support=%s substantive_items=%s",
                has_def_lang,
                len(substantive),
            )
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

    def evaluate_evidence_strength(
        self,
        *,
        query: str,
        query_understanding: QueryUnderstandingResult,
        context: Sequence[object],
    ) -> EvidenceStrengthEvaluation:
        """Evaluate structural/substantive evidence strength independent of coverage.

        This method only inspects retrieved context depth/breadth and explicitly
        separates titles/headings from substantive clause body text.
        """

        del query_understanding  # intentionally not a primary strength driver
        original_query = (query or "").strip()
        normalized_context = [self._normalize_context_item(item) for item in list(context or [])]

        if not normalized_context:
            return EvidenceStrengthEvaluation(
                original_query=original_query,
                evidence_strength="none",
                has_title_only_match=False,
                has_heading_only_match=False,
                has_substantive_clause_text=False,
                has_multiple_substantive_sections=False,
                distinct_parent_chunk_count=0,
                distinct_heading_count=0,
                approximate_text_span_count=0,
                strength_reason="no_context",
                warnings=["empty_retrieved_context"],
            )

        threshold = self.evidence_strength_thresholds
        total_items = len(normalized_context)
        title_only_items = [item for item in normalized_context if self._is_title_only(item)]
        heading_only_items = [item for item in normalized_context if self._is_heading_only(item)]
        substantive_items = [item for item in normalized_context if self._is_substantive_body(item)]
        thin_body_items = [item for item in normalized_context if self._is_thin_body(item)]

        distinct_parent_chunk_count = len({item["parent_chunk_id"] for item in normalized_context if item["parent_chunk_id"]})
        distinct_heading_count = len({item["heading"].lower() for item in normalized_context if item["heading"]})
        substantive_parent_count = len({item["parent_chunk_id"] for item in substantive_items if item["parent_chunk_id"]})
        substantive_heading_count = len({item["heading"].lower() for item in substantive_items if item["heading"]})
        # Deterministic definition: number of substantive body blocks.
        approximate_text_span_count = len(substantive_items)

        has_title_only_match = bool(title_only_items) and not substantive_items and len(title_only_items) == total_items
        has_heading_only_match = bool(heading_only_items) and not substantive_items and len(heading_only_items) == total_items
        has_substantive_clause_text = bool(substantive_items)
        has_multiple_substantive_sections = (
            substantive_heading_count >= threshold.multiple_substantive_sections_threshold
            or substantive_parent_count >= threshold.multiple_substantive_sections_threshold
        )

        supporting_signals: list[str] = []
        weakness_signals: list[str] = []
        warnings: list[str] = []

        if has_substantive_clause_text:
            supporting_signals.append(f"substantive_body_blocks:{len(substantive_items)}")
        if substantive_heading_count:
            supporting_signals.append(f"substantive_distinct_headings:{substantive_heading_count}")
        if substantive_parent_count:
            supporting_signals.append(f"substantive_distinct_parents:{substantive_parent_count}")

        if has_title_only_match:
            weakness_signals.append("title_only_signal_without_body")
        if has_heading_only_match:
            weakness_signals.append("heading_only_signal_without_body")
        if thin_body_items and not substantive_items:
            weakness_signals.append("only_thin_body_fragments_detected")
        if not substantive_items:
            warnings.append("no_substantive_clause_text_detected")

        if has_title_only_match:
            return EvidenceStrengthEvaluation(
                original_query=original_query,
                evidence_strength="weak",
                has_title_only_match=True,
                has_heading_only_match=False,
                has_substantive_clause_text=False,
                has_multiple_substantive_sections=False,
                distinct_parent_chunk_count=distinct_parent_chunk_count,
                distinct_heading_count=distinct_heading_count,
                approximate_text_span_count=approximate_text_span_count,
                strength_reason="title_only_match",
                supporting_signals=supporting_signals,
                weakness_signals=weakness_signals,
                warnings=warnings,
            )

        if has_heading_only_match:
            return EvidenceStrengthEvaluation(
                original_query=original_query,
                evidence_strength="weak",
                has_title_only_match=False,
                has_heading_only_match=True,
                has_substantive_clause_text=False,
                has_multiple_substantive_sections=False,
                distinct_parent_chunk_count=distinct_parent_chunk_count,
                distinct_heading_count=distinct_heading_count,
                approximate_text_span_count=approximate_text_span_count,
                strength_reason="heading_only_match",
                supporting_signals=supporting_signals,
                weakness_signals=weakness_signals,
                warnings=warnings,
            )

        if not substantive_items and thin_body_items:
            return EvidenceStrengthEvaluation(
                original_query=original_query,
                evidence_strength="weak",
                has_title_only_match=False,
                has_heading_only_match=False,
                has_substantive_clause_text=False,
                has_multiple_substantive_sections=False,
                distinct_parent_chunk_count=distinct_parent_chunk_count,
                distinct_heading_count=distinct_heading_count,
                approximate_text_span_count=approximate_text_span_count,
                strength_reason="thin_single_clause",
                supporting_signals=supporting_signals,
                weakness_signals=weakness_signals,
                warnings=warnings,
            )

        if approximate_text_span_count == 1 and has_substantive_clause_text:
            return EvidenceStrengthEvaluation(
                original_query=original_query,
                evidence_strength="moderate",
                has_title_only_match=False,
                has_heading_only_match=False,
                has_substantive_clause_text=True,
                has_multiple_substantive_sections=False,
                distinct_parent_chunk_count=distinct_parent_chunk_count,
                distinct_heading_count=distinct_heading_count,
                approximate_text_span_count=approximate_text_span_count,
                strength_reason="single_substantive_clause",
                supporting_signals=supporting_signals,
                weakness_signals=weakness_signals,
                warnings=warnings,
            )

        if has_multiple_substantive_sections:
            reason: EvidenceStrengthReason = "multiple_substantive_sections"
            if (
                substantive_parent_count >= threshold.broad_parent_chunk_threshold
                and approximate_text_span_count >= threshold.multiple_substantive_sections_threshold
            ):
                reason = "broad_multi_section_support"
            return EvidenceStrengthEvaluation(
                original_query=original_query,
                evidence_strength="strong",
                has_title_only_match=False,
                has_heading_only_match=False,
                has_substantive_clause_text=True,
                has_multiple_substantive_sections=True,
                distinct_parent_chunk_count=distinct_parent_chunk_count,
                distinct_heading_count=distinct_heading_count,
                approximate_text_span_count=approximate_text_span_count,
                strength_reason=reason,
                supporting_signals=supporting_signals,
                weakness_signals=weakness_signals,
                warnings=warnings,
            )

        return EvidenceStrengthEvaluation(
            original_query=original_query,
            evidence_strength="moderate" if has_substantive_clause_text else "weak",
            has_title_only_match=False,
            has_heading_only_match=False,
            has_substantive_clause_text=has_substantive_clause_text,
            has_multiple_substantive_sections=False,
            distinct_parent_chunk_count=distinct_parent_chunk_count,
            distinct_heading_count=distinct_heading_count,
            approximate_text_span_count=approximate_text_span_count,
            strength_reason="other",
            supporting_signals=supporting_signals,
            weakness_signals=weakness_signals,
            warnings=warnings,
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

    def _is_title_only(self, item: Mapping[str, str]) -> bool:
        body = (item.get("text") or "").strip()
        heading = (item.get("heading") or "").strip()
        source_name = (item.get("source_name") or "").strip()
        if not body:
            return False
        if heading and body.lower() == heading.lower() and source_name and body.lower() == source_name.lower():
            return True
        if source_name and body.lower() == source_name.lower() and len(body) <= self.evidence_strength_thresholds.thin_clause_char_threshold:
            return True
        return False

    def _is_heading_only(self, item: Mapping[str, str]) -> bool:
        body = (item.get("text") or "").strip()
        heading = (item.get("heading") or "").strip()
        if not body:
            return False
        return bool(heading and body.lower() == heading.lower())

    def _is_substantive_body(self, item: Mapping[str, str]) -> bool:
        text = (item.get("text") or "").strip()
        heading = (item.get("heading") or "").strip()
        if not text:
            return False
        if heading and text.lower() == heading.lower():
            return False
        if len(text) < self.evidence_strength_thresholds.substantive_clause_char_threshold:
            return False
        return True

    def _is_thin_body(self, item: Mapping[str, str]) -> bool:
        text = (item.get("text") or "").strip()
        heading = (item.get("heading") or "").strip()
        if not text:
            return False
        if heading and text.lower() == heading.lower():
            return False
        return len(text) < self.evidence_strength_thresholds.substantive_clause_char_threshold

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

    def _has_definition_support(self, query: str, substantive: Sequence[Mapping[str, str]]) -> bool:
        targets = self._definition_targets(query)
        if not targets:
            return False

        for item in substantive:
            text = (item.get("text") or "").lower()
            if not text:
                continue
            for target in targets:
                escaped = re.escape(target)
                patterns = (
                    rf"\b{escaped}\b\s+means\b",
                    rf"\b{escaped}\b\s+refers\s+to\b",
                    rf"\b{escaped}\b\s+is\s+defined\s+as\b",
                    rf"\b{escaped}\b\s+is\s+(?:an?|the)\b",
                    rf"\bdefinition\s+of\s+{escaped}\b",
                    rf"[\"“']{escaped}[\"”']\s+means\b",
                )
                if any(re.search(pattern, text) for pattern in patterns):
                    return True
        return False

    def _definition_targets(self, query: str) -> tuple[str, ...]:
        normalized = " ".join((query or "").lower().strip().split())
        if not normalized:
            return ()

        leads = (
            r"^(?:what|who)\s+(?:is|are)\s+",
            r"^define\s+",
            r"^what\s+does\s+.+?\s+mean\s*$",
            r"^meaning\s+of\s+",
            r"^definition\s+of\s+",
        )
        candidate = normalized
        for lead in leads:
            candidate = re.sub(lead, "", candidate)
        candidate = candidate.rstrip(" ?.!")
        candidate = re.sub(r"\b(?:in|under|for)\s+this\s+agreement\b.*$", "", candidate).strip()
        if not candidate:
            return ()

        targets: list[str] = [candidate]
        if " " in candidate:
            targets.append(candidate.split()[-1])
        return tuple(dict.fromkeys(t for t in targets if t))

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
    """Coordinator layer for deterministic answerability gating.

    `assess_answerability(...)` is intentionally a strict coordinator:
    - delegates expectation-fit to `evaluate_coverage(...)` (primary authority)
    - delegates structural depth to `evaluate_evidence_strength(...)`
    - combines both typed outputs with one conservative policy
    - never silently upgrades weak/partial signals into sufficient answerability
    """

    logger.info("assess_answerability_started")
    try:
        coverage = evaluate_coverage(
            query=query,
            query_understanding=query_understanding,
            context=retrieved_context,
        )
    except Exception as exc:
        logger.warning("assess_answerability_coverage_failure error_type=%s", type(exc).__name__)
        expectation = str(getattr(query_understanding, "answerability_expectation", "general_grounded_response"))
        question_type = str(getattr(query_understanding, "question_type", "other_query"))
        safe_warning = f"coverage_evaluation_failed:{type(exc).__name__}"
        return AnswerabilityAssessment(
            original_query=(query or "").strip(),
            question_type=question_type,
            answerability_expectation=expectation,
            has_relevant_context=False,
            sufficient_context=False,
            partially_supported=False,
            should_answer=False,
            support_level="none",
            insufficiency_reason="other",
            matched_parent_chunk_ids=[],
            matched_headings=[],
            evidence_notes=[],
            warnings=[safe_warning],
        )

    try:
        strength = evaluate_evidence_strength(
            query=query,
            query_understanding=query_understanding,
            context=retrieved_context,
        )
    except Exception as exc:
        logger.warning("assess_answerability_strength_failure error_type=%s", type(exc).__name__)
        expectation = str(getattr(query_understanding, "answerability_expectation", "general_grounded_response"))
        question_type = str(getattr(query_understanding, "question_type", "other_query"))
        safe_warning = f"strength_evaluation_failed:{type(exc).__name__}"
        return AnswerabilityAssessment(
            original_query=(query or "").strip(),
            question_type=question_type,
            answerability_expectation=expectation,
            has_relevant_context=False,
            sufficient_context=False,
            partially_supported=False,
            should_answer=False,
            support_level="none",
            insufficiency_reason="other",
            matched_parent_chunk_ids=[],
            matched_headings=[],
            evidence_notes=[],
            warnings=[safe_warning],
        )

    # Explicit single-place combination policy:
    # - coverage is primary and cannot be overridden by strength
    # - moderate strength is allowed only when coverage is sufficient
    # - weak strength is only allowed when coverage is explicitly sufficient
    #   and weak-ness is not from title/heading-only structural signals
    support_level: SupportLevel = coverage.coverage_status
    sufficient_context = False
    partially_supported = bool(coverage.partial_coverage)
    should_answer = False
    insufficiency_reason = _DEFAULT_ANSWERABILITY_ASSESSOR._map_coverage_reason(coverage)

    if coverage.sufficient_coverage:
        partially_supported = False
        if strength.evidence_strength == "strong":
            sufficient_context = True
            should_answer = query_understanding.answerability_expectation not in {"meta_response", "clarification_needed"}
            support_level = "sufficient"
            insufficiency_reason = None
        elif strength.evidence_strength == "moderate":
            sufficient_context = True
            should_answer = query_understanding.answerability_expectation not in {"meta_response", "clarification_needed"}
            support_level = "sufficient"
            insufficiency_reason = None
        else:
            weak_but_explicitly_supported = (
                coverage.coverage_reason in {"clause_supported", "fact_supported", "comparison_supported", "general_support"}
                and "heading_only_context" not in coverage.warnings
                and strength.strength_reason not in {"title_only_match", "heading_only_match"}
            )
            if weak_but_explicitly_supported:
                sufficient_context = True
                should_answer = query_understanding.answerability_expectation not in {"meta_response", "clarification_needed"}
                support_level = "sufficient"
                insufficiency_reason = None
            else:
                sufficient_context = False
                should_answer = False
                support_level = "weak"
                insufficiency_reason = "partial_evidence_only"
    else:
        sufficient_context = False
        should_answer = False
        support_level = coverage.coverage_status

    evidence_notes = [
        *list(coverage.supporting_signals),
        f"evidence_strength:{strength.evidence_strength}",
        *(f"strength_signal:{s}" for s in strength.supporting_signals),
        *(f"weakness_signal:{s}" for s in strength.weakness_signals),
    ]
    warnings = [*list(coverage.warnings), *list(strength.warnings)]

    assessment = AnswerabilityAssessment(
        original_query=coverage.original_query,
        question_type=query_understanding.question_type,
        answerability_expectation=coverage.answerability_expectation,
        has_relevant_context=coverage.has_any_coverage,
        sufficient_context=sufficient_context,
        partially_supported=partially_supported,
        should_answer=should_answer,
        support_level=support_level,
        insufficiency_reason=insufficiency_reason,
        matched_parent_chunk_ids=coverage.matched_parent_chunk_ids,
        matched_headings=coverage.matched_headings,
        evidence_notes=evidence_notes,
        warnings=warnings,
    )
    logger.info(
        "assess_answerability_combined coverage_status=%s sufficient_coverage=%s evidence_strength=%s final_support_level=%s sufficient_context=%s should_answer=%s insufficiency_reason=%s",
        coverage.coverage_status,
        coverage.sufficient_coverage,
        strength.evidence_strength,
        assessment.support_level,
        assessment.sufficient_context,
        assessment.should_answer,
        assessment.insufficiency_reason,
    )
    return assessment


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


def evaluate_evidence_strength(
    query: str,
    query_understanding: QueryUnderstandingResult,
    context: Sequence[object],
) -> EvidenceStrengthEvaluation:
    """Return deterministic evidence-strength evaluation for retrieved context.

    Unlike coverage, this only measures structural and textual evidence depth.
    """

    return _DEFAULT_ANSWERABILITY_ASSESSOR.evaluate_evidence_strength(
        query=query,
        query_understanding=query_understanding,
        context=context,
    )
