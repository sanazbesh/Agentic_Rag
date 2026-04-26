"""Deterministic answerability assessment before final legal answer generation.

Answerability checking is a dedicated safety gate between retrieval and final
answer synthesis: it verifies whether retrieved evidence is merely related or
actually sufficient for the query's answerability expectation.
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult

from agentic_rag.tools.evidence_units import EvidenceUnit, build_evidence_units
from agentic_rag.tools.party_role_resolution import (
    compare_query_entities_against_extracted_parties,
    extract_intro_party_role_assignment,
    has_intro_role_signal,
    is_usable_party_entity,
    normalize_party_text,
    parse_party_verification_query_entities,
    pick_company_party,
    pick_individual_party,
)


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
    party_role_resolution_debug: "PartyRoleResolutionDebug | None" = None


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
    party_role_resolution_debug: "PartyRoleResolutionDebug | None" = None


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


PARTY_ROLE_PREVIEW_START_CHARS = 400
PARTY_ROLE_PREVIEW_END_CHARS = 250


class PartyRoleParentChunkDebugPreview(BaseModel):
    """Machine-readable parent chunk diagnostics for party-role resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text_length_chars: int
    preview_start: str
    preview_end: str
    contains_between_keyword: bool
    contains_and_keyword: bool
    contains_employer_label: bool
    contains_employee_label: bool
    contains_role_parenthetical: bool
    intro_pattern_detected: bool
    resolver_considered_usable_intro_text: bool
    parent_chunk_id: str | None = None
    heading: str | None = None
    source_name: str | None = None
    document_id: str | None = None


class PartyRoleResolutionDebug(BaseModel):
    """Structured runtime diagnostics emitted by deterministic role resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    party_role_resolution_checked_parent_count: int
    party_role_resolution_debug_outcome: Literal["resolved", "not_found"]
    party_role_resolution_debug_reason: str
    party_role_resolution_checked_parent_ids: list[str] = Field(default_factory=list)
    party_role_resolution_intro_pattern_parent_ids: list[str] = Field(default_factory=list)
    checked_parent_previews: list[PartyRoleParentChunkDebugPreview] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class EvidenceStrengthThresholds:
    """Centralized deterministic thresholds for evidence-strength classification."""

    thin_clause_char_threshold: int = 80
    substantive_clause_char_threshold: int = 120
    multiple_substantive_sections_threshold: int = 2
    broad_parent_chunk_threshold: int = 2


@dataclass(slots=True, frozen=True)
class _PartyRoleAssignment:
    parties: tuple[str, ...]
    employer: str | None
    employee: str | None
    company_side_party: str | None
    individual_side_party: str | None


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

        normalized_context = [self._unit_to_item(unit) for unit in build_evidence_units(list(context or []))]
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
            party_role_resolution_debug: PartyRoleResolutionDebug | None = None,
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
                party_role_resolution_debug=party_role_resolution_debug,
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
            has_operational_clause_support = self._has_operational_clause_definition_support(
                query=original_query,
                query_understanding=query_understanding,
                substantive=substantive,
                require_label_anchor=True,
            )
            logger.debug(
                "evaluate_coverage_definition branch=substantive has_definitional_support=%s has_operational_clause_support=%s substantive_items=%s",
                has_def_lang,
                has_operational_clause_support,
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
            if has_operational_clause_support:
                return build(
                    status="sufficient",
                    has_any_coverage=True,
                    sufficient_coverage=True,
                    partial_coverage=False,
                    reason="general_support",
                    supporting_signals=["operative_clause_language_detected"],
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
                canonical_what_is = any(
                    note == "debug:canonical_what_is=true" for note in (query_understanding.routing_notes or [])
                )
                clause_override_triggered = any(
                    note == "debug:clause_override_triggered=true" for note in (query_understanding.routing_notes or [])
                )
                clause_hint_match = any(
                    note == "debug:clause_hint_match=true" for note in (query_understanding.routing_notes or [])
                )
                has_multiword_clause_hint = any(
                    len(self._canonical_phrase(hint).split()) >= 2 for hint in (query_understanding.resolved_clause_hints or [])
                )
                if canonical_what_is and clause_override_triggered and clause_hint_match and has_multiword_clause_hint:
                    has_operational_clause_support = self._has_operational_clause_definition_support(
                        query=original_query,
                        query_understanding=query_understanding,
                        substantive=substantive,
                    )
                    if not has_operational_clause_support:
                        return build(
                            status="weak",
                            has_any_coverage=True,
                            sufficient_coverage=False,
                            partial_coverage=False,
                            reason="no_relevant_support",
                            missing_requirements=["specific_clause_or_topic_support_in_substantive_text"],
                        )
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="clause_supported",
                        supporting_signals=["substantive_clause_text_present", "operative_clause_language_detected"],
                    )
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
            is_policy_issue_query = self._is_policy_issue_spotting_query(original_query, query_understanding)
            if is_policy_issue_query:
                policy_issue_support = self._evaluate_policy_issue_spotting_support(
                    query=original_query,
                    substantive=matched,
                )
                if policy_issue_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(policy_issue_support["signals"]),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=list(policy_issue_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                )

            is_mitigation_query = self._is_employment_mitigation_query(original_query, query_understanding)
            if is_mitigation_query:
                mitigation_support = self._evaluate_employment_mitigation_support(
                    query=original_query,
                    substantive=matched,
                )
                if mitigation_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(mitigation_support["signals"]),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=list(mitigation_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                )

            is_lifecycle_query = self._is_employment_contract_lifecycle_query(original_query, query_understanding)
            if is_lifecycle_query:
                lifecycle_support = self._evaluate_employment_lifecycle_support(
                    query=original_query,
                    substantive=matched,
                )
                if lifecycle_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(lifecycle_support["signals"]),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=list(lifecycle_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                )

            is_financial_query = self._is_financial_entitlement_query(original_query, query_understanding)
            if is_financial_query:
                financial_support = self._evaluate_financial_entitlement_support(
                    query=original_query,
                    substantive=matched,
                )
                if financial_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(financial_support["signals"]),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=list(financial_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                )

            is_correspondence_query = self._is_correspondence_litigation_milestone_query(original_query, query_understanding)
            if is_correspondence_query:
                correspondence_support = self._evaluate_correspondence_litigation_milestone_support(
                    query=original_query,
                    substantive=matched,
                )
                if correspondence_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(correspondence_support["signals"]),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=list(correspondence_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                )

            is_matter_metadata_query = self._is_matter_metadata_query(original_query, query_understanding)
            if is_matter_metadata_query:
                found = self._has_matter_metadata_evidence(substantive, original_query)
                if found:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=["matter_document_metadata_responsive_evidence_detected"],
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=["matter_or_document_metadata_evidence_in_context"],
                    warnings=["heading_only_context"] if heading_only else [],
                )

            is_party_role_query = self._is_party_role_entity_query(original_query, query_understanding)
            if is_party_role_query:
                role_support = self._evaluate_party_role_support(substantive, original_query)
                if role_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(role_support["signals"]),
                        party_role_resolution_debug=cast(PartyRoleResolutionDebug | None, role_support.get("debug")),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    supporting_signals=list(role_support["signals"]),
                    missing_requirements=list(role_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                    party_role_resolution_debug=cast(PartyRoleResolutionDebug | None, role_support.get("debug")),
                )

            is_chronology_query = self._is_chronology_date_event_query(original_query, query_understanding)
            if is_chronology_query:
                chronology_support = self._evaluate_chronology_support(
                    query=original_query,
                    substantive=substantive,
                )
                if chronology_support["supported"]:
                    return build(
                        status="sufficient",
                        has_any_coverage=True,
                        sufficient_coverage=True,
                        partial_coverage=False,
                        reason="fact_supported",
                        supporting_signals=list(chronology_support["signals"]),
                    )
                return build(
                    status="weak",
                    has_any_coverage=bool(matched),
                    sufficient_coverage=False,
                    partial_coverage=False,
                    reason="fact_not_found",
                    missing_requirements=list(chronology_support["missing"]),
                    warnings=["heading_only_context"] if heading_only else [],
                )

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

        original_query = (query or "").strip()
        normalized_context = [self._unit_to_item(unit) for unit in build_evidence_units(list(context or []))]

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
        metadata_responsive_support = self._is_matter_metadata_query(query, query_understanding) and self._has_matter_metadata_evidence(
            normalized_context,
            query,
        )
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
            if metadata_responsive_support:
                return EvidenceStrengthEvaluation(
                    original_query=original_query,
                    evidence_strength="moderate",
                    has_title_only_match=False,
                    has_heading_only_match=False,
                    has_substantive_clause_text=False,
                    has_multiple_substantive_sections=False,
                    distinct_parent_chunk_count=distinct_parent_chunk_count,
                    distinct_heading_count=distinct_heading_count,
                    approximate_text_span_count=approximate_text_span_count,
                    strength_reason="other",
                    supporting_signals=[*supporting_signals, "metadata_responsive_header_or_caption_evidence"],
                    weakness_signals=weakness_signals,
                    warnings=[],
                )
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
            party_role_resolution_debug=coverage.party_role_resolution_debug,
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

    def _unit_to_item(self, unit: EvidenceUnit) -> dict[str, Any]:
        return {
            "evidence_unit_id": unit.evidence_unit_id,
            "parent_chunk_id": unit.parent_chunk_id,
            "heading": unit.heading,
            "text": unit.evidence_text,
            "source_name": unit.source_name,
            "metadata": unit.metadata,
        }

    def _is_relevant(self, item: Mapping[str, Any], query_terms: set[str]) -> bool:
        haystack = f"{item.get('heading', '').lower()} {item.get('text', '').lower()}"
        tokens = set(re.findall(r"[a-z0-9]+", haystack))
        return any(term in tokens for term in query_terms)

    def _is_heading_or_title_only(self, item: Mapping[str, Any]) -> bool:
        body = (item.get("text") or "").strip()
        heading = (item.get("heading") or "").strip()
        if not body:
            return True
        if heading and body.lower() == heading.lower():
            return True
        return False

    def _is_chronology_date_event_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(note == "legal_question_family:chronology_date_event" for note in (query_understanding.routing_notes or [])):
            return True
        lowered = self._canonical_phrase(query)
        chronology_patterns = (
            r"\bwhen did\b",
            r"\bwhen was\b",
            r"\btimeline\b",
            r"\bchronology\b",
            r"\bwhat happened first\b",
            r"\bwhat happened last\b",
            r"\bwhat happened after\b",
            r"\bwhat happened before\b",
            r"\bbetween\b.+\band\b",
            r"\ball dated events\b",
        )
        return any(re.search(pattern, lowered) for pattern in chronology_patterns)

    def _is_employment_contract_lifecycle_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(
            note == "legal_question_family:employment_contract_lifecycle"
            for note in (query_understanding.routing_notes or [])
        ):
            return True
        lowered = self._canonical_phrase(query)
        patterns = (
            r"\bwhen\s+did\s+(?:employment|the employment relationship)\s+(?:begin|start|commence)\b",
            r"\b(?:employment\s+)?start\s+date\b",
            r"\bcommencement\s+date\b",
            r"\boffer\s+and\s+acceptance\b",
            r"\bwhen\s+was\s+the\s+offer\s+accepted\b",
            r"\bprobation(?:ary)?\b",
            r"\bcompensation\s+terms\b",
            r"\bsalary\b",
            r"\bbenefits\b",
            r"\btermination\s+effective\s+date\b",
            r"\bwhen\s+did\s+termination\s+take\s+effect\b",
            r"\bseverance\b",
            r"\broe\b",
            r"\brecord\s+of\s+employment\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _is_employment_mitigation_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(note == "legal_question_family:employment_mitigation" for note in (query_understanding.routing_notes or [])):
            return True
        lowered = self._canonical_phrase(query)
        patterns = (
            r"\bmitigat(?:e|ion)\b",
            r"\bmitigation efforts?\b",
            r"\bjob applications?\b",
            r"\bhow many job applications?\b",
            r"\binterviews?\b",
            r"\boffers?\s+(?:received|rejected)\b",
            r"\balternative employment\b",
            r"\bnew employment\b",
            r"\bmitigation evidence\b",
            r"\bjob search\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _is_correspondence_litigation_milestone_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(
            note == "legal_question_family:correspondence_litigation_milestone"
            for note in (query_understanding.routing_notes or [])
        ):
            return True
        lowered = self._canonical_phrase(query)
        patterns = (
            r"\bwhat\s+letters?\s+(?:were|was)\s+sent\b",
            r"\bwhat\s+emails?\s+(?:were|was)\s+sent\b",
            r"\bwhat\s+deadlines?\s+(?:were|was)\s+demanded\b",
            r"\bwhen\s+was\s+the\s+claim\s+filed\b",
            r"\bwhen\s+was\s+the\s+defen(?:c|s)e\s+(?:due|filed)\b",
            r"\bwhat\s+happened\s+procedurally\b",
            r"\bprocedural\s+(?:history|status)\b",
            r"\bcourt\s+filings?\b",
            r"\bdefault\s+notice\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _is_policy_issue_spotting_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(note == "legal_question_family:policy_issue_spotting" for note in (query_understanding.routing_notes or [])):
            return True
        lowered = self._canonical_phrase(query)
        if any(token in lowered for token in ("reimbursement", "reimbursements", "expense", "expenses", "salary", "payroll", "pay stub")) and not any(
            token in lowered for token in ("policy", "policies", "dispute", "claim", "legal issue", "legal issues", "clause", "clauses")
        ):
            return False
        patterns = (
            r"\bwhat\s+polic(?:y|ies)\s+(?:are|is)\s+relevant\b",
            r"\bwhat\s+legal\s+issues?\s+(?:are|is)\s+raised\b",
            r"\bwhat\s+(?:are|is)\s+the\s+key\s+issues?\b",
            r"\bnature\s+of\s+the\s+claim\b",
            r"\bwhat\s+is\s+the\s+nature\s+of\s+the\s+claim\b",
            r"\bclauses?\s+or\s+polic(?:y|ies)\s+relat(?:e|ed)\s+to\s+(?:this|the)\s+dispute\b",
        )
        if any(re.search(pattern, lowered) for pattern in patterns):
            return True
        return (
            any(token in lowered for token in ("policy", "policies", "handbook", "issue", "issues", "dispute", "claim"))
            and any(token in lowered for token in ("relevant", "raised", "key", "nature", "related", "relate", "at issue"))
        )

    def _policy_issue_target(self, lowered_query: str) -> str:
        if "nature of the claim" in lowered_query:
            return "nature_of_claim"
        if "key issues" in lowered_query:
            return "key_issues"
        if "legal issues" in lowered_query or ("issues" in lowered_query and "raised" in lowered_query):
            return "legal_issues"
        if "relevant policies" in lowered_query or ("policies" in lowered_query and "relevant" in lowered_query):
            return "relevant_policies"
        if "clause" in lowered_query and ("dispute" in lowered_query or "policy" in lowered_query):
            return "dispute_clause_or_policy"
        return "general_policy_issue"

    def _evaluate_policy_issue_spotting_support(
        self,
        *,
        query: str,
        substantive: Sequence[Mapping[str, str]],
    ) -> dict[str, object]:
        lowered_query = self._canonical_phrase(query)
        target = self._policy_issue_target(lowered_query)
        if not substantive:
            return {"supported": False, "signals": [], "missing": ["policy_or_issue_spotting_responsive_evidence"]}

        rows: list[str] = []
        for item in substantive:
            text = self._canonical_phrase(f"{item.get('heading', '')} {item.get('text', '')}")
            if text:
                rows.append(text)

        policy_markers = ("policy", "policies", "handbook", "workplace policy", "code of conduct", "procedure", "protocol")
        clause_markers = ("clause", "clauses", "section", "article", "provision", "term")
        legal_reference_markers = ("statute", "act", "regulation", "esa", "employment standards act", "human rights code")
        issue_markers = (
            "issue",
            "issues",
            "dispute",
            "allegation",
            "alleged",
            "violation",
            "breach",
            "non compliance",
            "wrongful dismissal",
            "constructive dismissal",
            "retaliation",
            "discrimination",
        )
        claim_markers = ("claim", "cause of action", "damages", "relief", "seeks", "asserts")
        rights_obligation_markers = ("right", "rights", "obligation", "obligations", "duty", "duties", "entitlement")

        def has_pair(
            primary_markers: Sequence[str],
            responsive_markers: Sequence[str],
            alternate_responsive: Sequence[str] = (),
        ) -> bool:
            for text in rows:
                if not any(marker in text for marker in primary_markers):
                    continue
                if any(marker in text for marker in responsive_markers):
                    return True
                if alternate_responsive and any(marker in text for marker in alternate_responsive):
                    return True
            return False

        if target == "relevant_policies":
            supported = has_pair(policy_markers, issue_markers, claim_markers) or has_pair(
                legal_reference_markers,
                issue_markers,
                rights_obligation_markers,
            )
            if supported:
                return {"supported": True, "signals": ["policy_issue_relevant_policy_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["policy_handbook_or_statutory_issue_responsive_evidence"]}

        if target == "legal_issues":
            supported = has_pair(issue_markers, claim_markers, rights_obligation_markers) or has_pair(
                legal_reference_markers,
                issue_markers,
                claim_markers,
            )
            if supported:
                return {"supported": True, "signals": ["policy_issue_legal_issue_framing_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["issue_framing_or_statutory_violation_evidence"]}

        if target == "dispute_clause_or_policy":
            supported = has_pair(clause_markers, issue_markers, claim_markers) or has_pair(
                policy_markers,
                issue_markers,
                claim_markers,
            )
            if supported:
                return {"supported": True, "signals": ["policy_issue_dispute_clause_or_policy_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["clause_or_policy_language_linked_to_dispute_evidence"]}

        if target == "nature_of_claim":
            supported = has_pair(claim_markers, issue_markers, legal_reference_markers) or has_pair(
                ("wrongful dismissal", "constructive dismissal", "harassment", "retaliation", "discrimination"),
                claim_markers,
                issue_markers,
            )
            if supported:
                return {"supported": True, "signals": ["policy_issue_nature_of_claim_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["claim_framing_or_alleged_violation_evidence"]}

        if target == "key_issues":
            supported = has_pair(issue_markers, claim_markers, rights_obligation_markers) or has_pair(
                issue_markers,
                legal_reference_markers,
                ("procedural status", "status"),
            )
            if supported:
                return {"supported": True, "signals": ["policy_issue_key_issues_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["key_issue_summary_or_dispute_theme_evidence"]}

        supported = (
            has_pair(policy_markers, issue_markers, claim_markers)
            or has_pair(clause_markers, issue_markers, claim_markers)
            or has_pair(legal_reference_markers, issue_markers, claim_markers)
        )
        if supported:
            return {"supported": True, "signals": ["policy_issue_general_responsive_evidence_detected"], "missing": []}
        return {"supported": False, "signals": [], "missing": ["policy_or_issue_spotting_responsive_evidence"]}

    def _procedural_target(self, lowered_query: str) -> str:
        if any(token in lowered_query for token in ("what letters", "what emails", "communications were sent", "correspondence")):
            return "communications_sent"
        if "deadline" in lowered_query and any(token in lowered_query for token in ("demand", "demanded", "demand letter")):
            return "demand_deadlines"
        if "claim" in lowered_query and "filed" in lowered_query:
            return "claim_filed"
        if "defen" in lowered_query and any(token in lowered_query for token in ("due", "filed")):
            return "defence_due_or_filed"
        if "procedural" in lowered_query or "court filing" in lowered_query or "what happened procedurally" in lowered_query:
            return "procedural_history"
        if any(token in lowered_query for token in ("pleading", "served", "service", "default notice", "settlement")):
            return "procedural_milestone"
        return "unknown"

    def _evaluate_employment_lifecycle_support(
        self,
        *,
        query: str,
        substantive: Sequence[Mapping[str, str]],
    ) -> dict[str, object]:
        lowered_query = self._canonical_phrase(query)
        requires_date = self._is_lifecycle_when_date_required_query(lowered_query)
        if not substantive:
            return {"supported": False, "signals": [], "missing": ["employment_lifecycle_responsive_evidence"]}

        def has_responsive_evidence(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for item in substantive:
                text = self._canonical_phrase(f"{item.get('heading', '')} {item.get('text', '')}")
                if not text:
                    continue
                if not all(token in text for token in required):
                    continue
                if any_of and not any(token in text for token in any_of):
                    continue
                return True
            return False

        def has_responsive_dated_evidence(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for item in substantive:
                text = self._canonical_phrase(f"{item.get('heading', '')} {item.get('text', '')}")
                if not text:
                    continue
                if not all(token in text for token in required):
                    continue
                if any_of and not any(token in text for token in any_of):
                    continue
                raw_text = f"{item.get('heading', '')} {item.get('text', '')}"
                if self._extract_datetimes(raw_text):
                    return True
            return False

        if "offer" in lowered_query and "accept" in lowered_query:
            if has_responsive_dated_evidence(("offer",), ("accept", "accepted", "acceptance")) if requires_date else has_responsive_evidence(("offer",), ("accept", "accepted", "acceptance")):
                return {"supported": True, "signals": ["employment_lifecycle_offer_acceptance_evidence_detected"], "missing": []}
            missing = "offer_and_acceptance_dated_evidence" if requires_date else "offer_and_acceptance_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        if any(token in lowered_query for token in ("start date", "commencement", "employment begin", "employment start", "employment relationship begin")):
            has_support = (
                has_responsive_evidence(("employment",), ("effective date", "commence", "start", "began", "begin"))
                or has_responsive_evidence(("commencement",), ("date",))
                or has_responsive_evidence(
                    ("effective date",),
                    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "20", "19"),
                )
                or has_responsive_evidence(("start date",))
            )
            has_dated_support = (
                has_responsive_dated_evidence(("employment",), ("effective date", "commence", "start", "began", "begin"))
                or has_responsive_dated_evidence(("commencement",), ("date",))
                or has_responsive_dated_evidence(("effective date",))
                or has_responsive_dated_evidence(("start date",))
            )
            if (has_dated_support if requires_date else has_support):
                return {
                    "supported": True,
                    "signals": [
                        "employment_lifecycle_start_or_commencement_evidence_detected",
                        "employment_start_date_supported",
                    ],
                    "missing": [],
                }
            missing = "employment_start_or_commencement_dated_evidence" if requires_date else "employment_start_or_commencement_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        if "probation" in lowered_query:
            if has_responsive_dated_evidence(("probation",), ("month", "day", "end", "period", "complete")) if requires_date else has_responsive_evidence(("probation",), ("month", "day", "end", "period", "complete")):
                return {"supported": True, "signals": ["employment_lifecycle_probation_evidence_detected"], "missing": []}
            missing = "probation_term_or_end_dated_evidence" if requires_date else "probation_term_or_end_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        if any(token in lowered_query for token in ("compensation", "salary", "wage", "remuneration")):
            if has_responsive_evidence(("compensation",), ("salary", "base", "$", "annual", "bonus")) or has_responsive_evidence(("salary",)):
                return {"supported": True, "signals": ["employment_lifecycle_compensation_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["compensation_or_salary_evidence"]}

        if "benefit" in lowered_query:
            if has_responsive_evidence(("benefit",), ("eligible", "coverage", "plan", "insurance", "vacation", "rrsp", "health")):
                return {"supported": True, "signals": ["employment_lifecycle_benefits_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["benefits_terms_or_eligibility_evidence"]}

        if "termination" in lowered_query and any(token in lowered_query for token in ("effective", "take effect", "date", "terminated")):
            if has_responsive_dated_evidence(("termination",), ("effective", "terminated on", "date", "cease")) if requires_date else has_responsive_evidence(("termination",), ("effective", "terminated on", "date", "cease")):
                return {"supported": True, "signals": ["employment_lifecycle_termination_effective_evidence_detected"], "missing": []}
            missing = "termination_effective_dated_evidence" if requires_date else "termination_effective_date_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        if "severance" in lowered_query:
            if has_responsive_evidence(("severance",), ("pay", "payable", "weeks", "salary", "offered")):
                return {"supported": True, "signals": ["employment_lifecycle_severance_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["severance_terms_evidence"]}

        if "roe" in lowered_query or "record of employment" in lowered_query:
            has_support = has_responsive_evidence(("record of employment",), ("roe", "issued", "issue", "provide", "days")) or has_responsive_evidence(("roe",), ("issued", "issue", "provide", "days"))
            has_dated_support = has_responsive_dated_evidence(("record of employment",), ("roe", "issued", "issue", "provide", "days")) or has_responsive_dated_evidence(("roe",), ("issued", "issue", "provide", "days"))
            if (has_dated_support if requires_date else has_support):
                return {"supported": True, "signals": ["employment_lifecycle_roe_evidence_detected"], "missing": []}
            missing = "roe_issuance_dated_evidence" if requires_date else "roe_issuance_timing_or_reference_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        return {"supported": False, "signals": [], "missing": ["employment_lifecycle_responsive_evidence"]}

    def _is_financial_entitlement_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(note == "legal_question_family:financial_entitlement" for note in (query_understanding.routing_notes or [])):
            return True
        lowered = self._canonical_phrase(query)
        patterns = (
            r"\bcompensation\b",
            r"\bpromised\s+compensation\b",
            r"\bsalary\b",
            r"\bpay\s+rate\b",
            r"\bunpaid\b",
            r"\bbonus\b",
            r"\bvacation\s+pay\b",
            r"\breimburse(?:ment|ments)?\b",
            r"\bexpenses?\b",
            r"\bseverance\b",
            r"\bfinancial\s+records?\b",
            r"\bpay\s+stub(?:s)?\b",
            r"\bpayroll\s+records?\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _financial_target(self, lowered_query: str) -> str:
        if "financial records" in lowered_query or "what records support" in lowered_query or "support the claim" in lowered_query:
            return "financial_records"
        if "unpaid" in lowered_query:
            return "unpaid_amounts"
        if "bonus" in lowered_query or "vacation pay" in lowered_query:
            return "bonus_or_vacation_pay"
        if "reimbursement" in lowered_query or "expense" in lowered_query:
            return "reimbursement"
        if "severance" in lowered_query:
            return "severance"
        if any(token in lowered_query for token in ("compensation", "salary", "pay rate", "remuneration", "promised")):
            return "compensation"
        return "general_financial_entitlement"

    def _evaluate_financial_entitlement_support(
        self,
        *,
        query: str,
        substantive: Sequence[Mapping[str, str]],
    ) -> dict[str, object]:
        lowered_query = self._canonical_phrase(query)
        target = self._financial_target(lowered_query)
        if not substantive:
            return {"supported": False, "signals": [], "missing": ["financial_entitlement_responsive_evidence"]}

        rows: list[str] = []
        for item in substantive:
            text = self._canonical_phrase(f"{item.get('heading', '')} {item.get('text', '')}")
            if text:
                rows.append(text)

        def has(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for text in rows:
                if not all(token in text for token in required):
                    continue
                if any_of and not any(token in text for token in any_of):
                    continue
                return True
            return False

        if target == "compensation":
            supported = (
                has(("compensation",), ("salary", "base", "annual", "payable", "$"))
                or has(("salary",), ("annual", "per year", "payable", "$", "bi weekly", "hourly"))
                or has(("pay rate",), ("hourly", "salary", "$"))
            )
            if supported:
                return {"supported": True, "signals": ["financial_entitlement_compensation_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["promised_compensation_or_salary_terms_evidence"]}

        if target == "unpaid_amounts":
            supported = has(("unpaid",), ("amount", "$", "wage", "salary", "bonus", "vacation", "claim")) or has(
                ("amount",),
                ("owing", "outstanding", "unpaid"),
            )
            if supported:
                return {"supported": True, "signals": ["financial_entitlement_unpaid_amount_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["unpaid_amount_or_outstanding_claim_evidence"]}

        if target == "bonus_or_vacation_pay":
            supported = has(("bonus",), ("entitle", "eligible", "earned", "payable", "claim")) or has(
                ("vacation",),
                ("pay", "accrued", "entitle", "earned", "claim"),
            )
            if supported:
                return {"supported": True, "signals": ["financial_entitlement_bonus_or_vacation_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["bonus_or_vacation_pay_entitlement_evidence"]}

        if target == "reimbursement":
            supported = has(("reimburse",), ("expense", "invoice", "receipt", "mileage", "claim", "submitted")) or has(
                ("expense",),
                ("reimburse", "receipt", "claim", "submitted"),
            )
            if supported:
                return {"supported": True, "signals": ["financial_entitlement_reimbursement_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["expense_or_reimbursement_record_evidence"]}

        if target == "severance":
            supported = has(("severance",), ("pay", "payable", "weeks", "salary", "offered", "entitle"))
            if supported:
                return {"supported": True, "signals": ["financial_entitlement_severance_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["severance_payment_or_entitlement_evidence"]}

        if target == "financial_records":
            supported = (
                has(("pay stub",), ("pay period", "gross", "net", "earnings"))
                or has(("payroll",), ("record", "register", "earnings", "deduction"))
                or has(("expense",), ("record", "receipt", "reimburse"))
                or has(("demand letter",), ("amount", "claim", "owed", "unpaid"))
                or has(("financial summary",), ("amount", "claim", "owed"))
            )
            if supported:
                return {"supported": True, "signals": ["financial_entitlement_records_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["pay_stub_payroll_expense_or_financial_records_evidence"]}

        supported = (
            has(("compensation",), ("salary", "pay", "bonus", "vacation"))
            or has(("unpaid",), ("amount", "claim", "owed"))
            or has(("severance",), ("pay", "weeks", "salary"))
        )
        if supported:
            return {"supported": True, "signals": ["financial_entitlement_responsive_evidence_detected"], "missing": []}
        return {"supported": False, "signals": [], "missing": ["financial_entitlement_responsive_evidence"]}

    def _is_lifecycle_when_date_required_query(self, lowered_query: str) -> bool:
        return "when" in lowered_query and any(
            token in lowered_query
            for token in (
                "start",
                "commencement",
                "probation",
                "termination",
                "offer",
                "accept",
            )
        )

    def _evaluate_correspondence_litigation_milestone_support(
        self,
        *,
        query: str,
        substantive: Sequence[Mapping[str, str]],
    ) -> dict[str, object]:
        lowered_query = self._canonical_phrase(query)
        target = self._procedural_target(lowered_query)
        requires_date = "when" in lowered_query or any(token in lowered_query for token in ("date", "due", "deadline"))
        if not substantive:
            return {"supported": False, "signals": [], "missing": ["procedural_or_correspondence_responsive_evidence"]}

        rows: list[tuple[str, str]] = []
        for item in substantive:
            raw_text = f"{item.get('heading', '')} {item.get('text', '')}".strip()
            canonical = self._canonical_phrase(raw_text)
            if canonical:
                rows.append((canonical, raw_text))

        def has(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for canonical, _raw in rows:
                if not all(token in canonical for token in required):
                    continue
                if any_of and not any(token in canonical for token in any_of):
                    continue
                return True
            return False

        def has_dated(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for canonical, raw in rows:
                if not all(token in canonical for token in required):
                    continue
                if any_of and not any(token in canonical for token in any_of):
                    continue
                if self._extract_datetimes(raw):
                    return True
            return False

        if target == "communications_sent":
            supported = has_dated(("letter",), ("sent", "email", "dated")) or has_dated(("email",), ("sent", "subject", "dated"))
            if not supported and not requires_date:
                supported = has(("letter",), ("sent", "email", "dated")) or has(("email",), ("sent", "subject", "dated"))
            if supported:
                return {"supported": True, "signals": ["correspondence_dated_communication_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["dated_letter_or_email_sent_evidence"]}

        if target == "demand_deadlines":
            supported = has_dated(("demand",), ("by", "no later", "deadline", "within")) if requires_date else has(("demand",), ("by", "no later", "deadline", "within"))
            if supported:
                return {"supported": True, "signals": ["procedural_demand_deadline_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["demand_deadline_language_evidence"]}

        if target == "claim_filed":
            supported = has_dated(("claim",), ("filed", "issued")) or has_dated(("statement", "claim"), ("filed", "issued"))
            if supported:
                return {"supported": True, "signals": ["procedural_claim_filing_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["claim_filed_reference_with_date"]}

        if target == "defence_due_or_filed":
            supported = has_dated(("defence",), ("due", "filed", "served")) or has_dated(("defense",), ("due", "filed", "served"))
            if not supported and not requires_date:
                supported = has(("defence",), ("due", "filed", "served")) or has(("defense",), ("due", "filed", "served"))
            if supported:
                return {"supported": True, "signals": ["procedural_defence_deadline_or_filing_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["defence_due_or_filed_evidence"]}

        if target in {"procedural_history", "procedural_milestone"}:
            milestone_required = ("filed", "served", "service", "pleading", "default notice", "settlement", "court filing", "issued")
            for canonical, raw in rows:
                if any(marker in canonical for marker in milestone_required):
                    if self._extract_datetimes(raw) or not requires_date:
                        return {"supported": True, "signals": ["procedural_milestone_responsive_evidence_detected"], "missing": []}
            missing = "procedural_milestone_dated_evidence" if requires_date else "procedural_milestone_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        return {"supported": False, "signals": [], "missing": ["procedural_or_correspondence_responsive_evidence"]}

    def _mitigation_target(self, lowered_query: str) -> str:
        if "how many" in lowered_query and "application" in lowered_query:
            return "application_count"
        if ("when" in lowered_query and "interview" in lowered_query) or ("interview date" in lowered_query):
            return "interview_dates"
        if "alternative employment" in lowered_query or "new employment" in lowered_query:
            return "alternative_employment"
        if re.search(r"\boffers?\b", lowered_query) and any(token in lowered_query for token in ("received", "reject")):
            return "offers"
        if "mitigation evidence" in lowered_query or ("what evidence" in lowered_query and "mitigation" in lowered_query):
            return "mitigation_evidence"
        if "mitigation" in lowered_query and "effort" in lowered_query:
            return "mitigation_efforts"
        if "mitigation" in lowered_query:
            return "mitigation_general"
        return "mitigation_general"

    def _evaluate_employment_mitigation_support(
        self,
        *,
        query: str,
        substantive: Sequence[Mapping[str, str]],
    ) -> dict[str, object]:
        lowered_query = self._canonical_phrase(query)
        target = self._mitigation_target(lowered_query)
        requires_date = "when" in lowered_query
        if not substantive:
            return {"supported": False, "signals": [], "missing": ["employment_mitigation_responsive_evidence"]}

        rows: list[tuple[str, str]] = []
        for item in substantive:
            raw_text = f"{item.get('heading', '')} {item.get('text', '')}".strip()
            canonical = self._canonical_phrase(raw_text)
            if canonical:
                rows.append((canonical, raw_text))

        mitigation_doc_markers = (
            "job search log",
            "mitigation journal",
            "application record",
            "application log",
            "resume",
            "interview invitation",
            "offer letter",
            "employment update",
            "email",
            "correspondence",
            "notes",
        )

        def has(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for canonical, _raw in rows:
                if not all(token in canonical for token in required):
                    continue
                if any_of and not any(token in canonical for token in any_of):
                    continue
                return True
            return False

        def has_dated(required: Sequence[str], any_of: Sequence[str] = ()) -> bool:
            for canonical, raw in rows:
                if not all(token in canonical for token in required):
                    continue
                if any_of and not any(token in canonical for token in any_of):
                    continue
                if self._extract_datetimes(raw):
                    return True
            return False

        if target == "application_count":
            supported = has(("application",), ("submitted", "applied", "resume", "candidate", "position", "job"))
            if supported:
                return {"supported": True, "signals": ["employment_mitigation_application_record_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["job_application_record_evidence"]}

        if target == "interview_dates":
            supported = has_dated(("interview",), ("invited", "scheduled", "conducted", "attended")) if requires_date else has(
                ("interview",),
                ("invited", "scheduled", "conducted", "attended"),
            )
            if supported:
                return {"supported": True, "signals": ["employment_mitigation_interview_evidence_detected"], "missing": []}
            missing = "interview_dated_evidence" if requires_date else "interview_record_evidence"
            return {"supported": False, "signals": [], "missing": [missing]}

        if target == "offers":
            supported = has(("offer",), ("received", "extended", "rejected", "accepted"))
            if supported:
                return {"supported": True, "signals": ["employment_mitigation_offer_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["offer_received_or_rejected_evidence"]}

        if target == "alternative_employment":
            supported = has(("employment",), ("new", "alternative", "accepted", "started", "start date")) or has(
                ("position",),
                ("accepted", "new employer", "start date"),
            )
            if supported:
                return {"supported": True, "signals": ["employment_mitigation_alternative_employment_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["alternative_or_new_employment_evidence"]}

        if target == "mitigation_evidence":
            for canonical, _raw in rows:
                if "mitigation" in canonical and any(marker in canonical for marker in mitigation_doc_markers):
                    return {"supported": True, "signals": ["employment_mitigation_evidence_source_detected"], "missing": []}
                if any(marker in canonical for marker in mitigation_doc_markers) and any(
                    token in canonical for token in ("application", "interview", "offer", "job search", "new employment")
                ):
                    return {"supported": True, "signals": ["employment_mitigation_evidence_source_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["mitigation_responsive_evidence_sources"]}

        if target in {"mitigation_efforts", "mitigation_general"}:
            supported = (
                has(("mitigation",), ("effort", "job search", "application", "interview", "offer", "employment"))
                or has(("job search",), ("application", "interview", "offer"))
                or has(("application",), ("submitted", "position", "job"))
            )
            if supported:
                return {"supported": True, "signals": ["employment_mitigation_efforts_evidence_detected"], "missing": []}
            return {"supported": False, "signals": [], "missing": ["mitigation_efforts_or_job_search_evidence"]}

        return {"supported": False, "signals": [], "missing": ["employment_mitigation_responsive_evidence"]}

    def _evaluate_chronology_support(
        self,
        *,
        query: str,
        substantive: Sequence[Mapping[str, str]],
    ) -> dict[str, object]:
        events = self._extract_chronology_events(substantive)
        if not events:
            return {
                "supported": False,
                "signals": [],
                "missing": ["chronology_responsive_dated_event_evidence"],
            }

        lowered_query = self._canonical_phrase(query)
        signals = ["chronology_responsive_dated_event_evidence_detected"]

        if "what happened first" in lowered_query or "first event" in lowered_query:
            if len(events) >= 2:
                return {"supported": True, "signals": [*signals, "chronology_ordering_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["multiple_dated_events_for_ordering"]}

        if "what happened last" in lowered_query or "last event" in lowered_query:
            if len(events) >= 2:
                return {"supported": True, "signals": [*signals, "chronology_ordering_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["multiple_dated_events_for_ordering"]}

        if "after" in lowered_query:
            anchor = self._extract_after_before_anchor(lowered_query, relation="after")
            if anchor and self._supports_relative_events(events, anchor, relation="after"):
                return {"supported": True, "signals": [*signals, "after_event_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["dated_anchor_and_subsequent_event_support"]}

        if "before" in lowered_query:
            anchor = self._extract_after_before_anchor(lowered_query, relation="before")
            if anchor and self._supports_relative_events(events, anchor, relation="before"):
                return {"supported": True, "signals": [*signals, "before_event_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["dated_anchor_and_prior_event_support"]}

        if "between" in lowered_query:
            query_dates = self._extract_datetimes(query)
            if len(query_dates) >= 2 and self._has_events_within_range(events, min(query_dates), max(query_dates)):
                return {"supported": True, "signals": [*signals, "date_range_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["dated_events_within_requested_range"]}

        if "all dated events" in lowered_query or "timeline" in lowered_query or "chronology" in lowered_query:
            return {"supported": True, "signals": [*signals, "timeline_event_set_supported"], "missing": []}

        if (
            "employment start" in lowered_query
            or "employment began" in lowered_query
            or "employment commence" in lowered_query
            or "start date" in lowered_query
        ):
            if self._has_event_with_markers(events, ("employment", "start"), optional=("commence", "effective", "begin")):
                return {"supported": True, "signals": [*signals, "employment_start_date_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["employment_start_dated_event_evidence"]}

        if "termination notice" in lowered_query or ("termination" in lowered_query and "notice" in lowered_query):
            if self._has_event_with_markers(events, ("termination", "notice"), optional=("letter", "email", "served")):
                return {"supported": True, "signals": [*signals, "termination_notice_date_supported"], "missing": []}
            return {"supported": False, "signals": signals, "missing": ["termination_notice_dated_event_evidence"]}

        return {"supported": True, "signals": signals, "missing": []}

    def _extract_chronology_events(self, substantive: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for item in substantive:
            text = (item.get("text") or "").strip()
            heading = (item.get("heading") or "").strip()
            if not text:
                continue
            for match in self._iter_date_matches(text):
                dt = self._parse_datetime(match.group(0))
                if dt is None:
                    continue
                start = max(0, match.start() - 90)
                end = min(len(text), match.end() + 120)
                snippet = " ".join(text[start:end].split())
                event_text = f"{heading}. {snippet}".strip(". ").lower()
                events.append({"datetime": dt, "event_text": event_text})
        events.sort(key=lambda item: cast(datetime, item["datetime"]))
        return events

    def _iter_date_matches(self, text: str) -> list[re.Match[str]]:
        pattern = re.compile(
            r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
            flags=re.IGNORECASE,
        )
        return list(pattern.finditer(text or ""))

    def _parse_datetime(self, raw: str) -> datetime | None:
        value = (raw or "").strip()
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _extract_datetimes(self, text: str) -> list[datetime]:
        values: list[datetime] = []
        for match in self._iter_date_matches(text):
            dt = self._parse_datetime(match.group(0))
            if dt is not None:
                values.append(dt)
        return values

    def _extract_after_before_anchor(self, query: str, *, relation: str) -> str:
        pattern = re.compile(rf"\b{relation}\s+(.+?)(?:\?|$)")
        match = pattern.search(query)
        if not match:
            return ""
        anchor = self._canonical_phrase(match.group(1))
        anchor = re.sub(r"\b(the|a|an)\b", " ", anchor)
        return " ".join(anchor.split())

    def _supports_relative_events(self, events: Sequence[Mapping[str, object]], anchor: str, *, relation: str) -> bool:
        if not anchor:
            return False
        anchor_events = [event for event in events if anchor in str(event.get("event_text", ""))]
        if not anchor_events:
            return False
        anchor_dt = min(cast(datetime, event["datetime"]) for event in anchor_events)
        if relation == "after":
            return any(cast(datetime, event["datetime"]) > anchor_dt for event in events)
        return any(cast(datetime, event["datetime"]) < anchor_dt for event in events)

    def _has_events_within_range(
        self,
        events: Sequence[Mapping[str, object]],
        start: datetime,
        end: datetime,
    ) -> bool:
        return any(start <= cast(datetime, event["datetime"]) <= end for event in events)

    def _has_event_with_markers(
        self,
        events: Sequence[Mapping[str, object]],
        required: Sequence[str],
        optional: Sequence[str] = (),
    ) -> bool:
        required_lower = [part.lower() for part in required]
        optional_lower = [part.lower() for part in optional]
        for event in events:
            text = str(event.get("event_text") or "")
            if not all(part in text for part in required_lower):
                continue
            if optional_lower and not any(part in text for part in optional_lower):
                continue
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
            return ("between", "party", "parties", "by and between", "made effective")
        if "employer" in lowered:
            return ("employer", "company", "between", "employment agreement", "by and between", "made effective")
        if "employee" in lowered:
            return ("employee", "between", "employment agreement", "by and between", "made effective")
        if "which company" in lowered and "agreement" in lowered:
            return ("company", "agreement", "between", "by and between", "made effective")
        if "notice period" in lowered:
            return ("notice period", "days notice", "written notice")
        return tuple(self._query_terms(query))

    def _is_party_role_entity_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(note == "legal_question_family:party_role_entity" for note in (query_understanding.routing_notes or [])):
            return True

        lowered = self._canonical_phrase(query)
        patterns = (
            r"\bwho\s+is\s+the\s+employer\b",
            r"\bwho\s+is\s+the\s+employee\b",
            r"\bwho\s+are\s+the\s+parties\b",
            r"\bidentify\s+(?:the\s+)?parties\b",
            r"\bwhich\s+company\s+is\s+this\s+agreement\s+for\b",
            r"\bwho\s+is\s+the\s+hiring\s+company\b",
            r"\bwhich\s+party\s+is\s+the\s+company\s+side\b",
            r"\bwhich\s+party\s+is\s+the\s+individual\s+side\b",
            r"\bis\s+this\s+agreement\s+between\b",
            r"\bis\s+(?:this|the)\s+agreement\s+with\b",
            r"\bis\s+(?:this|the)\s+agreement\s+for\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _is_matter_metadata_query(
        self,
        query: str,
        query_understanding: QueryUnderstandingResult,
    ) -> bool:
        if any(note == "legal_question_family:matter_document_metadata" for note in (query_understanding.routing_notes or [])):
            return True

        lowered = self._canonical_phrase(query)
        patterns = (
            r"\bwhat\s+is\s+the\s+file\s+number\b",
            r"\b(?:what|which)\s+jurisdiction\s+applies\b",
            r"\b(?:what|which)\s+court\s+is\s+involved\b",
            r"\bwho\s+is\s+the\s+client\b",
            r"\bwhat\s+is\s+the\s+(?:case|matter)\s+name\b",
            r"\bwhat\s+is\s+this\s+(?:matter|document)\s+about\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _is_company_side_query(self, lowered_query: str) -> bool:
        return any(
            re.search(pattern, lowered_query)
            for pattern in (
                r"\bwhich\s+company\s+is\s+this\s+agreement\s+for\b",
                r"\bwho\s+is\s+the\s+hiring\s+company\b",
                r"\bwhich\s+party\s+is\s+the\s+company\s+side\b",
            )
        )

    def _is_individual_side_query(self, lowered_query: str) -> bool:
        return bool(re.search(r"\bwhich\s+party\s+is\s+the\s+individual\s+side\b", lowered_query))

    def _is_party_set_query(self, lowered_query: str) -> bool:
        patterns = (
            r"\bwho\s+are\s+the\s+parties\b",
            r"\bwho\s+are\s+the\s+parties\s+involved\b",
            r"\bidentify\s+(?:the\s+)?parties\b",
            r"\bidentify\s+the\s+parties\s+involved\b",
            r"\bname\s+(?:the\s+)?parties\b",
            r"\blist\s+(?:the\s+)?parties\b",
        )
        return any(re.search(pattern, lowered_query) for pattern in patterns)

    def _has_matter_metadata_evidence(self, substantive: Sequence[Mapping[str, Any]], query: str) -> bool:
        lowered_query = self._canonical_phrase(query)
        target = self._metadata_target(lowered_query)
        if target == "unknown":
            return False

        heading_markers = (
            "caption",
            "header",
            "introduction",
            "matter information",
            "case information",
            "style of cause",
            "court",
            "file",
        )
        for item in substantive:
            text = (item.get("text") or "").strip().lower()
            heading = (item.get("heading") or "").strip().lower()
            source = (item.get("source_name") or "").strip().lower()
            metadata_map = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
            metadata_text = " ".join(self._metadata_values(metadata_map)).lower()
            haystack = "\n".join(part for part in (heading, text, source, metadata_text) if part).strip()
            if not haystack:
                continue

            has_metadata_container_signal = any(marker in haystack for marker in heading_markers)
            if target == "file_number":
                if re.search(r"\b(file|court file|docket|case)\s*(no\.?|number)\b", haystack):
                    return True
            elif target == "jurisdiction":
                if re.search(r"\bjurisdiction\b\s*[:\-]", haystack):
                    return True
                if "jurisdiction" in haystack and ("exclusive" in haystack or "governing" in haystack or "applies" in haystack):
                    return True
                if re.search(r"\bgoverned\s+by\s+the\s+laws?\s+of\b", haystack):
                    return True
            elif target == "court":
                if re.search(r"\b(supreme|district|chancery|appeal|high)\s+court\b", haystack):
                    return True
                if "court" in haystack and has_metadata_container_signal:
                    return True
            elif target == "client":
                if re.search(r"\bclient\b\s*[:\-]", haystack):
                    return True
                if "client" in haystack and has_metadata_container_signal:
                    return True
            elif target == "case_or_matter_name":
                if re.search(r"\b(case|matter)\s+name\b", haystack):
                    return True
                if " v. " in f" {haystack} " and has_metadata_container_signal:
                    return True
            elif target == "matter_about":
                if re.search(r"\b(subject|re|regarding|matter)\b\s*[:\-]", haystack):
                    return True
                if "this matter concerns" in haystack or "this matter is about" in haystack:
                    return True
        return False

    def _metadata_target(self, lowered_query: str) -> str:
        if "file number" in lowered_query or "docket number" in lowered_query or "court file" in lowered_query:
            return "file_number"
        if "jurisdiction" in lowered_query:
            return "jurisdiction"
        if "court" in lowered_query:
            return "court"
        if "client" in lowered_query:
            return "client"
        if "case name" in lowered_query or "matter name" in lowered_query:
            return "case_or_matter_name"
        if "matter about" in lowered_query or "document about" in lowered_query:
            return "matter_about"
        return "unknown"

    def _metadata_values(self, metadata: Mapping[str, object]) -> list[str]:
        values: list[str] = []
        for key, value in metadata.items():
            key_text = str(key).strip().lower()
            if key_text not in {
                "matter_name",
                "case_name",
                "client",
                "client_name",
                "file_number",
                "court_file_number",
                "docket_number",
                "jurisdiction",
                "court",
                "document_type",
                "subject",
                "matter",
                "title",
            }:
                continue
            if isinstance(value, str) and value.strip():
                values.append(f"{key_text}: {value.strip()}")
        return values

    def _evaluate_party_role_support(self, substantive: Sequence[Mapping[str, Any]], query: str) -> dict[str, object]:
        role_assignment, diagnostics = self._resolve_party_roles_from_intro(substantive)
        diagnostic_signals = diagnostics["signals"]
        diagnostic_missing = diagnostics["missing"]
        role_resolution_debug = diagnostics.get("debug")

        def with_debug(payload: dict[str, object]) -> dict[str, object]:
            payload["debug"] = role_resolution_debug
            return payload

        if role_assignment is None:
            return with_debug({
                "supported": False,
                "signals": list(diagnostic_signals),
                "missing": ["party_role_assignment_unresolved", *diagnostic_missing],
            })

        lowered_query = self._canonical_phrase(query)
        if "who is the employer" in lowered_query:
            if role_assignment.employer:
                return with_debug({
                    "supported": True,
                    "signals": [
                        *diagnostic_signals,
                        "party_role_responsive_evidence_detected",
                        "party_role_assignment_resolved",
                        "employer_role_assignment_resolved",
                    ],
                    "missing": [],
                })
            return with_debug({
                "supported": False,
                "signals": [*diagnostic_signals, "party_role_assignment_resolved", "party_role_resolution_outcome:ambiguous"],
                "missing": ["employer_role_assignment_missing_or_ambiguous", *diagnostic_missing],
            })

        if "who is the employee" in lowered_query:
            if role_assignment.employee:
                return with_debug({
                    "supported": True,
                    "signals": [
                        *diagnostic_signals,
                        "party_role_responsive_evidence_detected",
                        "party_role_assignment_resolved",
                        "employee_role_assignment_resolved",
                    ],
                    "missing": [],
                })
            return with_debug({
                "supported": False,
                "signals": [*diagnostic_signals, "party_role_assignment_resolved", "party_role_resolution_outcome:ambiguous"],
                "missing": ["employee_role_assignment_missing_or_ambiguous", *diagnostic_missing],
            })

        if self._is_party_set_query(lowered_query):
            if len(role_assignment.parties) >= 2:
                return with_debug({
                    "supported": True,
                    "signals": [
                        *diagnostic_signals,
                        "party_role_responsive_evidence_detected",
                        "party_role_assignment_resolved",
                        "party_set_resolved",
                    ],
                    "missing": [],
                })
            return with_debug({
                "supported": False,
                "signals": [*diagnostic_signals, "party_role_assignment_resolved", "party_role_resolution_outcome:ambiguous"],
                "missing": ["party_set_incomplete", *diagnostic_missing],
            })

        parsed_verification = self._parse_party_verification_query_entities(lowered_query)
        if parsed_verification is not None:
            if cast(bool, parsed_verification["ambiguous"]):
                return with_debug({
                    "supported": False,
                    "signals": [*diagnostic_signals, "party_role_assignment_resolved", "party_role_resolution_outcome:ambiguous"],
                    "missing": ["query_entity_set_incomplete_or_ambiguous", *diagnostic_missing],
                })
            comparison = self._compare_query_entities_against_extracted_parties(
                verification_targets=cast(tuple[str, ...], parsed_verification["targets"]),
                extracted_parties=role_assignment.parties,
            )
            comparison["signals"] = [*diagnostic_signals, *cast(list[str], comparison["signals"])]
            comparison["missing"] = [*cast(list[str], comparison["missing"]), *diagnostic_missing]
            return with_debug(comparison)

        if self._is_company_side_query(lowered_query):
            company = role_assignment.company_side_party or role_assignment.employer or self._pick_company_party(role_assignment.parties)
            if company:
                return with_debug({
                    "supported": True,
                    "signals": [
                        *diagnostic_signals,
                        "party_role_responsive_evidence_detected",
                        "party_role_assignment_resolved",
                        "company_side_party_identified",
                    ],
                    "missing": [],
                })
            return with_debug({
                "supported": False,
                "signals": [*diagnostic_signals, "party_role_assignment_resolved", "party_role_resolution_outcome:ambiguous"],
                "missing": ["company_side_party_not_identified", *diagnostic_missing],
            })
        if self._is_individual_side_query(lowered_query):
            individual = role_assignment.individual_side_party or role_assignment.employee or pick_individual_party(role_assignment.parties)
            if individual:
                return with_debug({
                    "supported": True,
                    "signals": [
                        *diagnostic_signals,
                        "party_role_responsive_evidence_detected",
                        "party_role_assignment_resolved",
                        "individual_side_party_identified",
                    ],
                    "missing": [],
                })
            return with_debug({
                "supported": False,
                "signals": [*diagnostic_signals, "party_role_assignment_resolved", "party_role_resolution_outcome:ambiguous"],
                "missing": ["individual_side_party_not_identified", *diagnostic_missing],
            })

        return with_debug({
            "supported": False,
            "signals": list(diagnostic_signals),
            "missing": ["party_role_query_not_supported", *diagnostic_missing],
        })

    def _extract_party_verification_targets(self, lowered_query: str) -> tuple[str, ...] | None:
        parsed = self._parse_party_verification_query_entities(lowered_query)
        if parsed is None or parsed["ambiguous"]:
            return None
        return cast(tuple[str, ...], parsed["targets"])

    def _parse_party_verification_query_entities(self, lowered_query: str) -> dict[str, object] | None:
        return parse_party_verification_query_entities(lowered_query)

    def _compare_query_entities_against_extracted_parties(
        self,
        *,
        verification_targets: tuple[str, ...],
        extracted_parties: Sequence[str],
    ) -> dict[str, object]:
        comparison = compare_query_entities_against_extracted_parties(
            verification_targets=verification_targets,
            extracted_parties=extracted_parties,
        )
        if comparison["status"] == "incomplete_party_set":
            return {
                "supported": False,
                "signals": ["party_role_assignment_resolved"],
                "missing": ["extracted_party_set_incomplete_or_ambiguous"],
            }
        if comparison["status"] == "query_ambiguous":
            return {
                "supported": False,
                "signals": ["party_role_assignment_resolved"],
                "missing": ["query_entity_set_incomplete_or_ambiguous"],
            }
        if comparison["status"] == "matched":
            return {
                "supported": True,
                "signals": [
                    "party_role_responsive_evidence_detected",
                    "party_role_assignment_resolved",
                    "agreement_between_pair_confirmed_from_extracted_parties",
                    "agreement_between_query_entity_set_matched_extracted_party_set",
                ],
                "missing": [],
            }

        return {
            "supported": False,
            "signals": ["party_role_assignment_resolved"],
            "missing": ["requested_query_entity_set_not_supported_by_extracted_party_set"],
        }

    def _resolve_party_roles_from_intro(
        self, substantive: Sequence[Mapping[str, Any]]
    ) -> tuple[_PartyRoleAssignment | None, dict[str, Any]]:
        checked_parent_ids: list[str] = []
        intro_signal_parent_ids: list[str] = []
        checked_parent_previews: list[PartyRoleParentChunkDebugPreview] = []
        for item in substantive:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            parent_id = str(item.get("parent_chunk_id") or "").strip()
            if parent_id:
                checked_parent_ids.append(parent_id)
            intro_pattern_detected = has_intro_role_signal(text)
            if intro_pattern_detected:
                if parent_id:
                    intro_signal_parent_ids.append(parent_id)
            parsed = extract_intro_party_role_assignment(text)
            checked_parent_previews.append(
                self._build_party_role_parent_preview(
                    item=item,
                    text=text,
                    intro_pattern_detected=intro_pattern_detected,
                    resolver_considered_usable_intro_text=parsed is not None,
                )
            )
            if parsed is not None:
                debug = PartyRoleResolutionDebug(
                    party_role_resolution_checked_parent_count=len(checked_parent_ids),
                    party_role_resolution_checked_parent_ids=checked_parent_ids,
                    party_role_resolution_intro_pattern_parent_ids=intro_signal_parent_ids,
                    party_role_resolution_debug_outcome="resolved",
                    party_role_resolution_debug_reason="resolver_found_intro_role_assignment",
                    checked_parent_previews=checked_parent_previews,
                )
                diagnostics = {
                    "signals": [
                        "party_role_resolution_invoked",
                        f"party_role_resolution_checked_parent_chunks:{len(checked_parent_ids)}",
                        f"party_role_resolution_checked_parent_ids:{','.join(checked_parent_ids) or 'none'}",
                        f"party_role_resolution_intro_pattern_parent_ids:{','.join(intro_signal_parent_ids) or 'none'}",
                        "party_role_resolution_outcome:resolved",
                    ],
                    "missing": [],
                    "debug": debug,
                }
                return (
                    _PartyRoleAssignment(
                        parties=parsed.parties,
                        employer=parsed.employer,
                        employee=parsed.employee,
                        company_side_party=parsed.company_side_party,
                        individual_side_party=parsed.individual_side_party,
                    ),
                    diagnostics,
                )
        not_found_reason = (
            "party_role_resolution_not_found_reason:intro_text_present_parser_miss"
            if intro_signal_parent_ids
            else "party_role_resolution_not_found_reason:intro_text_absent_from_runtime_context"
        )
        debug = PartyRoleResolutionDebug(
            party_role_resolution_checked_parent_count=len(checked_parent_ids),
            party_role_resolution_checked_parent_ids=checked_parent_ids,
            party_role_resolution_intro_pattern_parent_ids=intro_signal_parent_ids,
            party_role_resolution_debug_outcome="not_found",
            party_role_resolution_debug_reason=not_found_reason,
            checked_parent_previews=checked_parent_previews,
        )
        diagnostics = {
            "signals": [
                "party_role_resolution_invoked",
                f"party_role_resolution_checked_parent_chunks:{len(checked_parent_ids)}",
                f"party_role_resolution_checked_parent_ids:{','.join(checked_parent_ids) or 'none'}",
                f"party_role_resolution_intro_pattern_parent_ids:{','.join(intro_signal_parent_ids) or 'none'}",
                "party_role_resolution_outcome:not_found",
            ],
            "missing": [not_found_reason],
            "debug": debug,
        }
        return None, diagnostics

    def _build_party_role_parent_preview(
        self,
        *,
        item: Mapping[str, Any],
        text: str,
        intro_pattern_detected: bool,
        resolver_considered_usable_intro_text: bool,
    ) -> PartyRoleParentChunkDebugPreview:
        normalized = text.strip()
        lowered = normalized.lower()
        preview_start = normalized[:PARTY_ROLE_PREVIEW_START_CHARS]
        preview_end = normalized[-PARTY_ROLE_PREVIEW_END_CHARS:] if normalized else ""
        return PartyRoleParentChunkDebugPreview(
            parent_chunk_id=(str(item.get("parent_chunk_id") or "").strip() or None),
            heading=(str(item.get("heading") or "").strip() or None),
            source_name=(str(item.get("source_name") or "").strip() or None),
            document_id=(str(item.get("document_id") or "").strip() or None),
            text_length_chars=len(normalized),
            preview_start=preview_start,
            preview_end=preview_end,
            contains_between_keyword=bool(re.search(r"\bbetween\b", lowered)),
            contains_and_keyword=bool(re.search(r"\band\b", lowered)),
            contains_employer_label=bool(re.search(r"\bemployer\b", lowered)),
            contains_employee_label=bool(re.search(r"\bemployee\b", lowered)),
            contains_role_parenthetical=bool(re.search(r"\((?:[^)]*\b(?:employer|employee|company)\b[^)]*)\)", normalized, flags=re.IGNORECASE)),
            intro_pattern_detected=intro_pattern_detected,
            resolver_considered_usable_intro_text=resolver_considered_usable_intro_text,
        )

    def _extract_intro_assignment(self, text: str) -> _PartyRoleAssignment | None:
        parsed = extract_intro_party_role_assignment(text)
        if parsed is None:
            return None
        return _PartyRoleAssignment(
            parties=parsed.parties,
            employer=parsed.employer,
            employee=parsed.employee,
            company_side_party=parsed.company_side_party,
            individual_side_party=parsed.individual_side_party,
        )

    def _detect_inline_role(self, value: str) -> str | None:
        lowered = value.lower()
        if "employer" in lowered or "company" in lowered:
            return "employer"
        if "employee" in lowered:
            return "employee"
        return None

    def _infer_employer_employee(self, first: str, second: str, lowered_text: str) -> tuple[str | None, str | None]:
        if "employment agreement" not in lowered_text:
            return None, None
        first_is_org = self._looks_like_organization(first)
        second_is_org = self._looks_like_organization(second)
        if first_is_org == second_is_org:
            return None, None
        if first_is_org:
            return first, second
        return second, first

    def _clean_party_name(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"\([^)]*\)", " ", value)
        cleaned = re.sub(r"\b(?:the\s+)?(employer|employee|company|party|parties)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[\"'“”]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        return cleaned or None

    def _looks_like_organization(self, value: str) -> bool:
        lowered = value.lower()
        org_tokens = (
            " inc",
            " llc",
            " ltd",
            " corp",
            " corporation",
            " company",
            " co.",
            " limited",
            " plc",
            " lp",
            " llp",
            " holdings",
            " group",
        )
        if any(token in f" {lowered}" for token in org_tokens):
            return True
        return bool(re.search(r"\b[a-z]+\s+(?:inc\.?|llc|ltd\.?|corp\.?|company|holdings|group)\b", lowered))

    def _pick_company_party(self, parties: Sequence[str]) -> str | None:
        for party in parties:
            if self._looks_like_organization(party):
                return party
        return None

    def _is_placeholder_party(self, value: str) -> bool:
        lowered = value.lower().strip()
        placeholders = {
            "company",
            "the company",
            "employee",
            "the employee",
            "party",
            "the party",
            "parties",
            "the parties",
            "employer",
            "the employer",
        }
        return lowered in placeholders

    def _normalize_party_text(self, value: str) -> str:
        lowered = (value or "").lower().strip()
        lowered = re.sub(r"[\"'“”]", "", lowered)
        lowered = re.sub(r"\([^)]*\)", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\b(the|this|that|an|a)\b", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip(" ,;:-")

    def _is_usable_party_entity(self, value: str) -> bool:
        normalized = self._normalize_party_text(value)
        if not normalized:
            return False
        if self._is_placeholder_party(normalized):
            return False
        return len(normalized) >= 2

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

    def _has_operational_clause_definition_support(
        self,
        *,
        query: str,
        query_understanding: QueryUnderstandingResult,
        substantive: Sequence[Mapping[str, str]],
        require_label_anchor: bool = False,
    ) -> bool:
        targets = set(self._definition_targets(query))
        hint_candidates = {
            self._canonical_phrase(hint)
            for hint in [*query_understanding.resolved_clause_hints, *query_understanding.resolved_topic_hints]
            if self._canonical_phrase(hint)
        }
        candidates = {self._canonical_phrase(value) for value in targets if self._canonical_phrase(value)}
        candidates.update(hint_candidates)
        if any(len(candidate.split()) >= 2 for candidate in candidates):
            candidates = {candidate for candidate in candidates if len(candidate.split()) >= 2}
        if not candidates:
            return False

        for item in substantive:
            text = (item.get("text") or "").strip()
            if len(text) < self.min_substantive_chars:
                continue
            heading = self._canonical_phrase(item.get("heading") or "")
            if heading and any(self._is_strong_clause_label_match(heading, candidate) for candidate in candidates):
                return True
            if any(self._matches_leading_clause_label(text, candidate) for candidate in candidates):
                return True
            if require_label_anchor:
                continue
            if any(self._matches_clause_topic_in_body(text, candidate) for candidate in candidates):
                return True
        return False

    def _canonical_phrase(self, value: str) -> str:
        lowered = (value or "").lower().strip()
        lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        return " ".join(lowered.split())

    def _is_strong_clause_label_match(self, heading: str, candidate: str) -> bool:
        if not heading or not candidate:
            return False
        if heading == candidate:
            return True
        if heading in candidate or candidate in heading:
            heading_tokens = set(heading.split())
            candidate_tokens = set(candidate.split())
            if not heading_tokens or not candidate_tokens:
                return False
            overlap = len(heading_tokens & candidate_tokens) / max(len(heading_tokens), len(candidate_tokens))
            return overlap >= 0.8
        return False

    def _matches_leading_clause_label(self, text: str, candidate: str) -> bool:
        if not text or not candidate:
            return False
        escaped = re.escape(candidate)
        patterns = (
            rf"^\s*(?:section\s+\d+(?:\.\d+)*)?\s*{escaped}\s*[:\-–]\s+\S+",
            rf"^\s*(?:\d+(?:\.\d+)*)\s+{escaped}\s*[:\-–]\s+\S+",
            rf"^\s*(?:section\s+\d+(?:\.\d+)*)?\s*{escaped}\s*\.\s+\S+",
            rf"^\s*(?:\d+(?:\.\d+)*)\s+{escaped}\s*\.\s+\S+",
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _matches_clause_topic_in_body(self, text: str, candidate: str) -> bool:
        canonical_text = self._canonical_phrase(text)
        if not canonical_text or not candidate:
            return False
        if candidate in canonical_text:
            return True
        candidate_tokens = [token for token in candidate.split() if len(token) >= 3]
        if len(candidate_tokens) < 2:
            return False
        text_tokens = canonical_text.split()
        if len(text_tokens) < 2:
            return False

        def normalize(token: str) -> str:
            if len(token) > 4 and token.endswith("es"):
                return token[:-2]
            if len(token) > 3 and token.endswith("s"):
                return token[:-1]
            return token

        def token_matches(candidate_token: str, text_token: str) -> bool:
            lhs = normalize(candidate_token)
            rhs = normalize(text_token)
            if lhs == rhs:
                return True
            if len(lhs) >= 6 and len(rhs) >= 6:
                return lhs[:6] == rhs[:6]
            return False

        match_positions: list[int] = []
        search_start = 0
        for candidate_token in candidate_tokens:
            found_index: int | None = None
            for index in range(search_start, len(text_tokens)):
                if token_matches(candidate_token, text_tokens[index]):
                    found_index = index
                    break
            if found_index is None:
                return False
            match_positions.append(found_index)
            search_start = found_index + 1

        span = match_positions[-1] - match_positions[0]
        max_span = max(6, len(candidate_tokens) + 3)
        return span <= max_span

    def _field(self, item: object, key: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(key)
        return getattr(item, key, None)


_DEFAULT_ANSWERABILITY_ASSESSOR = AnswerabilityAssessor()
ALLOW_MODERATE_STRENGTH_WHEN_COVERAGE_SUFFICIENT = True


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _combine_coverage_and_strength(
    *,
    query_understanding: QueryUnderstandingResult,
    coverage: CoverageEvaluation,
    strength: EvidenceStrengthEvaluation,
) -> tuple[SupportLevel, bool, bool, bool, InsufficiencyReason | None]:
    """Apply one strict, deterministic answerability policy.

    assess_answerability(...) is a coordinator over coverage + evidence strength:
    - coverage is primary eligibility and cannot be overridden by strength
    - moderate strength follows one explicit policy constant
    - weak strength cannot be silently upgraded to sufficient answerability
    """

    support_level: SupportLevel = coverage.coverage_status
    sufficient_context = False
    should_answer = False
    partially_supported = bool(coverage.partial_coverage)
    insufficiency_reason = _DEFAULT_ANSWERABILITY_ASSESSOR._map_coverage_reason(coverage)

    if not coverage.sufficient_coverage:
        return support_level, sufficient_context, should_answer, partially_supported, insufficiency_reason

    partially_supported = False
    expectation_blocks_answer = query_understanding.answerability_expectation in {"meta_response", "clarification_needed"}

    if strength.evidence_strength == "strong":
        return "sufficient", True, not expectation_blocks_answer, False, None

    if strength.evidence_strength == "moderate":
        if ALLOW_MODERATE_STRENGTH_WHEN_COVERAGE_SUFFICIENT:
            return "sufficient", True, not expectation_blocks_answer, False, None
        return "moderate", False, False, False, "partial_evidence_only"

    if any(
        note == "legal_question_family:employment_contract_lifecycle"
        for note in (query_understanding.routing_notes or [])
    ):
        return "sufficient", True, not expectation_blocks_answer, False, None

    if any(
        note == "legal_question_family:correspondence_litigation_milestone"
        for note in (query_understanding.routing_notes or [])
    ):
        return "sufficient", True, not expectation_blocks_answer, False, None

    if any(
        note == "legal_question_family:financial_entitlement"
        for note in (query_understanding.routing_notes or [])
    ):
        return "sufficient", True, not expectation_blocks_answer, False, None

    if any(
        note == "legal_question_family:policy_issue_spotting"
        for note in (query_understanding.routing_notes or [])
    ):
        return "sufficient", True, not expectation_blocks_answer, False, None

    if any(
        note == "legal_question_family:party_role_entity"
        for note in (query_understanding.routing_notes or [])
    ) and any(
        signal in {
            "party_role_assignment_resolved",
            "employer_role_assignment_resolved",
            "employee_role_assignment_resolved",
            "party_set_resolved",
            "agreement_between_pair_confirmed_from_extracted_parties",
        }
        for signal in (coverage.supporting_signals or [])
    ):
        return "sufficient", True, not expectation_blocks_answer, False, None

    # Explicitly conservative for weak/none structural strength.
    return "weak", False, False, False, "partial_evidence_only"


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

    support_level, sufficient_context, should_answer, partially_supported, insufficiency_reason = _combine_coverage_and_strength(
        query_understanding=query_understanding,
        coverage=coverage,
        strength=strength,
    )

    evidence_notes = _dedupe_preserve_order([
        *list(coverage.supporting_signals),
        f"evidence_strength:{strength.evidence_strength}",
        *(f"strength_signal:{s}" for s in strength.supporting_signals),
        *(f"weakness_signal:{s}" for s in strength.weakness_signals),
    ])
    warnings = _dedupe_preserve_order([*list(coverage.warnings), *list(strength.warnings)])

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
        party_role_resolution_debug=coverage.party_role_resolution_debug,
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
