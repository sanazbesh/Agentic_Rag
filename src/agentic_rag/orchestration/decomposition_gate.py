"""Deterministic gate helper for retrieval-stage query decomposition decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

try:  # pragma: no cover - optional runtime dependency
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - fallback for constrained envs
    from agentic_rag._compat_pydantic import BaseModel, ConfigDict, Field

from agentic_rag.orchestration.query_understanding import QueryUnderstandingResult

DecompositionReason = Literal[
    "comparison_query",
    "multi_intent_conjunction",
    "amendment_vs_base",
    "temporal_relationship",
    "exception_chain",
    "cross_clause_obligation_condition",
    "context_dependent_followup",
    "simple_single_clause_lookup",
]

REASON_PRECEDENCE: tuple[DecompositionReason, ...] = (
    "comparison_query",
    "multi_intent_conjunction",
    "amendment_vs_base",
    "temporal_relationship",
    "exception_chain",
    "cross_clause_obligation_condition",
    "context_dependent_followup",
    "simple_single_clause_lookup",
)

STRONG_REASONS: frozenset[DecompositionReason] = frozenset(
    {"comparison_query", "amendment_vs_base", "exception_chain"}
)
CONSERVATIVE_REASONS: frozenset[DecompositionReason] = frozenset(
    {
        "multi_intent_conjunction",
        "temporal_relationship",
        "cross_clause_obligation_condition",
        "context_dependent_followup",
    }
)


@dataclass(frozen=True)
class _CategoryDefinition:
    label: DecompositionReason
    strong_by_default: bool


CATEGORY_REGISTRY: tuple[_CategoryDefinition, ...] = (
    _CategoryDefinition(label="comparison_query", strong_by_default=True),
    _CategoryDefinition(label="multi_intent_conjunction", strong_by_default=False),
    _CategoryDefinition(label="amendment_vs_base", strong_by_default=True),
    _CategoryDefinition(label="temporal_relationship", strong_by_default=False),
    _CategoryDefinition(label="exception_chain", strong_by_default=True),
    _CategoryDefinition(label="cross_clause_obligation_condition", strong_by_default=False),
    _CategoryDefinition(label="context_dependent_followup", strong_by_default=False),
)

SIMPLE_SINGLE_CLAUSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^what is [\w\s\-]+[?.]?$"),
    re.compile(r"^define [\w\s\-]+[?.]?$"),
    re.compile(r"^what does [\w\s\-]+ mean[?.]?$"),
    re.compile(r"^who is [\w\s\-]+[?.]?$"),
    re.compile(r"^what is the [\w\s\-]+ clause[?.]?$"),
    re.compile(r"^what is the [\w\s\-]+ provision[?.]?$"),
)

NON_TRIGGER_SHALLOW_AND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^what is the [\w\s\-]+ and [\w\s\-]+[?.]?$"),
    re.compile(r"^[\w\s\-]+ and [\w\s\-]+$"),
)

NON_TRIGGER_PURE_DATE_MENTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^what is the effective date(?: in \d{4})?[?.]?$"),
    re.compile(r"^what is the date(?: in \d{4})?[?.]?$"),
    re.compile(r"^when is [\w\s\-]+(?: in \d{4})?[?.]?$"),
)

NON_TRIGGER_VAGUE_FOLLOWUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^what about (that|this|it)(?: [\w\s\-]+)?[?.]?$"),
    re.compile(r"^and (that|this|it)\??$"),
    re.compile(r"^anything else\??$"),
)

COMPARISON_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcompare\b"),
    re.compile(r"\bcomparison\b"),
    re.compile(r"\bdifference(?:s)?\s+between\b"),
    re.compile(r"\bversus\b"),
    re.compile(r"\bvs\.?\b"),
    re.compile(r"\bcompared\s+to\b"),
)

MULTI_INTENT_CONNECTOR_PATTERN = re.compile(r"\b(and|as well as|along with)\b")
MULTI_INTENT_LEGAL_ASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bpart(?:y|ies)\b"),
    re.compile(r"\bobligations?\b"),
    re.compile(r"\bright(?:s)?\b"),
    re.compile(r"\bnotice\b"),
    re.compile(r"\bcure\b"),
    re.compile(r"\btermination\b"),
)
# ordinary list-like lookups should stay conservative
MULTI_INTENT_ORDINARY_COORDINATION_PATTERN = re.compile(
    r"\b(title and date|name and address|title and governing law|date and title)\b"
)

AMENDMENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bamend(?:ment|ed)?\b"),
    re.compile(r"\boriginal agreement\b"),
    re.compile(r"\bearlier version\b"),
    re.compile(r"\blater version\b"),
)
CHANGE_RELATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhow did\b.*\bchange\b"),
    re.compile(r"\bchanged\b"),
    re.compile(r"\bchange(?:d)?\s+from\b"),
    re.compile(r"\bbefore\b.*\bafter\b"),
    re.compile(r"\bprior to\b.*\bafter\b"),
)

TEMPORAL_RELATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bbefore\b.*\bafter\b"),
    re.compile(r"\bhow did\b.*\bchange\b"),
    re.compile(r"\bchange(?:d)?\s+over\s+time\b"),
    re.compile(r"\bearlier\b.*\blater\b"),
)
TEMPORAL_DATE_ONLY_PATTERN = re.compile(
    r"\b(effective date|date|year|years|month|months|day|days|\d{4})\b"
)

EXCEPTION_CHAIN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bexcept as otherwise provided\b"),
    re.compile(r"\bsubject to\b"),
    re.compile(r"\bprovided that\b"),
    re.compile(r"\bnotwithstanding\b"),
    re.compile(r"\bunless\b.+\b"),
    re.compile(r"\bexcept\b.+\b"),
)

CROSS_CLAUSE_PAIR_PATTERNS: tuple[tuple[re.Pattern[str], re.Pattern[str]], ...] = (
    (re.compile(r"\bpart(?:y|ies)\b"), re.compile(r"\bobligations?\b")),
    (re.compile(r"\bright(?:s)?\b"), re.compile(r"\bconditions?\b")),
    (re.compile(r"\btermination\b"), re.compile(r"\bnotice\b")),
    (re.compile(r"\bnotice\b"), re.compile(r"\bcure\b")),
    (re.compile(r"\bindemn(?:ity|ification)\b"), re.compile(r"\blimitation\b")),
    (re.compile(r"\bgoverning law\b"), re.compile(r"\bdispute resolution\b")),
)
CROSS_CLAUSE_STRUCTURE_PATTERN = re.compile(r"\b(clause|clauses|section|agreement|contract)\b")


class GateDecision(BaseModel):
    """Stable, typed decomposition gate output for deterministic routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs_decomposition: bool
    reasons: list[DecompositionReason] = Field(default_factory=list)


def _normalize(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


def _matches_any(normalized: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(normalized) for pattern in patterns)


def _is_simple_single_clause_lookup(normalized: str) -> bool:
    return any(pattern.match(normalized) for pattern in SIMPLE_SINGLE_CLAUSE_PATTERNS)


def _has_non_trigger_protection(normalized: str) -> bool:
    protections = (
        NON_TRIGGER_SHALLOW_AND_PATTERNS
        + NON_TRIGGER_PURE_DATE_MENTION_PATTERNS
        + NON_TRIGGER_VAGUE_FOLLOWUP_PATTERNS
    )
    return any(pattern.match(normalized) for pattern in protections)


def _detect_category_labels(
    normalized: str,
    query_context: Mapping[str, Any] | None,
    query_understanding: QueryUnderstandingResult | None,
) -> set[DecompositionReason]:
    labels: set[DecompositionReason] = set()

    has_comparison = _matches_any(normalized, COMPARISON_PATTERNS)
    if has_comparison:
        labels.add("comparison_query")

    has_connector = bool(MULTI_INTENT_CONNECTOR_PATTERN.search(normalized))
    legal_ask_hits = sum(1 for pattern in MULTI_INTENT_LEGAL_ASK_PATTERNS if pattern.search(normalized))
    if (
        has_connector
        and legal_ask_hits >= 2
        and not MULTI_INTENT_ORDINARY_COORDINATION_PATTERN.search(normalized)
    ):
        labels.add("multi_intent_conjunction")

    has_amendment = _matches_any(normalized, AMENDMENT_PATTERNS)
    has_change_relation = _matches_any(normalized, CHANGE_RELATION_PATTERNS)
    if has_amendment and (has_change_relation or has_comparison):
        labels.add("amendment_vs_base")

    has_temporal_relation = _matches_any(normalized, TEMPORAL_RELATION_PATTERNS)
    date_only_signal = bool(TEMPORAL_DATE_ONLY_PATTERN.search(normalized)) and not has_temporal_relation
    if has_temporal_relation and not date_only_signal:
        labels.add("temporal_relationship")

    if _matches_any(normalized, EXCEPTION_CHAIN_PATTERNS):
        labels.add("exception_chain")

    has_cross_clause_pair = any(
        left.search(normalized) and right.search(normalized)
        for left, right in CROSS_CLAUSE_PAIR_PATTERNS
    )
    if has_cross_clause_pair and CROSS_CLAUSE_STRUCTURE_PATTERN.search(normalized):
        labels.add("cross_clause_obligation_condition")

    is_context_dependent = bool(query_understanding and query_understanding.is_context_dependent)
    used_context = bool(query_context and query_context.get("used_conversation_context"))
    unresolved = bool(query_context and query_context.get("unresolved_references"))
    if is_context_dependent and (used_context or unresolved) and labels.intersection(STRONG_REASONS):
        labels.add("context_dependent_followup")

    return labels


def _ordered_reasons(labels: set[DecompositionReason]) -> list[DecompositionReason]:
    return [label for label in REASON_PRECEDENCE if label in labels]


def decide_decomposition_need(
    query: str,
    query_context: Mapping[str, Any] | None = None,
    query_understanding: QueryUnderstandingResult | None = None,
) -> GateDecision:
    """Conservative deterministic helper for deciding decomposition need.

    Final v1 rule:
    - `True` only when one or more explicit strong triggers are present.
    - Conservative/supporting labels do not independently flip to `True`.
    """

    normalized = _normalize(query)
    if not normalized:
        return GateDecision(needs_decomposition=False, reasons=["simple_single_clause_lookup"])

    detected = _detect_category_labels(
        normalized=normalized,
        query_context=query_context,
        query_understanding=query_understanding,
    )
    ordered = _ordered_reasons(detected)

    if any(label in STRONG_REASONS for label in detected):
        return GateDecision(needs_decomposition=True, reasons=ordered)

    has_negative_signal = _is_simple_single_clause_lookup(normalized) or _has_non_trigger_protection(
        normalized
    )
    if has_negative_signal or (query_understanding and query_understanding.may_need_decomposition):
        return GateDecision(needs_decomposition=False, reasons=["simple_single_clause_lookup"])

    return GateDecision(needs_decomposition=False, reasons=ordered)
