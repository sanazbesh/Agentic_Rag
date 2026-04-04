"""Deterministic gate helper for retrieval-stage query decomposition decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
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


class GateDecision(BaseModel):
    """Stable, typed decomposition gate output for deterministic routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    needs_decomposition: bool
    reasons: list[DecompositionReason] = Field(default_factory=list)


def decide_decomposition_need(
    query: str,
    query_context: Mapping[str, Any] | None = None,
    query_understanding: QueryUnderstandingResult | None = None,
) -> GateDecision:
    """Conservative deterministic helper for deciding decomposition need.

    Final v1 rule:
    - `True` only when one or more explicit strong triggers are present.
    - weak/ambiguous inputs (including follow-up status alone) default to `False`.
    """

    normalized = " ".join((query or "").strip().lower().split())
    if not normalized:
        return GateDecision(needs_decomposition=False, reasons=["simple_single_clause_lookup"])

    padded = f" {normalized} "
    strong_reasons: set[DecompositionReason] = set()

    comparison_markers = ("compare", "comparison", "difference", "differ", "versus", "vs ")
    conjunction_markers = (" and ", " as well as ", " along with ")
    amendment_markers = ("amendment", "amended", "original agreement", "base agreement")
    change_markers = ("change", "changed", "modified", "modification")
    temporal_markers = (
        "before and after",
        "over time",
        "since",
        "prior to",
        "after",
        "timeline",
        "change over time",
    )
    exception_markers = ("unless", "except", "subject to", "provided that", "notwithstanding")
    entity_markers = ("party", "parties", "buyer", "seller", "licensor", "licensee", "landlord", "tenant")
    obligation_markers = (
        "obligation",
        "obligations",
        "duty",
        "duties",
        "right",
        "rights",
        "condition",
        "conditions",
        "notice",
        "cure",
        "termination",
    )
    cross_clause_markers = ("clause", "clauses", "section", "agreement", "contract", "obligations")

    has_comparison = any(marker in normalized for marker in comparison_markers)
    if has_comparison:
        strong_reasons.add("comparison_query")

    legal_ask_markers = ("obligation", "obligations", "rights", "notice", "cure", "termination", "confidentiality", "law")
    has_multi_intent = any(marker in padded for marker in conjunction_markers) and (
        has_comparison
        or bool(re.search(r"\b(what|who|when|where|how|are there|does|did)\b", normalized))
        or sum(1 for marker in legal_ask_markers if marker in normalized) >= 2
    )
    if has_multi_intent:
        strong_reasons.add("multi_intent_conjunction")

    has_amendment = any(marker in normalized for marker in amendment_markers)
    has_change = any(marker in normalized for marker in change_markers)
    if has_amendment and (has_change or has_comparison):
        strong_reasons.add("amendment_vs_base")

    has_temporal = any(marker in normalized for marker in temporal_markers)
    if has_temporal and (has_change or has_comparison or has_amendment):
        strong_reasons.add("temporal_relationship")

    if any(marker in normalized for marker in exception_markers):
        strong_reasons.add("exception_chain")

    if (
        any(marker in normalized for marker in entity_markers)
        and any(marker in normalized for marker in obligation_markers)
        and any(marker in normalized for marker in cross_clause_markers)
    ):
        strong_reasons.add("cross_clause_obligation_condition")

    is_context_dependent = bool(query_understanding and query_understanding.is_context_dependent)
    used_context = bool(query_context and query_context.get("used_conversation_context"))
    unresolved = bool(query_context and query_context.get("unresolved_references"))
    if is_context_dependent and (used_context or unresolved) and has_multi_intent:
        strong_reasons.add("context_dependent_followup")

    ordered_strong_reasons = [label for label in REASON_PRECEDENCE if label in strong_reasons]
    if ordered_strong_reasons:
        return GateDecision(needs_decomposition=True, reasons=ordered_strong_reasons)

    simple_single_clause_patterns = (
        r"^what is [\w\s\-]+[?.]?$",
        r"^define [\w\s\-]+[?.]?$",
        r"^what is the [\w\s\-]+ clause[?.]?$",
    )
    explicit_simple_lookup = any(re.match(pattern, normalized) for pattern in simple_single_clause_patterns)
    if explicit_simple_lookup or (query_understanding and query_understanding.may_need_decomposition):
        return GateDecision(needs_decomposition=False, reasons=["simple_single_clause_lookup"])

    return GateDecision(needs_decomposition=False, reasons=[])
