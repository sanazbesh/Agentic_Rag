"""Deterministic citation quality checks for legal RAG eval cases.

This grader intentionally evaluates citation behavior only:
- citation presence
- citation relevance
- citation support match
- unused citation rate

It does not perform semantic answer-correctness grading or model-based judgment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CitationMetricResult:
    """One machine-readable citation metric outcome."""

    metric_name: str
    value: float | None
    passed: bool | None
    details: dict[str, Any]
    note: str | None = None


@dataclass(frozen=True, slots=True)
class CitationEvaluationResult:
    """Per-case citation evaluator output for offline runs."""

    evaluator_name: str
    case_id: str
    case_family: str
    passed: bool
    metrics: list[CitationMetricResult]
    notes: list[str]
    aggregation_fields: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "case_id": self.case_id,
            "case_family": self.case_family,
            "passed": self.passed,
            "metrics": [asdict(metric) for metric in self.metrics],
            "notes": list(self.notes),
            "aggregation_fields": self.aggregation_fields,
            "metadata": self.metadata,
        }


def evaluate_citation_checks(
    *,
    eval_case: Mapping[str, Any],
    final_result: Mapping[str, Any],
    debug_payload: Mapping[str, Any] | None = None,
) -> CitationEvaluationResult:
    """Evaluate deterministic citation checks for one eval case + system output."""

    case_id = str(eval_case.get("id") or "")
    case_family = str(eval_case.get("family") or "unknown")

    grounded = bool(final_result.get("grounded"))
    sufficient_context = bool(final_result.get("sufficient_context"))
    citation_rows = _extract_citations(final_result)

    # Repo-consistent source of truth: grounded final answers must carry citations.
    citation_required = grounded

    gold_evidence_ids = _string_set(eval_case.get("gold_evidence_ids"))
    selected_scope_ids = _resolve_selected_document_scope(eval_case=eval_case, debug_payload=debug_payload)
    strict_gold = bool(gold_evidence_ids)

    presence_metric = _metric_citation_presence(
        citation_required=citation_required,
        citation_count=len(citation_rows),
        grounded=grounded,
    )

    relevance_metric = _metric_citation_relevance(
        citations=citation_rows,
        selected_scope_ids=selected_scope_ids,
        gold_citation_doc_ids=_extract_gold_citation_doc_ids(eval_case),
    )

    support_metric = _metric_citation_support_match(
        citations=citation_rows,
        gold_evidence_ids=gold_evidence_ids,
        citation_required=citation_required,
    )

    unused_metric = _metric_unused_citation_rate(
        citations=citation_rows,
        gold_evidence_ids=gold_evidence_ids,
        selected_scope_ids=selected_scope_ids,
    )

    metrics = [presence_metric, relevance_metric, support_metric, unused_metric]

    # Overall pass/fail stays conservative and deterministic:
    # - required presence must pass
    # - applicable relevance/support checks must pass
    decisive = [metric.passed for metric in metrics if metric.passed is not None and metric.metric_name != "unused_citation_rate"]
    passed = all(decisive) if decisive else True

    grounded_with_missing_citation = bool(citation_required and len(citation_rows) == 0)

    notes: list[str] = []
    if grounded_with_missing_citation:
        notes.append("grounded answer missing citations")
    if support_metric.passed is False:
        notes.append("citations do not match expected supporting evidence")

    return CitationEvaluationResult(
        evaluator_name="citation_checks_v1",
        case_id=case_id,
        case_family=case_family,
        passed=passed,
        metrics=metrics,
        notes=notes,
        aggregation_fields={
            "case_id": case_id,
            "family": case_family,
            "grounded": grounded,
            "sufficient_context": sufficient_context,
            "citation_required": citation_required,
            "citation_count": len(citation_rows),
            "citation_presence_pass": presence_metric.passed,
            "citation_relevance_pass": relevance_metric.passed,
            "citation_support_match_pass": support_metric.passed,
            "unused_citation_rate": unused_metric.value,
            "grounded_with_missing_citation": grounded_with_missing_citation,
            "strict_gold_evidence_available": strict_gold,
        },
        metadata={
            "deterministic": True,
            "model_based_judgment": False,
            "assumptions": {
                "citation_presence_source_of_truth": "final_result.grounded",
                "support_match_uses_structural_id_overlap": "citation ids vs eval_case.gold_evidence_ids",
                "relevance_uses_scope_and_gold_doc_alignment": True,
            },
        },
    )


def aggregate_citation_results_by_family(
    results: Sequence[CitationEvaluationResult | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-case citation outputs by legal family."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        payload = result.to_dict() if isinstance(result, CitationEvaluationResult) else dict(result)
        fields = payload.get("aggregation_fields") or {}
        family = str(payload.get("case_family") or fields.get("family") or "unknown")
        buckets[family].append(payload)

    aggregated: dict[str, dict[str, Any]] = {}
    for family, items in sorted(buckets.items()):
        case_count = len(items)
        presence_pass_count = 0
        relevance_pass_count = 0
        relevance_applicable_count = 0
        support_pass_count = 0
        support_applicable_count = 0
        grounded_missing_count = 0
        unused_rates: list[float] = []

        for item in items:
            fields = item.get("aggregation_fields") or {}
            if fields.get("citation_presence_pass") is True:
                presence_pass_count += 1
            relevance_pass = fields.get("citation_relevance_pass")
            if isinstance(relevance_pass, bool):
                relevance_applicable_count += 1
                if relevance_pass:
                    relevance_pass_count += 1
            support_pass = fields.get("citation_support_match_pass")
            if isinstance(support_pass, bool):
                support_applicable_count += 1
                if support_pass:
                    support_pass_count += 1
            if fields.get("grounded_with_missing_citation") is True:
                grounded_missing_count += 1
            unused_rate = fields.get("unused_citation_rate")
            if isinstance(unused_rate, (int, float)):
                unused_rates.append(float(unused_rate))

        aggregated[family] = {
            "family": family,
            "case_count": case_count,
            "citation_presence_pass_rate": _safe_rate(presence_pass_count, case_count),
            "citation_relevance_pass_rate": _safe_rate(relevance_pass_count, relevance_applicable_count),
            "citation_support_match_pass_rate": _safe_rate(support_pass_count, support_applicable_count),
            "unused_citation_rate_avg": (sum(unused_rates) / len(unused_rates)) if unused_rates else None,
            "grounded_with_missing_citation_count": grounded_missing_count,
            "grounded_with_missing_citation_rate": _safe_rate(grounded_missing_count, case_count),
        }

    return aggregated


def _metric_citation_presence(*, citation_required: bool, citation_count: int, grounded: bool) -> CitationMetricResult:
    if citation_required and citation_count == 0:
        return CitationMetricResult(
            metric_name="citation_presence",
            value=0.0,
            passed=False,
            details={
                "citation_required": True,
                "citation_count": citation_count,
                "grounded": grounded,
                "failure_code": "grounded_without_citations",
            },
            note="Grounded answers must include citations.",
        )

    return CitationMetricResult(
        metric_name="citation_presence",
        value=1.0,
        passed=True,
        details={
            "citation_required": citation_required,
            "citation_count": citation_count,
            "grounded": grounded,
        },
    )


def _metric_citation_relevance(
    *,
    citations: Sequence[dict[str, Any]],
    selected_scope_ids: set[str],
    gold_citation_doc_ids: set[str],
) -> CitationMetricResult:
    if not citations:
        return CitationMetricResult(
            metric_name="citation_relevance",
            value=None,
            passed=None,
            details={"irrelevant_indices": [], "citation_count": 0},
            note="No citations to evaluate for relevance.",
        )

    irrelevant_indices: list[int] = []
    reasons: dict[int, list[str]] = {}

    cited_doc_ids = {row["document_id"] for row in citations if row.get("document_id")}

    for index, row in enumerate(citations):
        row_reasons: list[str] = []
        document_id = row.get("document_id")
        if selected_scope_ids and document_id and document_id not in selected_scope_ids:
            row_reasons.append("document_outside_selected_scope")
        if row_reasons:
            irrelevant_indices.append(index)
            reasons[index] = row_reasons

    if gold_citation_doc_ids and cited_doc_ids and cited_doc_ids.isdisjoint(gold_citation_doc_ids):
        for index in range(len(citations)):
            reasons.setdefault(index, []).append("citation_document_mismatch_with_gold_refs")
            if index not in irrelevant_indices:
                irrelevant_indices.append(index)

    irrelevant_indices.sort()
    relevance_score = 1.0 - (len(irrelevant_indices) / len(citations))
    return CitationMetricResult(
        metric_name="citation_relevance",
        value=relevance_score,
        passed=len(irrelevant_indices) == 0,
        details={
            "citation_count": len(citations),
            "irrelevant_indices": irrelevant_indices,
            "irrelevance_reasons": {str(k): v for k, v in sorted(reasons.items())},
            "selected_scope_ids": sorted(selected_scope_ids),
            "gold_citation_doc_ids": sorted(gold_citation_doc_ids),
        },
    )


def _metric_citation_support_match(
    *,
    citations: Sequence[dict[str, Any]],
    gold_evidence_ids: set[str],
    citation_required: bool,
) -> CitationMetricResult:
    if citation_required and not citations:
        return CitationMetricResult(
            metric_name="citation_support_match",
            value=None,
            passed=None,
            details={"matched_gold_evidence_ids": [], "gold_evidence_id_count": len(gold_evidence_ids)},
            note="Support match skipped because required citations are missing.",
        )

    if not citations:
        return CitationMetricResult(
            metric_name="citation_support_match",
            value=None,
            passed=None,
            details={"matched_gold_evidence_ids": [], "gold_evidence_id_count": len(gold_evidence_ids)},
            note="No citations to evaluate for support match.",
        )

    if not gold_evidence_ids:
        return CitationMetricResult(
            metric_name="citation_support_match",
            value=None,
            passed=None,
            details={"matched_gold_evidence_ids": [], "gold_evidence_id_count": 0},
            note="No gold_evidence_ids provided; deterministic support match is not applicable.",
        )

    cited_ids: set[str] = set()
    for row in citations:
        cited_ids.update(row.get("candidate_ids", set()))

    matched = sorted(cited_ids.intersection(gold_evidence_ids))
    passed = bool(matched)
    return CitationMetricResult(
        metric_name="citation_support_match",
        value=1.0 if passed else 0.0,
        passed=passed,
        details={
            "matched_gold_evidence_ids": matched,
            "gold_evidence_id_count": len(gold_evidence_ids),
            "cited_candidate_ids": sorted(cited_ids),
        },
        note=None if passed else "Citations did not match expected gold evidence ids.",
    )


def _metric_unused_citation_rate(
    *,
    citations: Sequence[dict[str, Any]],
    gold_evidence_ids: set[str],
    selected_scope_ids: set[str],
) -> CitationMetricResult:
    if not citations:
        return CitationMetricResult(
            metric_name="unused_citation_rate",
            value=0.0,
            passed=True,
            details={"citation_count": 0, "unused_count": 0, "unused_indices": []},
        )

    has_strict_signal = bool(gold_evidence_ids or selected_scope_ids)
    if not has_strict_signal:
        return CitationMetricResult(
            metric_name="unused_citation_rate",
            value=None,
            passed=None,
            details={"citation_count": len(citations), "unused_indices": []},
            note="Unused citation rate not computed: no strict gold evidence or selected scope provided.",
        )

    unused_indices: list[int] = []
    for index, row in enumerate(citations):
        row_unused = False
        if gold_evidence_ids:
            if row.get("candidate_ids", set()).isdisjoint(gold_evidence_ids):
                row_unused = True
        if selected_scope_ids:
            document_id = row.get("document_id")
            if document_id and document_id not in selected_scope_ids:
                row_unused = True
        if row_unused:
            unused_indices.append(index)

    rate = len(unused_indices) / len(citations)
    return CitationMetricResult(
        metric_name="unused_citation_rate",
        value=rate,
        passed=rate == 0.0,
        details={
            "citation_count": len(citations),
            "unused_count": len(unused_indices),
            "unused_indices": unused_indices,
        },
    )


def _extract_citations(final_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = final_result.get("citations")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []

    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            mapping = item
        elif hasattr(item, "__dict__"):
            mapping = vars(item)
        else:
            continue

        parent_chunk_id = _as_clean_str(mapping.get("parent_chunk_id"))
        evidence_unit_id = _as_clean_str(mapping.get("evidence_unit_id"))
        child_chunk_id = _as_clean_str(mapping.get("child_chunk_id"))
        chunk_id = _as_clean_str(mapping.get("chunk_id"))

        candidate_ids = {value for value in [parent_chunk_id, evidence_unit_id, child_chunk_id, chunk_id] if value}

        rows.append(
            {
                "document_id": _as_clean_str(mapping.get("document_id")),
                "parent_chunk_id": parent_chunk_id,
                "supporting_excerpt": _as_clean_str(mapping.get("supporting_excerpt")),
                "candidate_ids": candidate_ids,
            }
        )

    return rows


def _extract_gold_citation_doc_ids(eval_case: Mapping[str, Any]) -> set[str]:
    refs = eval_case.get("gold_citation_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
        return set()

    values: set[str] = set()
    for item in refs:
        if isinstance(item, Mapping):
            doc_id = _as_clean_str(item.get("document_id"))
            if doc_id:
                values.add(doc_id)
    return values


def _resolve_selected_document_scope(
    *,
    eval_case: Mapping[str, Any],
    debug_payload: Mapping[str, Any] | None,
) -> set[str]:
    selected = _string_set(eval_case.get("selected_document_ids"))
    if selected:
        return selected

    if not isinstance(debug_payload, Mapping):
        return set()

    meta = debug_payload.get("meta")
    if isinstance(meta, Mapping):
        selected.update(_string_set(meta.get("selected_document_ids")))
    selected.update(_string_set(debug_payload.get("resolved_document_scope")))
    return selected


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return set()
    return {clean for item in value if (clean := _as_clean_str(item))}


def _as_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "CitationEvaluationResult",
    "CitationMetricResult",
    "aggregate_citation_results_by_family",
    "evaluate_citation_checks",
]
