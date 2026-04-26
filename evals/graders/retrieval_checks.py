"""Deterministic retrieval-only checks for legal RAG eval cases.

This module intentionally stays narrow: it grades whether expected evidence was
retrieved, without any semantic/model-based judgment of the final answer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

DEFAULT_RECALL_KS: tuple[int, ...] = (5, 10)


@dataclass(frozen=True, slots=True)
class RetrievalMetricResult:
    """One machine-readable retrieval metric outcome."""

    metric_name: str
    value: float | None
    passed: bool | None
    details: dict[str, Any]
    note: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    """Per-case retrieval evaluator output for offline runs."""

    evaluator_name: str
    case_id: str
    case_family: str
    passed: bool
    metrics: list[RetrievalMetricResult]
    aggregation_fields: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "case_id": self.case_id,
            "case_family": self.case_family,
            "passed": self.passed,
            "metrics": [asdict(metric) for metric in self.metrics],
            "aggregation_fields": self.aggregation_fields,
            "metadata": self.metadata,
        }


def evaluate_retrieval_checks(
    *,
    eval_case: Mapping[str, Any],
    retrieval_payload: Mapping[str, Any],
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> RetrievalEvaluationResult:
    """Evaluate deterministic retrieval checks for one eval case + retrieval payload."""

    case_id = str(eval_case.get("id") or "")
    case_family = str(eval_case.get("family") or "unknown")
    gold_child_ids = _string_list(eval_case.get("gold_evidence_ids"))
    gold_parent_ids = _string_list(eval_case.get("gold_parent_chunk_ids"))
    gold_doc_ids = _extract_gold_citation_doc_ids(eval_case)

    ranked_hits = _extract_ranked_hits(retrieval_payload)
    observed_child_ids = [hit["child_chunk_id"] for hit in ranked_hits]
    id_match_strategy = _resolve_id_match_strategy(
        gold_ids=gold_child_ids,
        ranked_hits=ranked_hits,
        retrieval_payload=retrieval_payload,
        gold_doc_ids=gold_doc_ids,
    )

    metrics: list[RetrievalMetricResult] = []
    sorted_ks = sorted({int(k) for k in recall_ks if int(k) > 0})
    recall_by_k: dict[int, float | None] = {}
    recall_hit_by_k: dict[int, bool | None] = {}

    for k in sorted_ks:
        metric = _gold_chunk_recall_at_k(
            gold_child_ids=gold_child_ids,
            ranked_hits=ranked_hits,
            k=k,
            strategy=id_match_strategy,
        )
        metrics.append(metric)
        recall_by_k[k] = metric.value
        recall_hit_by_k[k] = metric.passed

    parent_metric = _gold_parent_recall(gold_parent_ids=gold_parent_ids, retrieval_payload=retrieval_payload)
    metrics.append(parent_metric)

    rank_metric = _best_rank_of_gold_evidence(
        gold_child_ids=gold_child_ids,
        ranked_hits=ranked_hits,
        strategy=id_match_strategy,
    )
    metrics.append(rank_metric)

    wrong_family_metric = _wrong_family_usage_rate(
        expected_family=case_family,
        ranked_hits=ranked_hits,
        retrieval_payload=retrieval_payload,
    )
    metrics.append(wrong_family_metric)

    passed = _case_passed(
        recall_hit_by_k=recall_hit_by_k,
        has_gold_evidence=bool(gold_child_ids),
        parent_metric=parent_metric,
        has_gold_parents=bool(gold_parent_ids),
        wrong_family_metric=wrong_family_metric,
    )

    best_rank = rank_metric.details.get("best_rank")
    return RetrievalEvaluationResult(
        evaluator_name="retrieval_checks_v1",
        case_id=case_id,
        case_family=case_family,
        passed=passed,
        metrics=metrics,
        aggregation_fields={
            "case_id": case_id,
            "family": case_family,
            "recall_at_k": {f"k_{k}": recall_by_k[k] for k in sorted_ks},
            "recall_hit_at_k": {f"k_{k}": recall_hit_by_k[k] for k in sorted_ks},
            "best_rank": int(best_rank) if isinstance(best_rank, int) else None,
            "gold_parent_recall": parent_metric.value,
            "wrong_family_usage_rate": wrong_family_metric.value,
            "wrong_family_dominated": wrong_family_metric.details.get("dominated_by_wrong_family"),
        },
        metadata={
            "deterministic": True,
            "model_based_judgment": False,
            "retrieval_mode": _detect_retrieval_mode(retrieval_payload),
            "ranked_child_result_count": len(observed_child_ids),
            "gold_child_id_count": len(gold_child_ids),
            "gold_parent_id_count": len(gold_parent_ids),
            "gold_citation_doc_id_count": len(gold_doc_ids),
            "id_match_layer": id_match_strategy["name"],
        },
    )


def aggregate_retrieval_results_by_family(
    results: Sequence[RetrievalEvaluationResult | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-case retrieval outputs by legal family."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if isinstance(result, RetrievalEvaluationResult):
            payload = result.to_dict()
        else:
            payload = dict(result)
        family = str(payload.get("case_family") or payload.get("aggregation_fields", {}).get("family") or "unknown")
        buckets[family].append(payload)

    aggregated: dict[str, dict[str, Any]] = {}
    for family, items in sorted(buckets.items()):
        recall_values: dict[str, list[float]] = defaultdict(list)
        best_ranks: list[int] = []
        parent_recalls: list[float] = []
        wrong_family_rates: list[float] = []
        pass_count = 0

        for item in items:
            if item.get("passed") is True:
                pass_count += 1
            fields = item.get("aggregation_fields", {})
            for k_name, value in (fields.get("recall_at_k") or {}).items():
                if isinstance(value, (int, float)):
                    recall_values[str(k_name)].append(float(value))
            best_rank = fields.get("best_rank")
            if isinstance(best_rank, int):
                best_ranks.append(best_rank)
            parent_recall = fields.get("gold_parent_recall")
            if isinstance(parent_recall, (int, float)):
                parent_recalls.append(float(parent_recall))
            wrong_rate = fields.get("wrong_family_usage_rate")
            if isinstance(wrong_rate, (int, float)):
                wrong_family_rates.append(float(wrong_rate))

        aggregated[family] = {
            "family": family,
            "case_count": len(items),
            "pass_rate": (pass_count / len(items)) if items else 0.0,
            "recall_at_k": {
                key: (sum(values) / len(values)) if values else None
                for key, values in sorted(recall_values.items())
            },
            "best_rank": {
                "avg": (sum(best_ranks) / len(best_ranks)) if best_ranks else None,
                "median": float(median(best_ranks)) if best_ranks else None,
                "count": len(best_ranks),
            },
            "gold_parent_recall_avg": (sum(parent_recalls) / len(parent_recalls)) if parent_recalls else None,
            "wrong_family_usage_rate_avg": (sum(wrong_family_rates) / len(wrong_family_rates)) if wrong_family_rates else None,
        }

    return aggregated


def _gold_chunk_recall_at_k(
    *,
    gold_child_ids: Sequence[str],
    ranked_hits: Sequence[dict[str, Any]],
    k: int,
    strategy: Mapping[str, Any],
) -> RetrievalMetricResult:
    name = f"gold_chunk_recall_at_{k}"
    if not gold_child_ids:
        return RetrievalMetricResult(
            metric_name=name,
            value=None,
            passed=None,
            details={"k": k, "matched_gold_ids": [], "gold_id_count": 0},
            note="No gold_evidence_ids provided for this case.",
        )

    top_hits = list(ranked_hits[:k])
    matched = sorted({gold_id for gold_id in gold_child_ids if any(_hit_matches_gold(hit, gold_id, strategy) for hit in top_hits)})
    any_match = bool(matched)
    return RetrievalMetricResult(
        metric_name=name,
        value=1.0 if any_match else 0.0,
        passed=any_match,
        details={
            "k": k,
            "matched_gold_ids": matched,
            "matched_gold_count": len(matched),
            "gold_id_count": len(gold_child_ids),
            "top_k_child_chunk_ids": [hit["child_chunk_id"] for hit in top_hits],
            "id_match_layer": str(strategy.get("name") or "unknown"),
        },
    )


def _gold_parent_recall(*, gold_parent_ids: Sequence[str], retrieval_payload: Mapping[str, Any]) -> RetrievalMetricResult:
    if not gold_parent_ids:
        return RetrievalMetricResult(
            metric_name="gold_parent_recall",
            value=None,
            passed=None,
            details={"matched_parent_ids": [], "gold_parent_id_count": 0},
            note="No gold_parent_chunk_ids provided for this case.",
        )

    observed = _extract_observed_parent_ids(retrieval_payload)
    matched = sorted(parent_id for parent_id in gold_parent_ids if parent_id in observed)
    recall = len(matched) / len(gold_parent_ids)
    return RetrievalMetricResult(
        metric_name="gold_parent_recall",
        value=recall,
        passed=bool(matched),
        details={
            "matched_parent_ids": matched,
            "matched_parent_count": len(matched),
            "gold_parent_id_count": len(gold_parent_ids),
            "observed_parent_id_count": len(observed),
        },
    )


def _best_rank_of_gold_evidence(
    *,
    gold_child_ids: Sequence[str],
    ranked_hits: Sequence[dict[str, Any]],
    strategy: Mapping[str, Any],
) -> RetrievalMetricResult:
    if not gold_child_ids:
        return RetrievalMetricResult(
            metric_name="best_rank_of_gold_evidence",
            value=None,
            passed=None,
            details={"best_rank": None},
            note="No gold_evidence_ids provided for this case.",
        )

    best_rank: int | None = None
    best_gold_id: str | None = None
    for idx, hit in enumerate(ranked_hits, start=1):
        matched_gold = next((gold_id for gold_id in gold_child_ids if _hit_matches_gold(hit, gold_id, strategy)), None)
        if matched_gold is None:
            continue
        best_rank = idx
        best_gold_id = matched_gold
        break

    return RetrievalMetricResult(
        metric_name="best_rank_of_gold_evidence",
        value=float(best_rank) if isinstance(best_rank, int) else None,
        passed=best_rank is not None,
        details={"best_rank": best_rank, "matched_gold_id": best_gold_id, "id_match_layer": str(strategy.get("name") or "unknown")},
        note=None if best_rank is not None else "No gold evidence appeared in the ranked retrieval list.",
    )


def _wrong_family_usage_rate(
    *,
    expected_family: str,
    ranked_hits: Sequence[dict[str, Any]],
    retrieval_payload: Mapping[str, Any],
) -> RetrievalMetricResult:
    observed_families: list[str] = []
    for hit in ranked_hits:
        inferred = _infer_family_for_hit(hit, retrieval_payload)
        if inferred:
            observed_families.append(inferred)

    if not observed_families:
        return RetrievalMetricResult(
            metric_name="wrong_family_evidence_usage_rate",
            value=None,
            passed=None,
            details={
                "expected_family": expected_family,
                "known_family_count": 0,
                "wrong_family_count": 0,
                "dominated_by_wrong_family": None,
            },
            note=(
                "No retrievable family metadata found on ranked evidence; "
                "wrong-family usage rate is unavailable for this case."
            ),
        )

    expected = _normalize_family(expected_family)
    wrong_count = sum(1 for item in observed_families if _normalize_family(item) != expected)
    rate = wrong_count / len(observed_families)
    dominated = rate > 0.5
    return RetrievalMetricResult(
        metric_name="wrong_family_evidence_usage_rate",
        value=rate,
        passed=not dominated,
        details={
            "expected_family": expected,
            "known_family_count": len(observed_families),
            "wrong_family_count": wrong_count,
            "dominated_by_wrong_family": dominated,
            "observed_families": observed_families,
        },
        note="Conservative structural metric based only on explicit family metadata.",
    )


def _case_passed(
    *,
    recall_hit_by_k: Mapping[int, bool | None],
    has_gold_evidence: bool,
    parent_metric: RetrievalMetricResult,
    has_gold_parents: bool,
    wrong_family_metric: RetrievalMetricResult,
) -> bool:
    recall_gate = True
    if has_gold_evidence:
        target_k = 10 if 10 in recall_hit_by_k else (max(recall_hit_by_k) if recall_hit_by_k else None)
        recall_gate = bool(recall_hit_by_k.get(target_k)) if target_k is not None else False

    parent_gate = True
    if has_gold_parents:
        parent_gate = bool(parent_metric.passed)

    wrong_family_gate = True
    if isinstance(wrong_family_metric.passed, bool):
        wrong_family_gate = wrong_family_metric.passed

    return recall_gate and parent_gate and wrong_family_gate


def _detect_retrieval_mode(payload: Mapping[str, Any]) -> str:
    plan = payload.get("decomposition_plan")
    if isinstance(plan, Mapping):
        return "decomposed"
    if hasattr(plan, "subqueries"):
        return "decomposed"
    return "single_query"


def _extract_ranked_hits(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the final ranked retrieval list used for answering.

    For decomposed retrieval, prefer ``parent_expansion_child_results`` when a
    valid merged candidate pool exists. Otherwise fall back to
    ``reranked_child_results`` (single-query shape and fallback path).
    """

    has_plan = payload.get("decomposition_plan") is not None
    has_merged = bool(payload.get("merged_candidates"))
    parent_expansion = _as_sequence(payload.get("parent_expansion_child_results"))
    reranked = _as_sequence(payload.get("reranked_child_results"))
    child_results = _as_sequence(payload.get("child_results"))

    if has_plan and has_merged and parent_expansion:
        chosen = parent_expansion
    elif reranked:
        chosen = reranked
    else:
        chosen = child_results

    normalized: list[dict[str, Any]] = []
    for item in chosen:
        child_chunk_id = str(_field(item, "child_chunk_id") or "")
        if not child_chunk_id:
            continue
        normalized.append(
            {
                "child_chunk_id": child_chunk_id,
                "parent_chunk_id": str(_field(item, "parent_chunk_id") or ""),
                "document_id": str(_field(item, "document_id") or ""),
                "metadata": _mapping(_field(item, "payload")) or _mapping(_field(item, "metadata")) or {},
            }
        )
    return normalized


def _extract_observed_parent_ids(payload: Mapping[str, Any]) -> set[str]:
    observed: set[str] = set()

    for key in ("parent_ids",):
        for item in _string_list(payload.get(key)):
            observed.add(item)

    for key in ("parent_chunks", "parent_expansion_child_results", "reranked_child_results"):
        for item in _as_sequence(payload.get(key)):
            parent_id = str(_field(item, "parent_chunk_id") or "")
            if parent_id:
                observed.add(parent_id)

    return observed


def _infer_family_for_hit(hit: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
    metadata = _mapping(hit.get("metadata")) or {}
    parent_lookup = _parent_family_lookup(payload)

    for candidate in (
        metadata.get("legal_family"),
        metadata.get("family"),
        metadata.get("legal_question_family"),
        metadata.get("question_family"),
        metadata.get("family_tag"),
    ):
        normalized = _extract_family_token(candidate)
        if normalized:
            return normalized

    routing_notes = metadata.get("routing_notes")
    if isinstance(routing_notes, Sequence) and not isinstance(routing_notes, (str, bytes)):
        for note in routing_notes:
            normalized = _extract_family_token(note)
            if normalized:
                return normalized

    parent_id = str(hit.get("parent_chunk_id") or "")
    if parent_id and parent_id in parent_lookup:
        return parent_lookup[parent_id]

    return None


def _parent_family_lookup(payload: Mapping[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in _as_sequence(payload.get("parent_chunks")):
        parent_id = str(_field(item, "parent_chunk_id") or "")
        if not parent_id:
            continue
        metadata = _mapping(_field(item, "metadata")) or {}
        for candidate in (
            metadata.get("legal_family"),
            metadata.get("family"),
            metadata.get("legal_question_family"),
            metadata.get("question_family"),
            metadata.get("family_tag"),
        ):
            family = _extract_family_token(candidate)
            if family:
                lookup[parent_id] = family
                break
    return lookup


def _extract_family_token(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    token = value.strip()
    if token.startswith("legal_question_family:"):
        token = token.split(":", 1)[1]
    return _normalize_family(token)


def _normalize_family(family: str) -> str:
    token = family.strip()
    aliases = {
        "party_role_entity": "party_role_verification",
        "employment_contract_lifecycle": "employment_lifecycle",
        "correspondence_litigation_milestone": "correspondence_litigation_milestones",
    }
    return aliases.get(token, token)


def _hit_matches_gold(hit: Mapping[str, Any], gold_id: str, strategy: Mapping[str, Any]) -> bool:
    strategy_name = str(strategy.get("name") or "")
    if strategy_name == "citation_parent_alignment":
        citation_parent_ids = strategy.get("citation_parent_ids") or set()
        if not isinstance(citation_parent_ids, set):
            citation_parent_ids = set(citation_parent_ids)
        if not citation_parent_ids:
            return False
        return str(hit.get("parent_chunk_id") or "") in citation_parent_ids

    if str(hit.get("child_chunk_id") or "") == gold_id:
        return True
    if str(hit.get("parent_chunk_id") or "") == gold_id:
        return True
    metadata = _mapping(hit.get("metadata")) or {}
    if str(metadata.get("evidence_unit_id") or "") == gold_id:
        return True
    evidence_ids = metadata.get("evidence_unit_ids")
    if isinstance(evidence_ids, Sequence) and not isinstance(evidence_ids, (str, bytes)):
        return gold_id in {str(item) for item in evidence_ids if isinstance(item, str)}
    return False


def _resolve_id_match_strategy(
    *,
    gold_ids: Sequence[str],
    ranked_hits: Sequence[Mapping[str, Any]],
    retrieval_payload: Mapping[str, Any],
    gold_doc_ids: set[str],
) -> dict[str, Any]:
    observed_child_ids = {str(hit.get("child_chunk_id") or "") for hit in ranked_hits if str(hit.get("child_chunk_id") or "")}
    observed_parent_ids = {str(hit.get("parent_chunk_id") or "") for hit in ranked_hits if str(hit.get("parent_chunk_id") or "")}
    observed_evidence_unit_ids: set[str] = set()
    for hit in ranked_hits:
        metadata = _mapping(hit.get("metadata")) or {}
        evidence_unit_id = str(metadata.get("evidence_unit_id") or "")
        if evidence_unit_id:
            observed_evidence_unit_ids.add(evidence_unit_id)
        evidence_ids = metadata.get("evidence_unit_ids")
        if isinstance(evidence_ids, Sequence) and not isinstance(evidence_ids, (str, bytes)):
            observed_evidence_unit_ids.update(str(item) for item in evidence_ids if isinstance(item, str) and item)

    gold_set = {item for item in gold_ids if item}
    if gold_set.intersection(observed_evidence_unit_ids):
        return {"name": "evidence_unit_id", "gold_id_type": "evidence_unit", "citation_parent_ids": set()}
    if gold_set.intersection(observed_child_ids):
        return {"name": "child_chunk_id", "gold_id_type": "child_chunk", "citation_parent_ids": set()}
    if gold_set.intersection(observed_parent_ids):
        return {"name": "parent_chunk_id", "gold_id_type": "parent_chunk", "citation_parent_ids": set()}

    citation_parent_ids = _extract_citation_parent_ids(retrieval_payload)
    if citation_parent_ids and (gold_doc_ids or gold_set):
        return {
            "name": "citation_parent_alignment",
            "gold_id_type": "legacy_or_unmapped",
            "citation_parent_ids": citation_parent_ids,
            "gold_citation_doc_ids": sorted(gold_doc_ids),
        }

    return {"name": "unmatched", "gold_id_type": "unknown", "citation_parent_ids": set()}


def _extract_citation_parent_ids(payload: Mapping[str, Any]) -> set[str]:
    citations = _as_sequence(payload.get("citations"))
    parent_ids: set[str] = set()
    for item in citations:
        parent = str(_field(item, "parent_chunk_id") or "")
        if parent:
            parent_ids.add(parent)
    return parent_ids


def _extract_gold_citation_doc_ids(eval_case: Mapping[str, Any]) -> set[str]:
    refs = eval_case.get("gold_citation_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        return set()
    doc_ids: set[str] = set()
    for item in refs:
        doc_id = str(_field(item, "document_id") or "").strip()
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def _field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]
