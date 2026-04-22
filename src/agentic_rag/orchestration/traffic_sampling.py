"""Local-first production traffic sampling for legal RAG requests.

Ticket 26 scope:
- deterministic configurable sampling strategies,
- one optional JSONL artifact per sampled request,
- no external observability vendor dependencies.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class TrafficSamplingConfig:
    """Sampling configuration for local production traffic review."""

    enabled: bool = False
    output_path: str = "data/sampling/production_traffic_samples.jsonl"

    random_sample_rate: float = 0.0
    high_risk_family_sample_rates: dict[str, float] = field(default_factory=dict)

    low_confidence_enabled: bool = True
    low_confidence_support_levels: tuple[str, ...] = ("none", "limited")
    low_confidence_sample_on_insufficient: bool = True
    low_confidence_sample_on_partial: bool = True
    low_confidence_warning_codes: tuple[str, ...] = (
        "definition_not_supported",
        "missing_evidence",
        "conflicting_evidence",
    )

    high_cost_cost_usd_threshold: float | None = None
    high_cost_total_tokens_threshold: int | None = None


def traffic_sampling_config_from_mapping(raw: Mapping[str, Any] | None) -> TrafficSamplingConfig:
    if not isinstance(raw, Mapping):
        return TrafficSamplingConfig()

    return TrafficSamplingConfig(
        enabled=bool(raw.get("enabled", False)),
        output_path=str(raw.get("output_path") or TrafficSamplingConfig.output_path),
        random_sample_rate=_safe_rate(raw.get("random_sample_rate"), default=0.0),
        high_risk_family_sample_rates=_normalize_family_rates(raw.get("high_risk_family_sample_rates")),
        low_confidence_enabled=bool(raw.get("low_confidence_enabled", True)),
        low_confidence_support_levels=_normalize_string_tuple(
            raw.get("low_confidence_support_levels"),
            default=TrafficSamplingConfig.low_confidence_support_levels,
        ),
        low_confidence_sample_on_insufficient=bool(raw.get("low_confidence_sample_on_insufficient", True)),
        low_confidence_sample_on_partial=bool(raw.get("low_confidence_sample_on_partial", True)),
        low_confidence_warning_codes=_normalize_string_tuple(
            raw.get("low_confidence_warning_codes"),
            default=TrafficSamplingConfig.low_confidence_warning_codes,
        ),
        high_cost_cost_usd_threshold=_safe_optional_float(raw.get("high_cost_cost_usd_threshold")),
        high_cost_total_tokens_threshold=_safe_optional_int(raw.get("high_cost_total_tokens_threshold")),
    )


def maybe_sample_production_traffic(
    *,
    state: Mapping[str, Any],
    final_answer: Any,
    config: TrafficSamplingConfig | Mapping[str, Any] | None = None,
    rng: Callable[[], float] | None = None,
) -> dict[str, Any] | None:
    """Evaluate configured sampling strategies and persist one JSONL sample record.

    Returns the stored record when sampling fired, otherwise ``None``.
    """

    resolved = config if isinstance(config, TrafficSamplingConfig) else traffic_sampling_config_from_mapping(config)
    if not resolved.enabled:
        return None

    random_fn = rng or random.random
    reasons = evaluate_sampling_reasons(state=state, final_answer=final_answer, config=resolved, rng=random_fn)
    if not reasons:
        return None

    record = build_sample_record(state=state, final_answer=final_answer, reasons=reasons)
    _append_jsonl(Path(resolved.output_path), record)
    return record


def evaluate_sampling_reasons(
    *,
    state: Mapping[str, Any],
    final_answer: Any,
    config: TrafficSamplingConfig,
    rng: Callable[[], float],
) -> list[str]:
    """Return stable de-duplicated sampling reasons for one completed request."""

    reasons: list[str] = []

    if _should_random_sample(config=config, rng=rng):
        reasons.append("random_traffic")

    family = _extract_family(state)
    if family:
        family_rate = config.high_risk_family_sample_rates.get(family)
        if _draw_rate(family_rate, rng=rng):
            reasons.append(f"high_risk_family:{family}")

    reasons.extend(_low_confidence_reasons(state=state, final_answer=final_answer, config=config))

    high_cost_reason = _high_cost_reason(state=state, config=config)
    if high_cost_reason:
        reasons.append(high_cost_reason)

    return _dedupe(reasons)


def build_sample_record(*, state: Mapping[str, Any], final_answer: Any, reasons: Sequence[str]) -> dict[str, Any]:
    trace = state.get("trace") if isinstance(state.get("trace"), Mapping) else None
    metrics = state.get("metrics") if isinstance(state.get("metrics"), Mapping) else None

    query = str(state.get("query") or (trace or {}).get("query") or "")
    family = _extract_family(state)

    output = _jsonable(final_answer)

    cost_usd, total_tokens = _extract_cost_and_tokens(state=state, metrics=metrics)
    sample_id = f"smp_{uuid4().hex}"

    return {
        "sample_id": sample_id,
        "timestamp_utc": _utc_now_iso(),
        "request_id": _safe_str((trace or {}).get("request_id")),
        "trace_id": _safe_str((trace or {}).get("trace_id")),
        "query": query,
        "selected_document_ids": _extract_selected_document_ids(state=state, trace=trace),
        "family": family,
        "final_result": output,
        "debug_payload": _build_debug_payload_fragment(state=state),
        "trace": _jsonable(trace),
        "sampling_reasons": list(reasons),
        "cost_usd": cost_usd,
        "total_tokens": total_tokens,
        "version_identifiers": _extract_version_identifiers(trace),
    }


def _extract_family(state: Mapping[str, Any]) -> str | None:
    trace = state.get("trace")
    if isinstance(trace, Mapping):
        active_family = trace.get("active_family")
        if isinstance(active_family, str) and active_family.strip():
            return active_family.strip()

    query_classification = state.get("query_classification")
    routing_notes = getattr(query_classification, "routing_notes", [])
    for note in routing_notes or []:
        if isinstance(note, str) and note.startswith("legal_question_family:"):
            family = note.split(":", 1)[1].strip()
            if family:
                return family
    return None


def _should_random_sample(*, config: TrafficSamplingConfig, rng: Callable[[], float]) -> bool:
    return _draw_rate(config.random_sample_rate, rng=rng)


def _draw_rate(rate: float | None, *, rng: Callable[[], float]) -> bool:
    if rate is None:
        return False
    stable_rate = _safe_rate(rate, default=0.0)
    if stable_rate <= 0.0:
        return False
    if stable_rate >= 1.0:
        return True
    return float(rng()) < stable_rate


def _low_confidence_reasons(*, state: Mapping[str, Any], final_answer: Any, config: TrafficSamplingConfig) -> list[str]:
    if not config.low_confidence_enabled:
        return []

    reasons: list[str] = []
    answerability = state.get("answerability_result")

    sufficient_context = bool(getattr(final_answer, "sufficient_context", False))
    if config.low_confidence_sample_on_insufficient and not sufficient_context:
        reasons.append("low_confidence:insufficient_context")

    if config.low_confidence_sample_on_partial and bool(getattr(answerability, "partially_supported", False)):
        reasons.append("low_confidence:partially_supported")

    support_level = getattr(answerability, "support_level", None)
    if isinstance(support_level, str) and support_level in set(config.low_confidence_support_levels):
        reasons.append(f"low_confidence:support_level:{support_level}")

    warnings = list(getattr(answerability, "warnings", []) or [])
    warning_codes = set(config.low_confidence_warning_codes)
    for warning in warnings:
        if isinstance(warning, str) and warning in warning_codes:
            reasons.append(f"low_confidence:warning:{warning}")

    should_answer = getattr(answerability, "should_answer", None)
    grounded = bool(getattr(final_answer, "grounded", False))
    if should_answer is False and grounded:
        reasons.append("low_confidence:answerability_grounded_conflict")

    return _dedupe(reasons)


def _high_cost_reason(*, state: Mapping[str, Any], config: TrafficSamplingConfig) -> str | None:
    metrics = state.get("metrics")
    metrics_map = metrics if isinstance(metrics, Mapping) else None
    cost_usd, total_tokens = _extract_cost_and_tokens(state=state, metrics=metrics_map)

    threshold_cost = config.high_cost_cost_usd_threshold
    if isinstance(threshold_cost, float) and threshold_cost >= 0 and isinstance(cost_usd, float) and cost_usd >= threshold_cost:
        return f"high_cost:cost_usd>={threshold_cost}"

    threshold_tokens = config.high_cost_total_tokens_threshold
    if isinstance(threshold_tokens, int) and threshold_tokens >= 0 and isinstance(total_tokens, int) and total_tokens >= threshold_tokens:
        return f"high_cost:total_tokens>={threshold_tokens}"

    return None


def _extract_cost_and_tokens(*, state: Mapping[str, Any], metrics: Mapping[str, Any] | None) -> tuple[float | None, int | None]:
    candidate_costs = [
        (metrics or {}).get("cost_usd"),
        state.get("cost_usd"),
        state.get("request_cost_usd"),
        state.get("total_cost_usd"),
    ]
    cost: float | None = None
    for item in candidate_costs:
        if isinstance(item, (int, float)):
            cost = float(item)
            break

    usage = state.get("token_usage")
    total_tokens = None
    if isinstance((metrics or {}).get("total_tokens"), (int, float)):
        total_tokens = int((metrics or {}).get("total_tokens"))
    elif isinstance(usage, Mapping) and isinstance(usage.get("total_tokens"), (int, float)):
        total_tokens = int(usage.get("total_tokens"))
    elif isinstance(state.get("total_tokens"), (int, float)):
        total_tokens = int(state.get("total_tokens"))

    if total_tokens is None and isinstance(usage, Mapping):
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        if isinstance(in_tok, (int, float)) or isinstance(out_tok, (int, float)):
            total_tokens = int(in_tok or 0) + int(out_tok or 0)

    return cost, total_tokens


def _extract_selected_document_ids(*, state: Mapping[str, Any], trace: Mapping[str, Any] | None) -> list[str]:
    ids: list[str] = []
    if isinstance(trace, Mapping):
        for item in trace.get("selected_document_ids", []) or []:
            if item:
                ids.append(str(item))

    for doc in state.get("selected_documents", []) or []:
        if isinstance(doc, Mapping):
            candidate = doc.get("id") or doc.get("document_id")
            if candidate:
                ids.append(str(candidate))
        else:
            candidate = getattr(doc, "id", None) or getattr(doc, "document_id", None)
            if candidate:
                ids.append(str(candidate))

    return _dedupe(ids)


def _build_debug_payload_fragment(*, state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "answerability_result": _jsonable(state.get("answerability_result")),
        "query_classification": _jsonable(state.get("query_classification")),
        "response_route": _jsonable(state.get("response_route")),
        "metrics": _jsonable(state.get("metrics")),
    }


def _extract_version_identifiers(trace: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trace, Mapping):
        return {}
    fields = (
        "schema_version",
        "pipeline_version",
        "retrieval_version",
        "answerability_version",
        "generation_version",
        "prompt_bundle_version",
        "model_version",
    )
    return {key: trace.get(key) for key in fields if key in trace}


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_rate(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return default


def _safe_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_optional_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _normalize_family_rates(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        family = str(key).strip()
        if not family:
            continue
        out[family] = _safe_rate(value, default=0.0)
    return out


def _normalize_string_tuple(raw: Any, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return default
    values = [str(item).strip() for item in raw if str(item).strip()]
    return tuple(values) if values else default


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def _safe_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    stable: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        stable.append(item)
    return stable
