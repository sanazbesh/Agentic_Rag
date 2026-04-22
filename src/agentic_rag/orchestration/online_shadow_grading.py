"""Lean local-first online shadow grading for sampled production requests.

Ticket 27 scope:
- background grading for sampled live requests,
- deterministic + model-based checks when available,
- stable trace linkage artifacts,
- no external queueing or observability dependencies.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from evals.graders.contract_checks import evaluate_contract_checks
from evals.graders.family_routing import evaluate_family_routing
from evals.graders.llm_judges.groundedness import evaluate_groundedness_with_llm
from evals.graders.llm_judges.safe_failure import evaluate_safe_failure_with_llm

JudgeCallable = Callable[[str], Mapping[str, Any] | str]
DeterministicEvaluator = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None], Mapping[str, Any]]
ModelGrader = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class OnlineShadowGradingConfig:
    enabled: bool = False
    deterministic_evaluators: tuple[str, ...] = ("contract_checks", "family_routing")
    model_graders: tuple[str, ...] = ()
    max_background_workers: int = 1
    output_path: str = "data/evals/online_shadow_eval_results.jsonl"
    trace_link_path: str = "data/evals/trace_shadow_eval_links.json"


def online_shadow_grading_config_from_mapping(raw: Mapping[str, Any] | None) -> OnlineShadowGradingConfig:
    if not isinstance(raw, Mapping):
        return OnlineShadowGradingConfig()

    def _string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return default
        out = [str(item).strip() for item in value if str(item).strip()]
        return tuple(dict.fromkeys(out)) or default

    workers = raw.get("max_background_workers", 1)
    stable_workers = int(workers) if isinstance(workers, (int, float)) else 1
    if stable_workers < 1:
        stable_workers = 1

    return OnlineShadowGradingConfig(
        enabled=bool(raw.get("enabled", False)),
        deterministic_evaluators=_string_tuple(raw.get("deterministic_evaluators"), OnlineShadowGradingConfig.deterministic_evaluators),
        model_graders=_string_tuple(raw.get("model_graders"), OnlineShadowGradingConfig.model_graders),
        max_background_workers=stable_workers,
        output_path=str(raw.get("output_path") or OnlineShadowGradingConfig.output_path),
        trace_link_path=str(raw.get("trace_link_path") or OnlineShadowGradingConfig.trace_link_path),
    )


class OnlineShadowGrader:
    """Small background worker for post-response shadow grading."""

    def __init__(
        self,
        *,
        config: OnlineShadowGradingConfig | Mapping[str, Any] | None = None,
        groundedness_judge_callable: JudgeCallable | None = None,
        safe_failure_judge_callable: JudgeCallable | None = None,
        deterministic_registry: Mapping[str, DeterministicEvaluator] | None = None,
        model_registry: Mapping[str, ModelGrader] | None = None,
    ) -> None:
        self._config = config if isinstance(config, OnlineShadowGradingConfig) else online_shadow_grading_config_from_mapping(config)
        self._groundedness_judge_callable = groundedness_judge_callable
        self._safe_failure_judge_callable = safe_failure_judge_callable
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_background_workers, thread_name_prefix="shadow-grader")
        self._futures: set[Future[dict[str, Any]]] = set()

        self._deterministic_registry = dict(deterministic_registry or _default_deterministic_registry())
        self._model_registry = dict(model_registry or self._build_default_model_registry())

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def wait_for_all(self, *, timeout: float | None = None) -> None:
        pending = list(self._futures)
        for future in pending:
            future.result(timeout=timeout)

    def schedule_sample(self, sampled_record: Mapping[str, Any]) -> Future[dict[str, Any]] | None:
        if not self._config.enabled:
            return None
        future = self._executor.submit(self.grade_sample_sync, sampled_record)
        self._futures.add(future)
        future.add_done_callback(lambda f: self._futures.discard(f))
        return future

    def grade_sample_sync(self, sampled_record: Mapping[str, Any]) -> dict[str, Any]:
        result = self._grade_sample(sampled_record)
        self._persist_result(result)
        return result

    def _grade_sample(self, sampled_record: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(sampled_record)
        trace_id = _safe_optional_str(record.get("trace_id"))
        sample_id = _safe_optional_str(record.get("sample_id"))
        family = _safe_optional_str(record.get("family")) or "unknown"

        result: dict[str, Any] = {
            "shadow_eval_id": f"se_{uuid4().hex}",
            "timestamp_utc": _utc_now_iso(),
            "grading_status": "success",
            "trace_id": trace_id,
            "request_id": _safe_optional_str(record.get("request_id")),
            "sample_id": sample_id,
            "family": family,
            "version_identifiers": record.get("version_identifiers") if isinstance(record.get("version_identifiers"), Mapping) else {},
            "deterministic_results": {},
            "model_results": {},
            "errors": [],
        }

        if trace_id is None:
            result["grading_status"] = "failed"
            result["errors"].append({"code": "missing_trace_id", "message": "Sampled record is missing trace_id."})
            return result

        final_result = record.get("final_result") if isinstance(record.get("final_result"), Mapping) else {}
        debug_payload = record.get("debug_payload") if isinstance(record.get("debug_payload"), Mapping) else None
        eval_case = _build_online_eval_case(record)

        for evaluator_name in self._config.deterministic_evaluators:
            evaluator = self._deterministic_registry.get(evaluator_name)
            if evaluator is None:
                result["deterministic_results"][evaluator_name] = {
                    "status": "unavailable",
                    "error": "evaluator_not_available",
                }
                continue
            try:
                payload = evaluator(eval_case, final_result, debug_payload)
                result["deterministic_results"][evaluator_name] = {"status": "ok", "result": payload}
            except Exception as exc:
                result["grading_status"] = "partial_failure"
                result["deterministic_results"][evaluator_name] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                result["errors"].append({"code": f"deterministic:{evaluator_name}", "message": str(exc)})

        for grader_name in self._config.model_graders:
            grader = self._model_registry.get(grader_name)
            if grader is None:
                result["model_results"][grader_name] = {
                    "status": "unavailable",
                    "error": "grader_not_available_or_not_configured",
                }
                continue
            try:
                payload = grader(eval_case, final_result, debug_payload)
                result["model_results"][grader_name] = {"status": "ok", "result": payload}
            except Exception as exc:
                result["grading_status"] = "partial_failure"
                result["model_results"][grader_name] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                result["errors"].append({"code": f"model:{grader_name}", "message": str(exc)})

        if not result["deterministic_results"] and not result["model_results"]:
            result["grading_status"] = "skipped"

        return result

    def _persist_result(self, result: Mapping[str, Any]) -> None:
        output_path = Path(self._config.output_path)
        trace_link_path = Path(self._config.trace_link_path)

        with self._lock:
            _append_jsonl(output_path, result)
            trace_id = _safe_optional_str(result.get("trace_id"))
            shadow_eval_id = _safe_optional_str(result.get("shadow_eval_id"))
            if trace_id and shadow_eval_id:
                _update_trace_links(trace_link_path, trace_id=trace_id, shadow_eval_id=shadow_eval_id)

    def _build_default_model_registry(self) -> dict[str, ModelGrader]:
        registry: dict[str, ModelGrader] = {}
        if self._groundedness_judge_callable is not None:
            registry["groundedness"] = lambda eval_case, final_result, debug_payload: evaluate_groundedness_with_llm(
                eval_case=eval_case,
                system_output=final_result,
                debug_payload=debug_payload,
                judge_callable=self._groundedness_judge_callable,
            ).to_dict()
        if self._safe_failure_judge_callable is not None:
            registry["safe_failure"] = lambda eval_case, final_result, debug_payload: evaluate_safe_failure_with_llm(
                eval_case=eval_case,
                system_output=final_result,
                debug_payload=debug_payload,
                judge_callable=self._safe_failure_judge_callable,
            ).to_dict()
        return registry


def _default_deterministic_registry() -> dict[str, DeterministicEvaluator]:
    return {
        "contract_checks": lambda _eval_case, final_result, debug_payload: evaluate_contract_checks(
            final_result=final_result,
            debug_payload=debug_payload,
        ).to_dict(),
        "family_routing": lambda eval_case, _final_result, debug_payload: evaluate_family_routing(
            eval_case=eval_case,
            system_output={
                "family": eval_case.get("family"),
                "query_classification": (debug_payload or {}).get("query_classification") if isinstance(debug_payload, Mapping) else None,
                "routing_notes": ((debug_payload or {}).get("query_classification") or {}).get("routing_notes") if isinstance((debug_payload or {}).get("query_classification"), Mapping) else None,
            },
        ).to_dict(),
    }


def _build_online_eval_case(sampled_record: Mapping[str, Any]) -> dict[str, Any]:
    family = _safe_optional_str(sampled_record.get("family")) or "unknown"
    sample_id = _safe_optional_str(sampled_record.get("sample_id")) or f"sample-{uuid4().hex[:8]}"
    return {
        "id": sample_id,
        "family": family,
        "query": _safe_optional_str(sampled_record.get("query")) or "",
        "expected_outcome": "unknown",
        "answerability_expected": "unknown",
        "safe_failure_expected": False,
        "evidence_requirement": "required",
        "gold_evidence_ids": [],
        "gold_citation_refs": [],
        "notes": "online_shadow_eval_sample",
    }


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _update_trace_links(path: Path, *, trace_id: str, shadow_eval_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    links: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                links = dict(loaded)
        except Exception:
            links = {}

    trace_links = links.get(trace_id)
    if not isinstance(trace_links, Mapping):
        trace_links = {}

    history = list(trace_links.get("shadow_eval_ids") or [])
    history.append(shadow_eval_id)

    links[trace_id] = {
        "trace_id": trace_id,
        "latest_shadow_eval_id": shadow_eval_id,
        "shadow_eval_ids": history,
        "updated_at_utc": _utc_now_iso(),
    }
    path.write_text(json.dumps(links, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _safe_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "OnlineShadowGradingConfig",
    "OnlineShadowGrader",
    "online_shadow_grading_config_from_mapping",
]
