from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app import build_real_debug_payload
from agentic_rag.versioning import get_version_attribution, normalize_model_version
from evals.graders.answerability_checks import evaluate_answerability_checks
from evals.graders.citation_checks import evaluate_citation_checks
from evals.graders.contract_checks import evaluate_contract_checks
from evals.graders.family_routing import evaluate_family_routing
from evals.graders.llm_judges.answer_correctness import (
    EVALUATOR_NAME as ANSWER_CORRECTNESS_EVALUATOR,
    build_answer_correctness_prompt,
    parse_answer_correctness_result,
)
from evals.graders.llm_judges.groundedness import evaluate_groundedness_with_llm
from evals.graders.llm_judges.safe_failure import evaluate_safe_failure_with_llm
from evals.graders.retrieval_checks import evaluate_retrieval_checks

try:
    from agentic_rag.orchestration.legal_rag_graph import LegalRagDependencies, run_legal_rag_turn_with_state
except Exception:  # pragma: no cover - allows import in constrained test setups
    LegalRagDependencies = Any  # type: ignore[misc,assignment]
    run_legal_rag_turn_with_state = None


CaseExecutor = Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
JudgeCallable = Callable[[str], Mapping[str, Any] | str]


@dataclass(frozen=True, slots=True)
class OfflineEvalRunResult:
    output_path: str
    summary: dict[str, Any]
    case_results: list[dict[str, Any]]


def run_offline_eval(
    *,
    output_path: str | Path,
    dataset_path: str | Path | None = None,
    family: str | None = None,
    run_all_families: bool = False,
    limit: int | None = None,
    run_deterministic_evaluators: bool = True,
    run_model_graders: bool = True,
    dependencies: LegalRagDependencies | None = None,
    case_executor: CaseExecutor | None = None,
    judge_callables: Mapping[str, JudgeCallable] | None = None,
) -> OfflineEvalRunResult:
    """Run offline legal RAG evaluation across one/all families and persist machine-readable JSON."""

    dataset_files = _resolve_dataset_files(dataset_path=dataset_path, run_all_families=run_all_families, family=family)
    all_cases = _load_cases(dataset_files)
    selected_cases = _filter_cases(all_cases, family=family, limit=limit)

    if case_executor is None:
        if run_legal_rag_turn_with_state is None or dependencies is None:
            raise ValueError("dependencies are required when case_executor is not provided")
        executor = _build_default_case_executor(dependencies=dependencies)
    else:
        executor = case_executor

    case_results: list[dict[str, Any]] = []
    for eval_case in selected_cases:
        case_id = str(eval_case.get("id") or "")
        case_versions = _resolve_case_versions(debug_payload=None, state_payload=None, final_result=None)
        base = {
            "case_id": case_id,
            "family": str(eval_case.get("family") or "unknown"),
            "query": str(eval_case.get("query") or ""),
            "selected_document_ids": [str(item) for item in list(eval_case.get("selected_document_ids") or [])],
            "runner_status": "ok",
            "error": None,
            "version_attribution": dict(case_versions),
            "system_result": {},
            "debug_payload": {},
            "deterministic_eval_results": {},
            "llm_judge_results": {},
        }
        try:
            final_result, debug_payload, state_payload = executor(eval_case)
            base["system_result"] = final_result
            base["debug_payload"] = debug_payload
            base["version_attribution"] = _resolve_case_versions(
                debug_payload=debug_payload,
                state_payload=state_payload,
                final_result=final_result,
            )

            if run_deterministic_evaluators:
                base["deterministic_eval_results"] = _run_deterministic_evaluators(
                    eval_case=eval_case,
                    final_result=final_result,
                    debug_payload=debug_payload,
                    state_payload=state_payload,
                )
            if run_model_graders:
                base["llm_judge_results"] = _run_model_graders(
                    eval_case=eval_case,
                    final_result=final_result,
                    debug_payload=debug_payload,
                    judge_callables=judge_callables or {},
                )
        except Exception as exc:
            base["runner_status"] = "failed"
            base["error"] = f"{type(exc).__name__}: {exc}"

        case_results.append(base)

    summary = _build_summary(case_results=case_results, dataset_files=dataset_files, family_filter=family)
    run_versions = _resolve_case_versions(debug_payload=None, state_payload=None, final_result=None)
    result_blob = {
        "runner": "offline_eval_runner_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_path": str(dataset_path) if dataset_path else None,
            "family": family,
            "run_all_families": run_all_families,
            "limit": limit,
            "run_deterministic_evaluators": run_deterministic_evaluators,
            "run_model_graders": run_model_graders,
        },
        "summary": summary,
        "version_attribution": dict(run_versions),
        "cases": case_results,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result_blob, ensure_ascii=False, indent=2), encoding="utf-8")
    return OfflineEvalRunResult(output_path=str(output), summary=summary, case_results=case_results)



def _resolve_case_versions(
    *,
    debug_payload: Mapping[str, Any] | None,
    state_payload: Mapping[str, Any] | None,
    final_result: Mapping[str, Any] | None,
) -> dict[str, str]:
    trace = debug_payload.get("trace") if isinstance(debug_payload, Mapping) else None
    if not isinstance(trace, Mapping) and isinstance(state_payload, Mapping):
        candidate = state_payload.get("trace")
        trace = candidate if isinstance(candidate, Mapping) else None

    model_version: Any = None
    if isinstance(trace, Mapping):
        model_version = trace.get("model_version")
    if model_version is None and isinstance(final_result, Mapping):
        model_version = final_result.get("model_version")

    return get_version_attribution(model_version=normalize_model_version(model_version))


def _resolve_dataset_files(*, dataset_path: str | Path | None, run_all_families: bool, family: str | None) -> list[Path]:
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"dataset path does not exist: {path}")
        return [path]

    dataset_dir = Path("evals/datasets")
    files = sorted(dataset_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no dataset files found in {dataset_dir}")

    # For family-only runs we still load known offline datasets and then filter by case family.
    if run_all_families or family:
        return files
    return files


def _load_cases(dataset_files: Sequence[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for file_path in dataset_files:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"{file_path}:{line_number} is not a JSON object")
                payload["_dataset_file"] = str(file_path)
                cases.append(payload)
    return cases


def _filter_cases(cases: Sequence[Mapping[str, Any]], *, family: str | None, limit: int | None) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for case in cases:
        if family and str(case.get("family")) != family:
            continue
        filtered.append(dict(case))
    if limit is not None and limit >= 0:
        return filtered[:limit]
    return filtered


def _build_default_case_executor(*, dependencies: LegalRagDependencies) -> CaseExecutor:
    def _execute(eval_case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        query = str(eval_case.get("query") or "")
        follow_up = eval_case.get("follow_up")
        recent_messages = None
        if isinstance(follow_up, Mapping) and follow_up.get("depends_on_prior_turn") is True:
            prior = follow_up.get("prior_messages")
            if isinstance(prior, list):
                recent_messages = [item for item in prior if isinstance(item, Mapping)]

        selected_documents = [
            {"id": str(doc_id)} for doc_id in list(eval_case.get("selected_document_ids") or []) if str(doc_id).strip()
        ]

        final_answer, state = run_legal_rag_turn_with_state(
            query=query,
            dependencies=dependencies,
            recent_messages=recent_messages,
            selected_documents=selected_documents,
        )
        state_payload = dict(state)
        debug_payload = build_real_debug_payload(latest_state=state_payload, selected_documents=selected_documents)
        return final_answer.model_dump(), debug_payload, state_payload

    return _execute


def _run_deterministic_evaluators(
    *,
    eval_case: Mapping[str, Any],
    final_result: Mapping[str, Any],
    debug_payload: Mapping[str, Any],
    state_payload: Mapping[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    try:
        results["contract_checks"] = evaluate_contract_checks(final_result=final_result, debug_payload=debug_payload).to_dict()
    except Exception as exc:
        results["contract_checks"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        retrieval_payload = _build_retrieval_payload(state_payload)
        results["retrieval_checks"] = evaluate_retrieval_checks(eval_case=eval_case, retrieval_payload=retrieval_payload).to_dict()
    except Exception as exc:
        results["retrieval_checks"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        results["answerability_checks"] = evaluate_answerability_checks(
            eval_case=eval_case,
            final_result=final_result,
            debug_payload=debug_payload,
        ).to_dict()
    except Exception as exc:
        results["answerability_checks"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        results["citation_checks"] = evaluate_citation_checks(
            eval_case=eval_case,
            final_result=final_result,
            debug_payload=debug_payload,
        ).to_dict()
    except Exception as exc:
        results["citation_checks"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    try:
        system_output = {
            "query_classification": debug_payload.get("query_classification"),
            "routing_notes": (debug_payload.get("query_classification") or {}).get("routing_notes", []),
            "family": (debug_payload.get("query_classification") or {}).get("family"),
        }
        results["family_routing"] = evaluate_family_routing(eval_case=eval_case, system_output=system_output).to_dict()
    except Exception as exc:
        results["family_routing"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    return results


def _run_model_graders(
    *,
    eval_case: Mapping[str, Any],
    final_result: Mapping[str, Any],
    debug_payload: Mapping[str, Any],
    judge_callables: Mapping[str, JudgeCallable],
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    groundedness_judge = judge_callables.get("groundedness")
    if groundedness_judge is None:
        results["groundedness"] = {"status": "skipped", "reason": "judge callable not provided"}
    else:
        results["groundedness"] = evaluate_groundedness_with_llm(
            eval_case=eval_case,
            system_output=final_result,
            debug_payload=debug_payload,
            judge_callable=groundedness_judge,
        ).to_dict()

    safe_failure_judge = judge_callables.get("safe_failure")
    if safe_failure_judge is None:
        results["safe_failure"] = {"status": "skipped", "reason": "judge callable not provided"}
    else:
        results["safe_failure"] = evaluate_safe_failure_with_llm(
            eval_case=eval_case,
            system_output=final_result,
            debug_payload=debug_payload,
            judge_callable=safe_failure_judge,
        ).to_dict()

    answer_correctness_judge = judge_callables.get("answer_correctness")
    if answer_correctness_judge is None:
        results["answer_correctness"] = {"status": "skipped", "reason": "judge callable not provided"}
    else:
        prompt = build_answer_correctness_prompt(eval_case=eval_case, system_output=final_result)
        case_id = str(eval_case.get("id") or "")
        family = str(eval_case.get("family") or "unknown")
        try:
            raw = answer_correctness_judge(prompt)
            parsed = parse_answer_correctness_result(raw)
            payload = parsed.to_dict()
            payload["family"] = family
            payload["aggregation_fields"] = {
                "case_id": case_id,
                "family": family,
                "label": parsed.label,
                "passed": parsed.passed,
            }
            results["answer_correctness"] = payload
        except Exception as exc:
            results["answer_correctness"] = {
                "evaluator_name": ANSWER_CORRECTNESS_EVALUATOR,
                "label": "malformed_judge_output",
                "passed": False,
                "confidence_band": "low",
                "short_reason": "Judge output malformed; classified as failure.",
                "supporting_notes": [str(exc)],
                "family": family,
                "aggregation_fields": {"case_id": case_id, "family": family, "label": "malformed_judge_output", "passed": False},
                "metadata": {
                    "model_based_judgment": True,
                    "rubric_version": "answer_correctness_v1",
                    "malformed_output_fallback": True,
                },
            }

    return results


def _build_retrieval_payload(state_payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "decomposition_plan",
        "merged_candidates",
        "parent_expansion_child_results",
        "reranked_child_results",
        "child_results",
        "parent_ids",
        "parent_chunks",
    )
    return {key: state_payload.get(key) for key in keys}


def _build_summary(*, case_results: Sequence[Mapping[str, Any]], dataset_files: Sequence[Path], family_filter: str | None) -> dict[str, Any]:
    total = len(case_results)
    failed = sum(1 for item in case_results if item.get("runner_status") == "failed")
    families = sorted({str(item.get("family") or "unknown") for item in case_results})
    return {
        "case_count": total,
        "failed_case_count": failed,
        "passed_case_count": total - failed,
        "family_filter": family_filter,
        "families": families,
        "dataset_files": [str(path) for path in dataset_files],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline legal RAG evals and save machine-readable JSON output.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--dataset-path", help="Optional single dataset JSONL file path.")
    parser.add_argument("--family", help="Optional family filter, e.g. party_role_verification.")
    parser.add_argument("--all-families", action="store_true", help="Run all available families from eval datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of cases after filtering.")
    parser.add_argument("--skip-deterministic", action="store_true", help="Disable deterministic evaluators.")
    parser.add_argument("--skip-llm-graders", action="store_true", help="Disable model-based graders.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_offline_eval(
        output_path=args.output,
        dataset_path=args.dataset_path,
        family=args.family,
        run_all_families=args.all_families,
        limit=args.limit,
        run_deterministic_evaluators=not args.skip_deterministic,
        run_model_graders=not args.skip_llm_graders,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
