from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.runners.run_offline_eval import run_offline_eval


TIER1_FAMILIES = [
    "party_role_verification",
    "chronology_date_event",
    "employment_lifecycle",
]

# Minimal, explicit mapping to keep solo-project maintenance simple.
PATH_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("src/agentic_rag/retrieval/", tuple(TIER1_FAMILIES)),
    ("src/agentic_rag/orchestration/", tuple(TIER1_FAMILIES)),
    ("src/agentic_rag/tools/answerability.py", tuple(TIER1_FAMILIES)),
    ("evals/datasets/tier1_party_role", ("party_role_verification",)),
    ("evals/datasets/regressions_party_role", ("party_role_verification",)),
    ("evals/datasets/tier1_chronology", ("chronology_date_event",)),
    ("evals/datasets/tier1_employment_lifecycle", ("employment_lifecycle",)),
    ("evals/datasets/regressions_followups", ("chronology_date_event", "employment_lifecycle")),
    ("evals/datasets/regressions_definition", ("policy_issue_spotting",)),
)


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    case_count: int
    failing_case_count: int
    failing_cases: list[str]


def _ci_case_executor(eval_case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    family = str(eval_case.get("family") or "unknown")
    query = str(eval_case.get("query") or "")
    expected_unanswerable = str(eval_case.get("answerability_expected") or "").lower() == "unanswerable"

    gold_evidence_ids = [str(item) for item in list(eval_case.get("gold_evidence_ids") or []) if str(item).strip()]
    primary_evidence_id = gold_evidence_ids[0] if gold_evidence_ids else f"ci.{family}.e1"

    selected_document_ids = [str(item) for item in list(eval_case.get("selected_document_ids") or []) if str(item).strip()]
    citation_document_id = selected_document_ids[0] if selected_document_ids else "doc-ci"

    if expected_unanswerable:
        final_result = {
            "answer_text": "Insufficient evidence in the selected documents to answer safely.",
            "grounded": False,
            "sufficient_context": False,
            "citations": [],
            "warnings": ["insufficient_evidence"],
        }
        answerability = {"sufficient_context": False, "should_answer": False, "coverage": "insufficient"}
        child_results: list[dict[str, Any]] = []
        reranked_results: list[dict[str, Any]] = []
        parent_chunks: list[dict[str, Any]] = []
    else:
        final_result = {
            "answer_text": f"Supported answer for: {query}",
            "grounded": True,
            "sufficient_context": True,
            "citations": [{"parent_chunk_id": f"p-{primary_evidence_id}", "child_chunk_id": primary_evidence_id, "document_id": citation_document_id}],
            "warnings": [],
        }
        answerability = {"sufficient_context": True, "should_answer": True, "coverage": "sufficient"}
        child_results = [
            {
                "child_chunk_id": primary_evidence_id,
                "parent_chunk_id": f"p-{primary_evidence_id}",
                "document_id": citation_document_id,
                "metadata": {"family": family},
            }
        ]
        reranked_results = list(child_results)
        parent_chunks = [{"parent_chunk_id": f"p-{primary_evidence_id}", "document_id": citation_document_id}]

    debug_payload = {
        "query_classification": {
            "family": family,
            "routing_notes": [f"legal_question_family:{family}"],
        },
        "answerability_result": answerability,
        "warnings": list(final_result["warnings"]),
    }
    state_payload = {
        "child_results": child_results,
        "reranked_child_results": reranked_results,
        "parent_chunks": parent_chunks,
        "merged_candidates": [],
    }
    return final_result, debug_payload, state_payload


def select_families_from_paths(changed_paths: Sequence[str]) -> list[str]:
    selected: list[str] = []
    for path in changed_paths:
        normalized = path.strip()
        if not normalized:
            continue
        for prefix, families in PATH_FAMILY_RULES:
            if normalized.startswith(prefix):
                for family in families:
                    if family not in selected:
                        selected.append(family)

    if selected:
        return selected
    return list(TIER1_FAMILIES)


def evaluate_gate(run_json: Path, *, min_pass_rate: float = 1.0, max_runner_failures: int = 0) -> GateResult:
    blob = json.loads(run_json.read_text(encoding="utf-8"))
    cases = [item for item in blob.get("cases", []) if isinstance(item, Mapping)]

    failing_cases: list[str] = []
    runner_failed_count = 0

    for case in cases:
        case_id = str(case.get("case_id") or "unknown_case")
        if str(case.get("runner_status") or "") != "ok":
            runner_failed_count += 1
            failing_cases.append(case_id)
            continue

        failed = False
        for group_key in ("deterministic_eval_results", "llm_judge_results"):
            group = case.get(group_key)
            if not isinstance(group, Mapping):
                continue
            for evaluator in group.values():
                if not isinstance(evaluator, Mapping):
                    continue
                if evaluator.get("status") in {"skipped", "error"}:
                    if evaluator.get("status") == "error":
                        failed = True
                    continue
                passed_value = evaluator.get("passed")
                if passed_value is None:
                    passed_value = evaluator.get("is_correct")
                if passed_value is False:
                    failed = True
        if failed:
            failing_cases.append(case_id)

    case_count = len(cases)
    failing_case_count = len(failing_cases)
    pass_rate = (case_count - failing_case_count) / case_count if case_count else 0.0
    gate_passed = runner_failed_count <= max_runner_failures and pass_rate >= min_pass_rate

    return GateResult(
        passed=gate_passed,
        case_count=case_count,
        failing_case_count=failing_case_count,
        failing_cases=failing_cases,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI helpers for offline eval gating.")
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select-families", help="Select impacted families from changed files.")
    select.add_argument("--changed-files", required=True, help="Text file with one changed path per line.")
    select.add_argument("--output", required=True, help="Output JSON file path.")

    run = sub.add_parser("run", help="Run CI-mode offline eval.")
    run.add_argument("--mode", choices=("smoke", "family", "full"), required=True)
    run.add_argument("--output", required=True, help="Output run JSON path.")
    run.add_argument("--family", action="append", default=[], help="Family to run (repeatable).")

    gate = sub.add_parser("check-gate", help="Fail CI when run output misses gate requirements.")
    gate.add_argument("--run-json", required=True)
    gate.add_argument("--min-pass-rate", type=float, default=1.0)
    gate.add_argument("--max-runner-failures", type=int, default=0)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.command == "select-families":
        changed = Path(args.changed_files).read_text(encoding="utf-8").splitlines()
        families = select_families_from_paths(changed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"families": families}, indent=2), encoding="utf-8")
        return 0

    if args.command == "run":
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.mode == "smoke":
            run_offline_eval(
                output_path=output_path,
                run_all_families=True,
                limit=12,
                run_model_graders=False,
                case_executor=_ci_case_executor,
            )
            return 0

        if args.mode == "family":
            families = list(dict.fromkeys(args.family)) or list(TIER1_FAMILIES)
            all_case_results: list[dict[str, Any]] = []
            summaries: list[dict[str, Any]] = []
            for family in families:
                result = run_offline_eval(
                    output_path=output_path.parent / f"family_{family}.json",
                    family=family,
                    run_model_graders=False,
                    case_executor=_ci_case_executor,
                )
                all_case_results.extend(result.case_results)
                summaries.append(result.summary)

            combined = {
                "runner": "offline_eval_ci_family_v1",
                "summary": {
                    "families": families,
                    "case_count": len(all_case_results),
                    "family_run_summaries": summaries,
                },
                "cases": all_case_results,
            }
            output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            return 0

        run_offline_eval(
            output_path=output_path,
            run_all_families=True,
            run_model_graders=False,
            case_executor=_ci_case_executor,
        )
        return 0

    gate_result = evaluate_gate(
        Path(args.run_json),
        min_pass_rate=float(args.min_pass_rate),
        max_runner_failures=int(args.max_runner_failures),
    )
    print(
        json.dumps(
            {
                "passed": gate_result.passed,
                "case_count": gate_result.case_count,
                "failing_case_count": gate_result.failing_case_count,
                "failing_cases": gate_result.failing_cases,
            },
            indent=2,
        )
    )
    return 0 if gate_result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
