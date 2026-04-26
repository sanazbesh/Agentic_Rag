from __future__ import annotations

import json
from pathlib import Path

from evals.ci.offline_eval_ci import evaluate_gate, select_families_from_paths


def test_select_families_from_paths_returns_mapped_families() -> None:
    families = select_families_from_paths(
        [
            "src/agentic_rag/retrieval/parent_child.py",
            "evals/datasets/tier1_party_role.jsonl",
        ]
    )
    assert "party_role_verification" in families
    assert "chronology_date_event" in families
    assert "employment_lifecycle" in families


def test_select_families_from_paths_falls_back_to_tier1() -> None:
    families = select_families_from_paths(["README.md"])
    assert families == ["party_role_verification", "chronology_date_event", "employment_lifecycle"]


def test_evaluate_gate_flags_failed_cases(tmp_path: Path) -> None:
    run = {
        "cases": [
            {
                "case_id": "ok-1",
                "runner_status": "ok",
                "deterministic_eval_results": {
                    "contract_checks": {"passed": True},
                },
                "llm_judge_results": {},
            },
            {
                "case_id": "bad-1",
                "runner_status": "ok",
                "deterministic_eval_results": {
                    "contract_checks": {"passed": False},
                },
                "llm_judge_results": {},
            },
        ]
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")

    result = evaluate_gate(path, min_pass_rate=1.0, max_runner_failures=0)
    assert result.passed is False
    assert result.failing_case_count == 1
    assert result.failing_cases == ["bad-1"]
