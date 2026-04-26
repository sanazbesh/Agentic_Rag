from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evals.runners import run_offline_eval as runner_module
from evals.runners.run_offline_eval import run_offline_eval
from ui.local_backend import build_local_backend_dependencies
from evals.runners.offline_fixture_registry import known_offline_fixture_ids


def _write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _case(
    case_id: str,
    family: str,
    *,
    query: str = "Who is the employer?",
    selected_document_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "family": family,
        "query": query,
        "selected_document_ids": selected_document_ids or [],
        "expected_answer_type": "fact_extraction",
        "expected_outcome": "answered",
        "answerability_expected": "answerable",
        "gold_evidence_ids": ["eu-1"],
        "evidence_requirement": "required",
        "safe_failure_expected": False,
        "difficulty": "easy",
        "notes": "test",
    }


def _executor(eval_case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_id = str(eval_case["id"])
    if case_id == "case-fail":
        raise RuntimeError("boom")

    final_result = {
        "answer_text": f"answer::{eval_case['query']}",
        "grounded": True,
        "sufficient_context": True,
        "citations": [{"parent_chunk_id": "eu-1", "document_id": "doc-1"}],
        "warnings": [],
    }
    debug_payload = {
        "query_classification": {"family": eval_case["family"], "routing_notes": [f"legal_question_family:{eval_case['family']}"]},
        "answerability_result": {"sufficient_context": True, "should_answer": True},
        "warnings": [],
    }
    state_payload = {
        "child_results": [{"child_chunk_id": "eu-1", "parent_chunk_id": "p-1", "document_id": "doc-1", "metadata": {"family": eval_case["family"]}}],
        "reranked_child_results": [{"child_chunk_id": "eu-1", "parent_chunk_id": "p-1", "document_id": "doc-1", "metadata": {"family": eval_case["family"]}}],
        "parent_chunks": [{"parent_chunk_id": "p-1", "document_id": "doc-1"}],
    }
    return final_result, debug_payload, state_payload


def _judge_ok(_: str) -> dict[str, Any]:
    return {
        "label": "grounded_answer",
        "confidence_band": "high",
        "short_reason": "supported",
        "supporting_notes": [],
    }


def _safe_failure_ok(_: str) -> dict[str, Any]:
    return {
        "label": "acceptable_insufficient_response",
        "confidence_band": "medium",
        "short_reason": "safe",
        "supporting_notes": [],
    }


def _answer_correctness_ok(_: str) -> dict[str, Any]:
    return {
        "label": "correct",
        "confidence_band": "high",
        "short_reason": "correct",
        "supporting_notes": [],
    }


def test_runner_loads_dataset_executes_cases_and_writes_machine_readable_output(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, [_case("case-1", "party_role_verification")])
    output = tmp_path / "out.json"

    result = run_offline_eval(
        output_path=output,
        dataset_path=dataset,
        case_executor=_executor,
        judge_callables={
            "groundedness": _judge_ok,
            "safe_failure": _safe_failure_ok,
            "answer_correctness": _answer_correctness_ok,
        },
    )

    assert result.summary["case_count"] == 1
    blob = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(blob["cases"], list)
    row = blob["cases"][0]
    assert row["case_id"] == "case-1"
    assert row["system_result"]["answer_text"].startswith("answer::")
    assert row["debug_payload"]["answerability_result"]["should_answer"] is True
    assert "contract_checks" in row["deterministic_eval_results"]
    assert "groundedness" in row["llm_judge_results"]
    for key in (
        "retrieval_version",
        "answerability_version",
        "generation_version",
        "prompt_bundle_version",
        "model_version",
    ):
        assert key in blob["version_attribution"]
        assert key in row["version_attribution"]



def test_runner_can_filter_to_one_family(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset,
        [
            _case("case-party", "party_role_verification"),
            _case("case-chrono", "chronology_date_event"),
        ],
    )

    result = run_offline_eval(
        output_path=tmp_path / "out_family.json",
        dataset_path=dataset,
        family="party_role_verification",
        case_executor=_executor,
        run_model_graders=False,
    )

    assert result.summary["case_count"] == 1
    assert result.case_results[0]["family"] == "party_role_verification"


def test_runner_can_run_all_families_from_repo_dataset_dir(tmp_path: Path, monkeypatch: Any) -> None:
    repo = tmp_path / "repo"
    (repo / "evals" / "datasets").mkdir(parents=True)
    _write_dataset(repo / "evals" / "datasets" / "a.jsonl", [_case("a1", "party_role_verification")])
    _write_dataset(repo / "evals" / "datasets" / "b.jsonl", [_case("b1", "chronology_date_event")])

    monkeypatch.chdir(repo)
    result = run_offline_eval(
        output_path=repo / "offline.json",
        run_all_families=True,
        case_executor=_executor,
        run_model_graders=False,
    )

    assert result.summary["case_count"] == 2
    assert set(result.summary["families"]) == {"party_role_verification", "chronology_date_event"}


def test_runner_handles_per_case_failures_without_aborting(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, [_case("case-ok", "party_role_verification"), _case("case-fail", "party_role_verification")])

    result = run_offline_eval(
        output_path=tmp_path / "out_errors.json",
        dataset_path=dataset,
        case_executor=_executor,
        run_deterministic_evaluators=False,
        run_model_graders=False,
    )

    assert result.summary["case_count"] == 2
    failed_rows = [row for row in result.case_results if row["runner_status"] == "failed"]
    assert len(failed_rows) == 1
    assert "RuntimeError" in (failed_rows[0]["error"] or "")


def test_runner_marks_missing_model_graders_as_skipped(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, [_case("case-1", "party_role_verification")])

    result = run_offline_eval(
        output_path=tmp_path / "out_skip_graders.json",
        dataset_path=dataset,
        case_executor=_executor,
        judge_callables={},
    )

    judges = result.case_results[0]["llm_judge_results"]
    assert judges["groundedness"]["status"] == "skipped"
    assert judges["safe_failure"]["status"] == "skipped"
    assert judges["answer_correctness"]["status"] == "skipped"


def test_runner_uses_explicit_unknown_model_fallback_when_model_version_is_missing(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, [_case("case-1", "party_role_verification")])

    def missing_model_executor(eval_case: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        _ = eval_case
        return (
            {
                "answer_text": "ok",
                "grounded": True,
                "sufficient_context": True,
                "citations": [],
                "warnings": [],
            },
            {"trace": {"retrieval_version": "retrieval.v1"}},
            {},
        )

    result = run_offline_eval(
        output_path=tmp_path / "out_unknown_model.json",
        dataset_path=dataset,
        case_executor=missing_model_executor,
        run_deterministic_evaluators=False,
        run_model_graders=False,
    )

    assert result.case_results[0]["version_attribution"]["model_version"] == "unknown_model_version"


def test_version_attribution_enriches_metadata_without_mutating_core_system_result(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, [_case("case-1", "party_role_verification")])

    result = run_offline_eval(
        output_path=tmp_path / "out_stability.json",
        dataset_path=dataset,
        case_executor=_executor,
        run_deterministic_evaluators=False,
        run_model_graders=False,
    )

    system_result = result.case_results[0]["system_result"]
    assert system_result == {
        "answer_text": "answer::Who is the employer?",
        "grounded": True,
        "sufficient_context": True,
        "citations": [{"parent_chunk_id": "eu-1", "document_id": "doc-1"}],
        "warnings": [],
    }


def test_version_attribution_is_stable_across_repeated_offline_eval_runs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset, [_case("case-1", "party_role_verification")])

    first = run_offline_eval(
        output_path=tmp_path / "out_stable_1.json",
        dataset_path=dataset,
        case_executor=_executor,
        run_deterministic_evaluators=False,
        run_model_graders=False,
    )
    second = run_offline_eval(
        output_path=tmp_path / "out_stable_2.json",
        dataset_path=dataset,
        case_executor=_executor,
        run_deterministic_evaluators=False,
        run_model_graders=False,
    )

    assert first.case_results[0]["version_attribution"] == second.case_results[0]["version_attribution"]


def test_main_wires_cli_case_executor(monkeypatch: Any, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def _fake_cli_executor() -> Any:
        return "cli_executor"

    def _fake_run_offline_eval(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return None

    monkeypatch.setattr(runner_module, "_build_cli_case_executor", _fake_cli_executor)
    monkeypatch.setattr(runner_module, "run_offline_eval", _fake_run_offline_eval)

    output_path = tmp_path / "offline.json"
    exit_code = runner_module.main(["--output", str(output_path), "--family", "party_role_verification"])

    assert exit_code == 0
    assert captured["case_executor"] == "cli_executor"
    assert captured["output_path"] == str(output_path)
    assert captured["family"] == "party_role_verification"


def test_cli_selected_documents_merges_ids_paths_and_existing_descriptors() -> None:
    selected = runner_module._resolve_cli_selected_documents(
        {
            "selected_documents": [{"id": "doc-existing", "path": "fixtures/existing.md", "name": "existing"}],
            "selected_document_ids": ["doc-existing", "doc-added"],
            "selected_document_paths": ["fixtures/existing.md", "fixtures/added.md", "fixtures/path_only.md"],
        }
    )

    assert selected == [
        {"id": "doc-existing", "path": "fixtures/existing.md", "name": "existing"},
        {"id": "doc-added", "path": "fixtures/added.md"},
        {"path": "fixtures/path_only.md"},
    ]


def test_cli_selected_documents_resolves_known_offline_fixture_ids() -> None:
    assert "doc-employment-master" in known_offline_fixture_ids()

    selected = runner_module._resolve_cli_selected_documents(
        {
            "selected_document_ids": ["doc-employment-master", "doc-chronology"],
        }
    )

    assert len(selected) == 2
    assert selected[0]["id"] == "doc-employment-master"
    assert selected[0]["source"] == "offline_fixture"
    assert Path(selected[0]["path"]).is_file()
    assert selected[1]["id"] == "doc-chronology"
    assert Path(selected[1]["path"]).is_file()


def test_cli_selected_documents_raises_clear_error_for_unknown_fixture_id() -> None:
    try:
        runner_module._resolve_cli_selected_documents({"selected_document_ids": ["doc-does-not-exist"]})
    except ValueError as exc:
        assert str(exc) == "offline_fixture_not_found:doc-does-not-exist"
    else:
        raise AssertionError("expected unknown fixture id to raise a clear error")


def test_fixture_resolution_populates_local_backend_scope_and_retrieval() -> None:
    selected = runner_module._resolve_cli_selected_documents(
        {
            "selected_document_ids": ["doc-employment-master"],
        }
    )

    build = build_local_backend_dependencies(selected)

    assert build.scope_meta["loaded_document_count"] > 0
    assert build.scope_meta["parent_chunk_count"] > 0
    assert build.scope_meta["child_chunk_count"] > 0

    hits = build.dependencies.retrieval.hybrid_search(
        "Who is the employer?",
        filters={"selected_document_ids": ["doc-employment-master"]},
        top_k=5,
    )
    assert hits
