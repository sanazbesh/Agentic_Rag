from __future__ import annotations

from typing import Any

from ui.backend_adapter import run_backend_query
from ui.session_memory import append_conversation_turn, build_backend_context, clear_conversation


def _valid_result(answer_text: str) -> dict[str, Any]:
    return {
        "answer_text": answer_text,
        "grounded": True,
        "sufficient_context": True,
        "citations": [],
        "warnings": [],
    }


def _run_turn(
    state: dict[str, Any],
    *,
    query: str,
    runner_calls: list[dict[str, Any]],
    recent_messages_override: list[dict[str, Any]] | None = None,
) -> None:
    conversation_summary, recent_messages, _ = build_backend_context(
        history=state["conversation_history"],
        conversation_summary_input=state.get("conversation_summary_input"),
        recent_messages_override=recent_messages_override,
    )

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        runner_calls.append(kwargs)
        return _valid_result(answer_text=f"answer for {kwargs['query']}")

    response = run_backend_query(
        query=query,
        conversation_summary=conversation_summary,
        recent_messages=recent_messages,
        selected_documents=[],
        use_mock_backend=False,
        real_backend_runner=fake_runner,
    )

    state["latest_result"] = response.final_result
    state["latest_debug_payload"] = response.debug_payload
    state["conversation_history"] = append_conversation_turn(
        history=state["conversation_history"],
        query=query,
        answer_text=response.final_result["answer_text"],
        metadata={},
    )
    state["current_query_input"] = ""


def test_consecutive_runs_succeed_without_reset() -> None:
    state = {
        "current_query_input": "",
        "conversation_summary_input": "",
        "conversation_history": [],
        "latest_result": None,
        "latest_debug_payload": None,
    }
    calls: list[dict[str, Any]] = []

    _run_turn(state, query="Question one", runner_calls=calls)
    _run_turn(state, query="Question two", runner_calls=calls)

    assert len(calls) == 2
    assert calls[1]["query"] == "Question two"


def test_conversation_history_appends_across_runs() -> None:
    state = {"conversation_history": [], "conversation_summary_input": "", "current_query_input": ""}
    calls: list[dict[str, Any]] = []

    _run_turn(state, query="Q1", runner_calls=calls)
    _run_turn(state, query="Q2", runner_calls=calls)

    assert len(state["conversation_history"]) == 2
    assert state["conversation_history"][0]["query"] == "Q1"
    assert state["conversation_history"][1]["query"] == "Q2"


def test_reset_starts_fresh_conversation() -> None:
    state = {
        "conversation_history": [{"query": "Q1", "answer_text": "A1", "metadata": {}}],
        "latest_result": _valid_result("A1"),
        "latest_debug_payload": {"debug": True},
        "current_query_input": "next",
    }

    state["conversation_history"] = clear_conversation()
    state["latest_result"] = None
    state["latest_debug_payload"] = None
    state["current_query_input"] = ""

    assert state["conversation_history"] == []
    assert state["latest_result"] is None
    assert state["current_query_input"] == ""


def test_latest_result_persists_while_input_remains_usable() -> None:
    state = {"conversation_history": [], "conversation_summary_input": "", "current_query_input": "question"}
    calls: list[dict[str, Any]] = []

    _run_turn(state, query="Question", runner_calls=calls)

    assert state["latest_result"]["answer_text"] == "answer for Question"
    assert state["current_query_input"] == ""


def test_second_run_passes_prior_conversation_context_by_default() -> None:
    state = {"conversation_history": [], "conversation_summary_input": "", "current_query_input": ""}
    calls: list[dict[str, Any]] = []

    _run_turn(state, query="First", runner_calls=calls)
    _run_turn(state, query="Follow up", runner_calls=calls)

    assert calls[1]["recent_messages"] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "answer for First", "metadata": {}},
    ]


def test_recent_messages_override_remains_compatible() -> None:
    state = {"conversation_history": [], "conversation_summary_input": "", "current_query_input": ""}
    calls: list[dict[str, Any]] = []
    override = [{"role": "user", "content": "manual context"}]

    _run_turn(state, query="Q1", runner_calls=calls)
    _run_turn(state, query="Q2", runner_calls=calls, recent_messages_override=override)

    assert calls[1]["recent_messages"] == override


def test_no_forced_reset_regression_between_normal_turns() -> None:
    state = {"conversation_history": [], "conversation_summary_input": "", "current_query_input": ""}
    calls: list[dict[str, Any]] = []

    _run_turn(state, query="Turn 1", runner_calls=calls)
    _run_turn(state, query="Turn 2", runner_calls=calls)

    assert state["conversation_history"][-1]["query"] == "Turn 2"
    assert len(calls) == 2
