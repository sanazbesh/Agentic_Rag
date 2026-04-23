from __future__ import annotations

from typing import Any

from ui import components
from ui.local_backend import effective_local_llm_settings


def test_local_llm_session_defaults_persist_existing_values(monkeypatch) -> None:
    session_state: dict[str, Any] = {
        "local_llm_enabled": True,
        "local_llm_model": "mistral:7b",
        "local_llm_base_url": "http://localhost:9999",
    }
    monkeypatch.setattr(components.st, "session_state", session_state, raising=False)

    components.initialize_session_state()

    assert session_state["local_llm_enabled"] is True
    assert session_state["local_llm_model"] == "mistral:7b"
    assert session_state["local_llm_base_url"] == "http://localhost:9999"


def test_effective_runtime_settings_switch_between_deterministic_and_ollama() -> None:
    deterministic = effective_local_llm_settings(
        enable_local_llm=False,
        provider="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
        timeout_seconds=8.0,
        use_rewrite=True,
        use_decomposition=True,
        use_synthesis=True,
        mock_backend_active=False,
    )
    assisted = effective_local_llm_settings(
        enable_local_llm=True,
        provider="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0.2,
        timeout_seconds=12.0,
        use_rewrite=True,
        use_decomposition=False,
        use_synthesis=True,
        mock_backend_active=False,
    )

    assert deterministic.enabled is False
    assert assisted.enabled is True
    assert assisted.stages.rewrite is True
    assert assisted.stages.decomposition is False
    assert assisted.stages.synthesis is True


def test_runtime_status_reflects_effective_backend_mode(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(components.st, "info", lambda message: calls.append(("info", str(message))))
    monkeypatch.setattr(components.st, "caption", lambda message: calls.append(("caption", str(message))))
    monkeypatch.setattr(components.st, "success", lambda message: calls.append(("success", str(message))))

    components.render_runtime_mode_status(
        use_mock_backend=False,
        debug_payload={
            "local_llm_runtime": {
                "effective_mode": "ollama_assisted",
                "stages_using_ollama": ["rewrite", "synthesis"],
            }
        },
    )

    assert any(kind == "success" and "Ollama-assisted" in message for kind, message in calls)
