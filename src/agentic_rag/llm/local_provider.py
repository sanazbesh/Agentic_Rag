"""Small local-first LLM provider abstraction with Ollama default."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


class ChatProvider(Protocol):
    """Minimal chat-style provider interface."""

    def chat(self, messages: list[dict[str, str]], *, temperature: float, timeout_seconds: float) -> str:
        """Return plain assistant text for the provided messages."""


@dataclass(slots=True, frozen=True)
class LocalLLMConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    timeout_seconds: float = 8.0


def local_llm_config_from_env(env: dict[str, str] | None = None) -> LocalLLMConfig:
    values = env if env is not None else dict(os.environ)
    enabled = str(values.get("AGENTIC_RAG_LOCAL_LLM_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    provider = str(values.get("AGENTIC_RAG_LOCAL_LLM_PROVIDER", "ollama")).strip() or "ollama"
    model = str(values.get("AGENTIC_RAG_LOCAL_LLM_MODEL", "llama3.1:8b")).strip() or "llama3.1:8b"
    base_url = str(values.get("AGENTIC_RAG_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")).strip() or "http://127.0.0.1:11434"

    try:
        temperature = float(values.get("AGENTIC_RAG_LOCAL_LLM_TEMPERATURE", "0.0"))
    except Exception:
        temperature = 0.0
    try:
        timeout_seconds = float(values.get("AGENTIC_RAG_LOCAL_LLM_TIMEOUT_SECONDS", "8.0"))
    except Exception:
        timeout_seconds = 8.0

    return LocalLLMConfig(
        enabled=enabled,
        provider=provider.lower(),
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout_seconds=max(0.5, timeout_seconds),
    )


@dataclass(slots=True)
class OllamaChatProvider:
    base_url: str
    model: str

    def chat(self, messages: list[dict[str, str]], *, temperature: float, timeout_seconds: float) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:  # noqa: S310 - local provider endpoint
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama_unavailable:{exc}") from exc

        parsed = json.loads(body)
        message = parsed.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("ollama_invalid_response:missing_message")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("ollama_invalid_response:missing_content")
        return content.strip()


@dataclass(slots=True)
class PromptLLMClient:
    """Adapter that exposes a prompt completion interface."""

    provider: ChatProvider
    config: LocalLLMConfig

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.provider.chat(
            messages,
            temperature=self.config.temperature,
            timeout_seconds=self.config.timeout_seconds,
        )


def build_local_prompt_llm(config: LocalLLMConfig) -> PromptLLMClient | None:
    if not config.enabled:
        return None
    if config.provider != "ollama":
        return None
    return PromptLLMClient(provider=OllamaChatProvider(base_url=config.base_url, model=config.model), config=config)


def build_local_prompt_llm_from_env(env: dict[str, str] | None = None) -> PromptLLMClient | None:
    return build_local_prompt_llm(local_llm_config_from_env(env))
