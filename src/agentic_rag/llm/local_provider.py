"""Small local-first LLM provider abstraction with llama.cpp default."""

from __future__ import annotations

import os
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ChatProvider(Protocol):
    """Minimal chat-style provider interface."""

    def chat(self, messages: list[dict[str, str]], *, temperature: float, timeout_seconds: float) -> str:
        """Return plain assistant text for the provided messages."""


@dataclass(slots=True, frozen=True)
class LocalLLMConfig:
    enabled: bool = False
    provider: str = "llama_cpp"
    model_path: str = ""
    n_ctx: int = 4096
    temperature: float = 0.0
    timeout_seconds: float = 8.0
    max_tokens: int = 512
    n_gpu_layers: int = 0
    threads: int | None = None


def local_llm_config_from_env(env: dict[str, str] | None = None) -> LocalLLMConfig:
    values = env if env is not None else dict(os.environ)
    enabled = str(values.get("AGENTIC_RAG_LOCAL_LLM_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    provider = str(values.get("AGENTIC_RAG_LOCAL_LLM_PROVIDER", "llama_cpp")).strip() or "llama_cpp"
    model_path = str(values.get("AGENTIC_RAG_LOCAL_LLM_MODEL_PATH", "")).strip()

    try:
        temperature = float(values.get("AGENTIC_RAG_LOCAL_LLM_TEMPERATURE", "0.0"))
    except Exception:
        temperature = 0.0
    try:
        timeout_seconds = float(values.get("AGENTIC_RAG_LOCAL_LLM_TIMEOUT_SECONDS", "8.0"))
    except Exception:
        timeout_seconds = 8.0
    try:
        n_ctx = int(values.get("AGENTIC_RAG_LOCAL_LLM_N_CTX", "4096"))
    except Exception:
        n_ctx = 4096
    try:
        max_tokens = int(values.get("AGENTIC_RAG_LOCAL_LLM_MAX_TOKENS", "512"))
    except Exception:
        max_tokens = 512
    try:
        n_gpu_layers = int(values.get("AGENTIC_RAG_LOCAL_LLM_N_GPU_LAYERS", "0"))
    except Exception:
        n_gpu_layers = 0
    threads_value = values.get("AGENTIC_RAG_LOCAL_LLM_THREADS")
    try:
        threads = int(threads_value) if threads_value not in {None, ""} else None
    except Exception:
        threads = None

    return LocalLLMConfig(
        enabled=enabled,
        provider=provider.lower(),
        model_path=model_path,
        n_ctx=max(128, n_ctx),
        temperature=temperature,
        timeout_seconds=max(0.5, timeout_seconds),
        max_tokens=max(32, max_tokens),
        n_gpu_layers=max(0, n_gpu_layers),
        threads=threads if threads is None else max(1, threads),
    )


@dataclass(slots=True)
class LlamaCppChatProvider:
    model_path: str
    n_ctx: int
    max_tokens: int = 512
    n_gpu_layers: int = 0
    threads: int | None = None

    def chat(self, messages: list[dict[str, str]], *, temperature: float, timeout_seconds: float) -> str:
        try:
            from llama_cpp import Llama  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(f"llama_cpp_import_failed:{type(exc).__name__}") from exc

        if not Path(self.model_path).is_file():
            raise RuntimeError("llama_cpp_model_missing")

        try:
            llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.threads,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"llama_cpp_model_load_failed:{type(exc).__name__}") from exc
        try:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"llama_cpp_generation_failed:{type(exc).__name__}:{timeout_seconds}") from exc
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not choices:
            raise RuntimeError("llama_cpp_invalid_response:missing_choices")
        message = choices[0].get("message", {})
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise RuntimeError("llama_cpp_invalid_response:missing_content")
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
    client, _ = build_local_prompt_llm_with_diagnostics(config)
    return client


def build_local_prompt_llm_with_diagnostics(config: LocalLLMConfig) -> tuple[PromptLLMClient | None, dict[str, str | bool | None]]:
    diagnostics: dict[str, str | bool | None] = {
        "local_llm_attempted": bool(config.enabled and config.provider == "llama_cpp"),
        "provider_init_status": "not_attempted",
        "provider_init_error": None,
        "provider_init_reason": None,
    }
    if not config.enabled:
        diagnostics["provider_init_status"] = "disabled"
        return None, diagnostics
    if config.provider != "llama_cpp":
        diagnostics["provider_init_status"] = "unsupported_provider"
        diagnostics["provider_init_reason"] = "provider_init_failed"
        return None, diagnostics
    if not config.model_path or not Path(config.model_path).is_file():
        diagnostics["provider_init_status"] = "failed"
        diagnostics["provider_init_reason"] = "invalid_model_path"
        return None, diagnostics
    if find_spec("llama_cpp") is None:
        diagnostics["provider_init_status"] = "failed"
        diagnostics["provider_init_reason"] = "provider_import_failed"
        diagnostics["provider_init_error"] = "llama_cpp_import_unavailable"
        return None, diagnostics
    diagnostics["provider_init_status"] = "ready"
    return PromptLLMClient(
        provider=LlamaCppChatProvider(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            max_tokens=config.max_tokens,
            n_gpu_layers=config.n_gpu_layers,
            threads=config.threads,
        ),
        config=config,
    ), diagnostics


def build_local_prompt_llm_from_env(env: dict[str, str] | None = None) -> PromptLLMClient | None:
    return build_local_prompt_llm(local_llm_config_from_env(env))
