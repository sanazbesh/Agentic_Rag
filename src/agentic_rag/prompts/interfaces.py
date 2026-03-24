"""Base interfaces for prompt template management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping


class PromptTemplateStore(ABC):
    """Loads reusable prompt templates."""

    @abstractmethod
    def get_template(self, name: str) -> str:
        """Return template text by identifier."""


class PromptBuilder(ABC):
    """Fills templates with runtime values."""

    @abstractmethod
    def build(self, template: str, variables: Mapping[str, str]) -> str:
        """Return a formatted prompt for model execution."""
