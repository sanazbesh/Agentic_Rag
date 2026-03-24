"""Base interfaces for callable tools used by agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class Tool(ABC):
    """A callable utility that an agent can invoke."""

    name: str
    description: str

    @abstractmethod
    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        """Execute tool logic with structured arguments."""


class ToolRegistry(ABC):
    """Stores and resolves available tools."""

    @abstractmethod
    def register(self, tool: Tool) -> None:
        """Register a tool for runtime usage."""

    @abstractmethod
    def get(self, name: str) -> Tool:
        """Retrieve a previously registered tool by name."""
