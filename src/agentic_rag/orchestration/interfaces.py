"""Base interfaces for agent and workflow orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from agentic_rag.types import Generation, RetrievedItem


class Agent(ABC):
    """Runnable unit that handles a specific task within a workflow."""

    @abstractmethod
    def run(self, prompt: str, context: Sequence[RetrievedItem]) -> Generation:
        """Generate an output given prompt and retrieval context."""


class Orchestrator(ABC):
    """Coordinates end-to-end execution for a user request."""

    @abstractmethod
    def execute(self, query: str) -> Generation:
        """Run the full agentic RAG pipeline for a query."""
