"""Base interfaces for offline and online evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class Metric(ABC):
    """Calculates a single quality signal."""

    name: str

    @abstractmethod
    def compute(self, prediction: str, reference: str) -> float:
        """Return a scalar score for one example."""


class Evaluator(ABC):
    """Runs collections of metrics over benchmark examples."""

    @abstractmethod
    def evaluate(self, dataset: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
        """Return aggregate metric values."""
