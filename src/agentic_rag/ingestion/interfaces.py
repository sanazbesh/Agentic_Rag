"""Base interfaces for data ingestion workflows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from agentic_rag.types import Document


class DataConnector(ABC):
    """Reads raw records from an external source."""

    @abstractmethod
    def fetch(self) -> Iterable[dict]:
        """Yield source-native records from upstream systems."""


class DocumentIngestor(ABC):
    """Transforms source records into normalized documents."""

    @abstractmethod
    def ingest(self, records: Iterable[dict]) -> list[Document]:
        """Convert records into canonical document objects."""
