"""Base interfaces for configuration and dependency wiring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class ConfigLoader(ABC):
    """Loads configuration from file, env, or remote source."""

    @abstractmethod
    def load(self) -> Mapping[str, Any]:
        """Return normalized configuration values."""


class SettingsProvider(ABC):
    """Exposes typed settings to the application."""

    @abstractmethod
    def get(self, key: str, default: Any | None = None) -> Any:
        """Return one config value by key."""
