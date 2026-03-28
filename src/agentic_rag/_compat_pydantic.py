"""Minimal pydantic compatibility shim for constrained environments."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Callable, ClassVar

ConfigDict = dict[str, Any]


def Field(*, default: Any = MISSING, default_factory: Callable[[], Any] | Any = MISSING) -> Any:
    if default_factory is not MISSING:
        return field(default_factory=default_factory)
    if default is not MISSING:
        return field(default=default)
    return field()


class BaseModel:
    model_config: ClassVar[ConfigDict] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not is_dataclass(cls):
            dataclass(slots=True)(cls)

    @classmethod
    def model_validate(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            allowed = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in value.items() if k in allowed}
            return cls(**filtered)
        raise TypeError(f"Cannot validate {type(value)} as {cls.__name__}")

    def model_dump(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
