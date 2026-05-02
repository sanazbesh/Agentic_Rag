"""Regression test for SQLAlchemy enum construction in storage models."""


def test_storage_models_base_imports_without_type_error() -> None:
    """Importing Base should not trigger SQLAlchemy Enum TypeError."""

    from agentic_rag.storage.models import Base

    assert Base is not None
