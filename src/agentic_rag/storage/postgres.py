"""Postgres connection foundation for persistent ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(slots=True, frozen=True)
class PostgresConfig:
    """Environment-backed Postgres runtime configuration."""

    database_url: str = ""
    echo: bool = False

    @property
    def enabled(self) -> bool:
        """Return whether a usable database URL is configured."""
        return bool(self.database_url)


def postgres_config_from_env(env: dict[str, str] | None = None) -> PostgresConfig:
    """Load Postgres configuration from environment variables.

    Variables:
    - DATABASE_URL
    - AGENTIC_RAG_DB_ECHO (optional bool)
    """

    values = env if env is not None else dict(os.environ)
    database_url = str(values.get("DATABASE_URL", "")).strip()
    echo_value = str(values.get("AGENTIC_RAG_DB_ECHO", "false")).strip().lower()
    echo = echo_value in {"1", "true", "yes", "on"}
    return PostgresConfig(database_url=database_url, echo=echo)


def _load_sqlalchemy() -> dict[str, Any]:
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.exc import SQLAlchemyError
        from sqlalchemy.orm import Session, sessionmaker
    except Exception as exc:  # pragma: no cover - dependency issue only
        raise RuntimeError(
            "sqlalchemy_not_available: install SQLAlchemy and psycopg to use Postgres storage"
        ) from exc

    return {
        "create_engine": create_engine,
        "text": text,
        "SQLAlchemyError": SQLAlchemyError,
        "Session": Session,
        "sessionmaker": sessionmaker,
    }


def get_postgres_engine(
    config: PostgresConfig | None = None,
    *,
    env: dict[str, str] | None = None,
    **engine_kwargs: Any,
) -> Any:
    """Create a SQLAlchemy engine using Postgres configuration."""

    effective_config = config or postgres_config_from_env(env)
    if not effective_config.database_url:
        raise ValueError("DATABASE_URL is required to create a Postgres engine")

    sqlalchemy = _load_sqlalchemy()
    return sqlalchemy["create_engine"](
        effective_config.database_url,
        echo=effective_config.echo,
        future=True,
        **engine_kwargs,
    )


def get_postgres_session_factory(
    engine: Any | None = None,
    config: PostgresConfig | None = None,
    *,
    env: dict[str, str] | None = None,
    **session_kwargs: Any,
) -> Any:
    """Create a SQLAlchemy session factory bound to a Postgres engine."""

    sqlalchemy = _load_sqlalchemy()
    bound_engine = engine or get_postgres_engine(config=config, env=env)
    return sqlalchemy["sessionmaker"](
        bind=bound_engine,
        autoflush=False,
        autocommit=False,
        future=True,
        **session_kwargs,
    )


def postgres_health_check(engine: Any) -> bool:
    """Return whether the database responds to a trivial query."""

    sqlalchemy = _load_sqlalchemy()
    try:
        with engine.connect() as connection:
            connection.execute(sqlalchemy["text"]("SELECT 1"))
        return True
    except sqlalchemy["SQLAlchemyError"]:
        return False
