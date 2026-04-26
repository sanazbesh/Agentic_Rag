from __future__ import annotations

from agentic_rag.storage import postgres as pg


def test_postgres_config_from_env_defaults() -> None:
    config = pg.postgres_config_from_env({})

    assert config.database_url == ""
    assert config.echo is False
    assert config.enabled is False


def test_postgres_config_from_env_parses_values() -> None:
    config = pg.postgres_config_from_env(
        {
            "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/legal_rag",
            "AGENTIC_RAG_DB_ECHO": "true",
        }
    )

    assert config.database_url == "postgresql+psycopg://user:pass@localhost:5432/legal_rag"
    assert config.echo is True
    assert config.enabled is True


def test_get_postgres_engine_uses_sqlalchemy_create_engine(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_create_engine(url: str, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return "engine-object"

    monkeypatch.setattr(
        pg,
        "_load_sqlalchemy",
        lambda: {
            "create_engine": _fake_create_engine,
            "text": lambda query: query,
            "SQLAlchemyError": Exception,
            "Session": object,
            "sessionmaker": lambda **kwargs: kwargs,
        },
    )

    engine = pg.get_postgres_engine(
        config=pg.PostgresConfig(
            database_url="postgresql+psycopg://user:pass@localhost:5432/legal_rag",
            echo=True,
        ),
        pool_pre_ping=True,
    )

    assert engine == "engine-object"
    assert captured["url"] == "postgresql+psycopg://user:pass@localhost:5432/legal_rag"
    assert captured["kwargs"] == {
        "echo": True,
        "future": True,
        "pool_pre_ping": True,
    }


def test_get_postgres_engine_requires_database_url() -> None:
    try:
        pg.get_postgres_engine(config=pg.PostgresConfig(database_url=""))
    except ValueError as exc:
        assert "DATABASE_URL" in str(exc)
    else:
        raise AssertionError("expected ValueError when DATABASE_URL is missing")


def test_get_postgres_session_factory_uses_existing_engine(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_sessionmaker(**kwargs):
        captured["kwargs"] = kwargs
        return "session-factory"

    monkeypatch.setattr(
        pg,
        "_load_sqlalchemy",
        lambda: {
            "create_engine": lambda *_args, **_kwargs: "engine",
            "text": lambda query: query,
            "SQLAlchemyError": Exception,
            "Session": object,
            "sessionmaker": _fake_sessionmaker,
        },
    )

    result = pg.get_postgres_session_factory(engine="bound-engine", expire_on_commit=False)

    assert result == "session-factory"
    assert captured["kwargs"] == {
        "bind": "bound-engine",
        "autoflush": False,
        "autocommit": False,
        "future": True,
        "expire_on_commit": False,
    }


def test_postgres_health_check_true_on_success(monkeypatch) -> None:
    monkeypatch.setattr(
        pg,
        "_load_sqlalchemy",
        lambda: {
            "create_engine": lambda *_args, **_kwargs: "engine",
            "text": lambda query: query,
            "SQLAlchemyError": Exception,
            "Session": object,
            "sessionmaker": lambda **kwargs: kwargs,
        },
    )

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, _query):
            return None

    class _Engine:
        def connect(self):
            return _Connection()

    assert pg.postgres_health_check(_Engine()) is True


def test_postgres_health_check_false_on_sqlalchemy_error(monkeypatch) -> None:
    class _FakeSQLAlchemyError(Exception):
        pass

    monkeypatch.setattr(
        pg,
        "_load_sqlalchemy",
        lambda: {
            "create_engine": lambda *_args, **_kwargs: "engine",
            "text": lambda query: query,
            "SQLAlchemyError": _FakeSQLAlchemyError,
            "Session": object,
            "sessionmaker": lambda **kwargs: kwargs,
        },
    )

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, _query):
            raise _FakeSQLAlchemyError("boom")

    class _Engine:
        def connect(self):
            return _Connection()

    assert pg.postgres_health_check(_Engine()) is False
