from __future__ import annotations

from datetime import datetime, timezone

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
sessionmaker = pytest.importorskip("sqlalchemy.orm").sessionmaker

from agentic_rag.storage.models import Base, Document, LifecycleStatus
from ui.persisted_documents import PersistedDocumentRow, list_persisted_documents, ready_persisted_documents


def test_list_persisted_documents_returns_rows(monkeypatch) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    with SessionFactory() as session:
        session.add(
            Document(
                id="doc-1",
                source_name="Policy",
                source_type="md",
                status=LifecycleStatus.READY,
                current_version_id="ver-1",
            )
        )
        session.commit()

    import agentic_rag.storage as storage

    monkeypatch.setattr(storage, "postgres_config_from_env", lambda: type("C", (), {"enabled": True})())
    monkeypatch.setattr(storage, "get_postgres_engine", lambda config: engine)
    monkeypatch.setattr(storage, "get_postgres_session_factory", lambda _engine: SessionFactory)

    rows = list_persisted_documents()
    assert len(rows) == 1
    assert rows[0].document_id == "doc-1"
    assert rows[0].status == LifecycleStatus.READY.value


def test_ready_persisted_documents_filters_non_ready() -> None:
    rows = [
        PersistedDocumentRow("1", "A", "md", "READY", "v1", datetime.now(timezone.utc)),
        PersistedDocumentRow("2", "B", "pdf", "PROCESSING", "v2", datetime.now(timezone.utc)),
    ]
    ready = ready_persisted_documents(rows)
    assert [row.document_id for row in ready] == ["1"]


def test_ready_persisted_documents_rejects_type_input() -> None:
    with pytest.raises(Exception, match="expected iterable/instances, got type object"):
        ready_persisted_documents(PersistedDocumentRow)


def test_list_persisted_documents_rejects_type_session_factory(monkeypatch) -> None:
    import agentic_rag.storage as storage

    monkeypatch.setattr(storage, "postgres_config_from_env", lambda: type("C", (), {"enabled": True})())
    monkeypatch.setattr(storage, "get_postgres_engine", lambda config: object())
    monkeypatch.setattr(storage, "get_postgres_session_factory", lambda _engine: type)

    with pytest.raises(Exception, match="Invalid session factory"):
        list_persisted_documents()
