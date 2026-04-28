from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace


def _install_streamlit_stub() -> None:
    if "streamlit" in sys.modules:
        return
    sidebar_stub = SimpleNamespace()
    streamlit_stub = ModuleType("streamlit")
    streamlit_stub.sidebar = sidebar_stub
    streamlit_stub.session_state = {}
    sys.modules["streamlit"] = streamlit_stub


def test_ui_components_imports_when_sqlalchemy_missing(monkeypatch) -> None:
    _install_streamlit_stub()
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sqlalchemy"):
            raise ModuleNotFoundError("No module named 'sqlalchemy'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("ui.components", None)

    module = importlib.import_module("ui.components")

    assert module is not None


def test_persistent_ingestion_dependency_error_is_reported(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ui.persistent_ingestion":
            raise ModuleNotFoundError("No module named 'ui.persistent_ingestion'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("ui.persistent_ingestion", None)

    build_runtime, ingest_document, error_message = components._load_persistent_ingestion_dependencies()

    assert build_runtime is None
    assert ingest_document is None
    assert error_message is not None
    assert "Persistent ingestion is unavailable" in error_message


def test_persisted_document_dependency_error_is_reported(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("ui.persisted_documents"):
            raise ModuleNotFoundError("No module named 'ui.persisted_documents'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    list_docs, ready_docs, to_descriptor, error_message = components._load_persisted_document_dependencies()

    assert list_docs is None
    assert ready_docs is None
    assert to_descriptor is None
    assert error_message is not None
    assert "Persisted document selection is unavailable" in error_message
