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
    class SessionState(dict):
        __getattr__ = dict.get
        def __setattr__(self, key, value):
            self[key] = value

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


def test_file_uploader_extensions_guard_against_type_object(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components

    monkeypatch.setattr(components, "ALLOWED_EXTENSIONS", type)

    assert components._file_uploader_types() == ["md", "pdf", "txt"]


def test_file_uploader_extensions_guard_against_type_members(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components

    monkeypatch.setattr(components, "ALLOWED_EXTENSIONS", ["pdf", type])

    assert components._file_uploader_types() == ["md", "pdf", "txt"]


def test_persistent_ingestion_exception_renders_traceback_in_ui(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components
    class SessionState(dict):
        __getattr__ = dict.get

        def __setattr__(self, key, value):
            self[key] = value

    class SidebarStub:
        def __init__(self) -> None:
            self.errors: list[str] = []
            self.codes: list[str] = []

        def subheader(self, *_args, **_kwargs): pass
        def file_uploader(self, *_args, **kwargs):
            return None if kwargs.get("accept_multiple_files") else object()
        def info(self, *_args, **_kwargs): pass
        def button(self, label: str, *_args, **_kwargs): return label == "Ingest uploaded document"
        def error(self, message: str): self.errors.append(message)
        def code(self, message: str): self.codes.append(message)
        def markdown(self, *_args, **_kwargs): pass
        def json(self, *_args, **_kwargs): pass
        def divider(self): pass
        def caption(self, *_args, **_kwargs): pass
        def success(self, *_args, **_kwargs): pass
        def warning(self, *_args, **_kwargs): pass

    sidebar = SidebarStub()
    monkeypatch.setattr(components.st, "sidebar", sidebar)
    monkeypatch.setattr(components.st, "session_state", SessionState(uploaded_documents=[]))
    monkeypatch.setattr(components.st, "rerun", lambda: None, raising=False)

    def explode_runtime():
        raise TypeError("object of type 'type' has no len()")

    monkeypatch.setattr(
        components,
        "_load_persistent_ingestion_dependencies",
        lambda: (explode_runtime, lambda *_a, **_k: None, None),
    )

    components._render_upload_controls()

    assert sidebar.errors == ["Persistent ingestion failed"]
    assert sidebar.codes and "TypeError: object of type 'type' has no len()" in sidebar.codes[0]


def test_missing_database_url_shows_setup_message_without_traceback(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components
    from ui.persistent_ingestion import PersistentIngestionSetupError

    class SessionState(dict):
        __getattr__ = dict.get

        def __setattr__(self, key, value):
            self[key] = value

    class SidebarStub:
        def __init__(self) -> None:
            self.errors: list[str] = []
            self.codes: list[str] = []
            self.infos: list[str] = []

        def subheader(self, *_args, **_kwargs): pass
        def file_uploader(self, *_args, **kwargs):
            return None if kwargs.get("accept_multiple_files") else object()
        def info(self, message: str): self.infos.append(message)
        def button(self, label: str, *_args, **_kwargs): return label == "Ingest uploaded document"
        def error(self, message: str): self.errors.append(message)
        def code(self, message: str): self.codes.append(message)
        def markdown(self, *_args, **_kwargs): pass
        def json(self, *_args, **_kwargs): pass
        def divider(self): pass
        def caption(self, *_args, **_kwargs): pass
        def success(self, *_args, **_kwargs): pass
        def warning(self, *_args, **_kwargs): pass

    sidebar = SidebarStub()
    monkeypatch.setattr(components.st, "sidebar", sidebar)
    monkeypatch.setattr(components.st, "session_state", SessionState(uploaded_documents=[]))
    monkeypatch.setattr(components.st, "rerun", lambda: None, raising=False)

    def missing_database_url():
        raise PersistentIngestionSetupError(
            "Persistent ingestion requires DATABASE_URL. Set DATABASE_URL or run the Docker Compose stack."
        )

    monkeypatch.setattr(
        components,
        "_load_persistent_ingestion_dependencies",
        lambda: (missing_database_url, lambda *_a, **_k: None, None),
    )

    components._render_upload_controls()

    assert sidebar.errors == []
    assert sidebar.codes == []
    assert sidebar.infos == [
        "Persistent ingestion requires DATABASE_URL. Set DATABASE_URL or run the Docker Compose stack."
    ]


def test_setup_path_import_failure_uses_generic_fallback_without_unboundlocal(monkeypatch) -> None:
    _install_streamlit_stub()
    import ui.components as components

    class SessionState(dict):
        __getattr__ = dict.get

        def __setattr__(self, key, value):
            self[key] = value

    class SidebarStub:
        def __init__(self) -> None:
            self.errors: list[str] = []
            self.codes: list[str] = []
            self.infos: list[str] = []

        def subheader(self, *_args, **_kwargs): pass
        def file_uploader(self, *_args, **kwargs):
            return None if kwargs.get("accept_multiple_files") else object()
        def info(self, message: str): self.infos.append(message)
        def button(self, label: str, *_args, **_kwargs): return label == "Ingest uploaded document"
        def error(self, message: str): self.errors.append(message)
        def code(self, message: str): self.codes.append(message)
        def markdown(self, *_args, **_kwargs): pass
        def json(self, *_args, **_kwargs): pass
        def divider(self): pass
        def caption(self, *_args, **_kwargs): pass
        def success(self, *_args, **_kwargs): pass
        def warning(self, *_args, **_kwargs): pass

    sidebar = SidebarStub()
    monkeypatch.setattr(components.st, "sidebar", sidebar)
    monkeypatch.setattr(components.st, "session_state", SessionState(uploaded_documents=[]))
    monkeypatch.setattr(components.st, "rerun", lambda: None, raising=False)

    def import_failure():
        raise ModuleNotFoundError("No module named 'ui.persistent_ingestion'")

    monkeypatch.setattr(
        components,
        "_load_persistent_ingestion_dependencies",
        lambda: (import_failure, lambda *_a, **_k: None, None),
    )

    components._render_upload_controls()

    assert sidebar.infos == []
    assert sidebar.errors == ["Persistent ingestion failed"]
    assert sidebar.codes and "ModuleNotFoundError" in sidebar.codes[0]
