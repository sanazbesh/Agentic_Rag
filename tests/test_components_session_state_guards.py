from __future__ import annotations

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


def test_safe_session_list_guards_type_object() -> None:
    _install_streamlit_stub()
    import ui.components as components

    assert components._safe_session_list(type) == []
    assert components._safe_session_list(["a"]) == ["a"]
