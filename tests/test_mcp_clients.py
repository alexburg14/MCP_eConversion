"""Tests for the remote MCP client bridge (mcp_clients + openai_tools glue).

These test the *mechanics* of exposing the already-running eLabFTW/DataTagger
MCP servers to the chat: schema conversion, namespacing, dispatch, and failure
degradation. They mock the network layer (no live tokens needed); the live E2E
check runs on the server against the real /el and /dt endpoints.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mcp_clients
import openai_tools


# --- Schema conversion ------------------------------------------------------

def test_to_openai_schema_adds_required_keys():
    tool = {
        "name": "get_experiment",
        "description": "Fetch one experiment",
        "inputSchema": {"properties": {"id": {"type": "integer"}}},
    }
    schema = mcp_clients._to_openai_schema("elab_get_experiment", tool)
    fn = schema["function"]
    assert fn["name"] == "elab_get_experiment"
    assert fn["parameters"]["type"] == "object"  # added
    assert fn["parameters"]["properties"] == {"id": {"type": "integer"}}
    assert schema["type"] == "function"


def test_to_openai_schema_handles_missing_input_schema():
    schema = mcp_clients._to_openai_schema("dt_x", {"name": "x"})
    assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


# --- Auth mode --------------------------------------------------------------

def test_auth_query_mode_appends_token_to_url():
    url, headers = mcp_clients._auth("https://x/mcp", "TOK", "query")
    assert url == "https://x/mcp?token=TOK"
    assert headers == {}


def test_auth_query_mode_preserves_existing_query():
    url, headers = mcp_clients._auth("https://x/mcp?a=1", "TOK", "query")
    assert url == "https://x/mcp?a=1&token=TOK"
    assert headers == {}


def test_auth_header_mode_sets_bearer():
    url, headers = mcp_clients._auth("https://x/mcp", "TOK", "header")
    assert url == "https://x/mcp"
    assert headers == {"Authorization": "Bearer TOK"}


def test_auth_modes_map():
    assert mcp_clients.AUTH_MODES["elab"] == "query"
    assert mcp_clients.AUTH_MODES["dt"] == "header"


# --- Friendly errors --------------------------------------------------------

class _Fake401(Exception):
    def __init__(self):
        super().__init__("Client error '401 Unauthorized' for url 'https://x'")


def test_friendly_error_401():
    msg = mcp_clients._friendly_error(_Fake401())
    assert "ungültig" in msg or "401" in msg


def test_friendly_error_403():
    class _Fake403(Exception):
        def __init__(self):
            super().__init__("Client error '403 Forbidden' for url 'https://x'")
    msg = mcp_clients._friendly_error(_Fake403())
    assert "403" in msg or "Nicht berechtigt" in msg


def test_friendly_error_nested_exceptiongroup():
    # ExceptionGroup -> HTTPStatusError-like leaf must be found recursively.
    leaf = _Fake401()
    eg = ExceptionGroup("unhandled errors in a TaskGroup", [leaf])
    msg = mcp_clients._friendly_error(eg)
    assert "ungültig" in msg or "401" in msg


def test_friendly_error_unknown_type_returns_something():
    class _Weird(Exception):
        pass
    assert mcp_clients._friendly_error(_Weird("kaputt")) != ""


# --- RemoteClients with mocked network -------------------------------------

def test_build_openai_tools_namespaces_and_skips_failures(monkeypatch):
    fake_tools = [
        {"name": "get_experiment", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "list_items", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
    ]

    async def fake_fetch(url, token, mode):
        return fake_tools

    monkeypatch.setattr(mcp_clients, "_fetch_tools", fake_fetch)
    rc = mcp_clients.RemoteClients(elab_token="tok")
    tools = rc.build_openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert names == ["elab_get_experiment", "elab_list_items"]


def test_build_openai_tools_server_error_yields_unavailable_tool(monkeypatch):
    async def boom(url, token, mode):
        raise ExceptionGroup("eg", [_Fake401()])

    monkeypatch.setattr(mcp_clients, "_fetch_tools", boom)
    rc = mcp_clients.RemoteClients(elab_token="tok")
    tools = rc.build_openai_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "elab___unavailable__"
    assert "ungültig" in tools[0]["function"]["description"]


def test_no_active_sources_yields_no_tools():
    rc = mcp_clients.RemoteClients()
    assert rc.build_openai_tools() == []


def test_call_dispatches_to_right_server(monkeypatch):
    calls = {}

    def fake_call(url, token, mode, name, arguments):
        calls["url"] = url
        calls["name"] = name
        calls["token"] = token
        calls["mode"] = mode
        return json.dumps({"ok": True})

    monkeypatch.setattr(mcp_clients, "_call_tool", fake_call)
    rc = mcp_clients.RemoteClients(elab_token="elab-tok", dt_token="dt-tok")
    assert rc.call("elab_get_experiment", {"id": 1}) == '{"ok": true}'
    assert calls["url"] == mcp_clients.ELAB_MCP_URL
    assert calls["token"] == "elab-tok"
    assert calls["name"] == "get_experiment"
    assert calls["mode"] == "query"  # elab uses query auth


def test_call_uses_header_mode_for_datatagger(monkeypatch):
    calls = {}

    def fake_call(url, token, mode, name, arguments):
        calls.update(url=url, token=token, mode=mode, name=name)
        return json.dumps({"ok": True})

    monkeypatch.setattr(mcp_clients, "_call_tool", fake_call)
    rc = mcp_clients.RemoteClients(dt_token="dt-tok")
    rc.call("dt_list_projects", {})
    assert calls["url"] == mcp_clients.DATATAGGER_MCP_URL
    assert calls["mode"] == "header"  # dt uses header auth


def test_call_unknown_prefix_returns_error_json():
    rc = mcp_clients.RemoteClients()
    out = json.loads(rc.call("nope_tool", {}))
    assert "error" in out


# --- openai_tools glue ------------------------------------------------------

def test_set_remote_clients_and_chat_tools(monkeypatch):
    # Local tools come from the real registry (needs server caches); to keep
    # this unit hermetic we only assert the remote part is appended when a
    # RemoteClients instance is installed.
    class FakeRC:
        def build_openai_tools(self):
            return [{"type": "function", "function": {"name": "elab_x", "description": "d",
                                                     "parameters": {"type": "object", "properties": {}}}}]

    openai_tools.set_remote_clients(FakeRC())
    try:
        chat = openai_tools.build_chat_tools()
        names = [t["function"]["name"] for t in chat]
        assert "elab_x" in names
        # local tools still present (server registry)
        assert any(not n.startswith(("elab_", "dt_")) for n in names)
    finally:
        openai_tools.set_remote_clients(None)


def test_call_tool_remote_without_client_returns_error_json():
    openai_tools.set_remote_clients(None)
    out = json.loads(openai_tools.call_tool("elab_get_experiment", {}))
    assert "error" in out


def test_call_tool_remote_dispatches(monkeypatch):
    class FakeRC:
        def call(self, name, arguments):
            return json.dumps({"dispatched": name})

    openai_tools.set_remote_clients(FakeRC())
    try:
        out = json.loads(openai_tools.call_tool("elab_get_experiment", {"id": 1}))
        assert out == {"dispatched": "elab_get_experiment"}
    finally:
        openai_tools.set_remote_clients(None)
