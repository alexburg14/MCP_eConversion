"""The MCP->OpenAI bridge and server health.

These encode the invariant that replaced the old hand-maintained duplicate: the
OpenAI schemas are derived from the one registry, so they cannot drift and every
parameter carries its help.
"""
import json

import openai_tools
import server


def test_bridge_exposes_every_registered_tool():
    registered = {t.name for t in server.mcp._tool_manager.list_tools()}
    derived = {t["function"]["name"] for t in openai_tools.build_openai_tools()}
    assert derived == registered
    assert len(derived) >= 13


def test_every_tool_parameter_has_a_description():
    # The reason to move help into the signatures — a bare param would ship an
    # undocumented argument to the model.
    for t in openai_tools.build_openai_tools():
        for pname, spec in t["function"]["parameters"].get("properties", {}).items():
            assert spec.get("description"), f"{t['function']['name']}.{pname} has no description"


def test_call_tool_unknown_returns_error_json():
    assert "error" in json.loads(openai_tools.call_tool("no_such_tool", {}))


def test_call_tool_applies_defaults_for_omitted_optionals():
    # collaboration_centrality(top_k=10) — omitting top_k must use the default,
    # not raise, now that dispatch is fn(**args).
    out = json.loads(openai_tools.call_tool("collaboration_centrality", {}))
    assert isinstance(out, list) and len(out) >= 1


def test_server_status_reports_all_caches():
    status = json.loads(server.server_status())
    assert status["cluster"]
    assert status["tools"] == len(server.mcp._tool_manager.list_tools())
    # every cache in the status registry reports an availability flag
    for name, entry in status["caches"].items():
        assert "available" in entry


def test_cache_status_contract():
    # Structure, not presence: works whether or not a given cache is built.
    expected = {"papers", "abstracts", "fulltext", "pis", "embeddings", "graph"}
    assert set(server.CACHE_STATUS) == expected
    for entry in server.CACHE_STATUS.values():
        assert isinstance(entry["available"], bool)
