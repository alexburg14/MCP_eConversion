"""Bridge tool registries to OpenAI function-calling.

Local tools are defined once in server.py via ``@mcp.tool()``. Remote tools
come from the *already running* MCP servers (eLabFTW proxy, DataTagger proxy)
via ``mcp_clients.RemoteClients``; they are fetched per user session because
each user holds their own JWT.

This module derives both the OpenAI-format schema list and the name->callable
dispatch from those registries, so the chat app never re-declares them.
(Until 2026-08 both were hand-written in app.py in parallel and had already
drifted; deriving them makes drift structurally impossible.)

``call_tool`` is the single invocation path — it centralizes argument handling,
error trapping, and per-call logging.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import server
from logging_config import get_logger

log = get_logger("tools")

# Remote client holder. app.py sets this once per user session (tokens live in
# Streamlit session state). Kept as module state so the existing call path
# (call_tool) stays unchanged.
_remote: Any = None


def set_remote_clients(remote: Any) -> None:
    """Install the session's RemoteClients (tokens already attached)."""
    global _remote
    _remote = remote


def get_remote_clients() -> Any:
    return _remote


def _registry() -> list:
    # Private FastMCP accessor. If the mcp package reshapes this, only this one
    # line needs updating (mcp is pinned in requirements.txt).
    return server.mcp._tool_manager.list_tools()


def build_openai_tools() -> list[dict]:
    """OpenAI ``tools=`` list, derived from the *local* tool registry only.

    Kept for backward compatibility / tests. The chat uses build_chat_tools().
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters,
            },
        }
        for t in _registry()
    ]


def build_chat_tools() -> list[dict]:
    """Local tools + remote tools (if a RemoteClients instance is installed).

    Remote tools are namespaced (``elab_*`` / ``dt_*``) by mcp_clients so they
    cannot collide with local names. A failing remote source degrades to a
    single descriptive ``<prefix>___unavailable__`` tool instead of breaking
    the chat.
    """
    tools = build_openai_tools()
    if _remote is not None:
        tools.extend(_remote.build_openai_tools())
    return tools


def _dispatch() -> dict[str, Callable[..., str]]:
    return {t.name: t.fn for t in _registry()}


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a tool by name with a kwargs dict; always returns a JSON string.

    Namespaced remote names (``elab_*`` / ``dt_*``) are dispatched to the
    installed RemoteClients; everything else goes to the local registry.

    Unknown tools, bad arguments, and tool exceptions are caught and returned
    as error JSON so a single failing tool call never aborts the chat loop.
    """
    # Remote dispatch first — prefix decides, no local fallback for these.
    if name.startswith(("elab_", "dt_")):
        if _remote is not None:
            return _remote.call(name, arguments)
        log.warning("remote tool without client", extra={"fields": {"tool": name}})
        return json.dumps({"error": f"No remote session for tool: {name}"})

    fn = _dispatch().get(name)
    if fn is None:
        log.warning("unknown tool", extra={"fields": {"tool": name}})
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
    except TypeError as exc:
        log.warning("tool bad args", extra={"fields": {"tool": name, "error": str(exc)}})
        return json.dumps({"error": f"Bad arguments for {name}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        log.error("tool failed", exc_info=True, extra={"fields": {"tool": name}})
        return json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"})
    log.info("tool call", extra={"fields": {"tool": name, "args": arguments}})
    return result
