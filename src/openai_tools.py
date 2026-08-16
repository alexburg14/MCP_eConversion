"""Bridge the FastMCP tool registry to OpenAI function-calling.

Tools are defined once in server.py via ``@mcp.tool()``. This module derives
both the OpenAI-format schema list and the name->callable dispatch from that one
registry, so the chat app never re-declares them. (Until 2026-08 both were
hand-written in app.py in parallel and had already drifted; deriving them makes
drift structurally impossible.)

``call_tool`` is the single invocation path — it centralizes argument handling,
error trapping, and per-call logging.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import server
from logging_config import get_logger

log = get_logger("tools")


def _registry() -> list:
    # Private FastMCP accessor. If the mcp package reshapes this, only this one
    # line needs updating (mcp is pinned in requirements.txt).
    return server.mcp._tool_manager.list_tools()


def build_openai_tools() -> list[dict]:
    """OpenAI ``tools=`` list, derived from the tool registry."""
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


def _dispatch() -> dict[str, Callable[..., str]]:
    return {t.name: t.fn for t in _registry()}


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a tool by name with a kwargs dict; always returns a JSON string.

    Unknown tools, bad arguments, and tool exceptions are caught and returned as
    error JSON so a single failing tool call never aborts the chat loop.
    """
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
