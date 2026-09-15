"""Remote MCP clients for the e-conversion chat.

The chat talks to the *already running* MCP servers on the researchmcp stack
(eLabFTW via the elabmcp-proxy, DataTagger via the datatagger-proxy) as an MCP
*client* -- it never re-implements their tools.  Each proxy issues a personal,
HMAC-signed JWT (via its /register page) that embeds the user's credentials and
the enabled-tools selection; the proxy filters ``tools/list`` by that token, so
whatever the token allows is exactly what the chat exposes.

Design notes
------------
* Tokens are per-user, per-session (BYOK).  They are kept in Streamlit
  session state -- never logged.
* Auth transport differs per proxy:
    - elabmcp-proxy reads the token ONLY from the URL query (?token=...),
      not from a header (verified 2026-09).
    - datatagger-proxy accepts the token as query param OR as
      Authorization: Bearer header.
  So each endpoint carries its own auth mode; mcp_clients applies it.
* The remote tool schemas are converted to OpenAI function-calling schemas and
  namespaced (``elab_*`` / ``dt_*``) so they cannot collide with the local
  paper tools.  ``call_tool`` dispatches on that prefix.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from logging_config import get_logger

log = get_logger("mcp_clients")

# The two *already running* MCP endpoints on the researchmcp stack.
# URL_PREFIX of each proxy is baked into its public path (/el, /dt).
ELAB_MCP_URL = "https://researchmcp.duckdns.org/el/mcp"
DATATAGGER_MCP_URL = "https://researchmcp.duckdns.org/dt/mcp"

# Auth mode per endpoint:
#   "query"  -> token appended as ?token=... (elabmcp-proxy ONLY reads query)
#   "header" -> Authorization: Bearer <token> (datatagger-proxy)
AUTH_MODES = {
    "elab": "query",
    "dt": "header",
}

# Prefixes used to namespace remote tools in the OpenAI tool list and to
# dispatch calls back to the right server.
PREFIXES = {
    "elab": ELAB_MCP_URL,
    "dt": DATATAGGER_MCP_URL,
}


def _auth(url: str, token: str, mode: str) -> tuple[str, dict[str, str]]:
    """Return (url_with_token, headers) for the given auth mode."""
    if mode == "query":
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}token={token}", {}
    return url, {"Authorization": f"Bearer {token}"}


def _to_openai_schema(name: str, tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one MCP tool definition to an OpenAI function schema."""
    fn = {
        "name": name,
        "description": tool.get("description") or "",
        "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
    }
    # Some MCP servers omit top-level type; OpenAI requires it.
    fn["parameters"].setdefault("type", "object")
    fn["parameters"].setdefault("properties", {})
    return {"type": "function", "function": fn}


async def _fetch_tools(url: str, token: str, mode: str) -> list[dict[str, Any]]:
    """Return the raw MCP tool list from one server (already JWT-filtered)."""
    url_auth, headers = _auth(url, token, mode)
    async with AsyncExitStack() as stack:
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(url_auth, headers=headers)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await session.list_tools()
        return [t.model_dump() for t in tools.tools]


def _call_tool(url: str, token: str, mode: str, name: str, arguments: dict[str, Any]) -> str:
    """Call one tool on one remote server; always returns a JSON string."""
    import anyio

    async def _run() -> str:
        url_auth, headers = _auth(url, token, mode)
        async with AsyncExitStack() as stack:
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(url_auth, headers=headers)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            res = await session.call_tool(name, arguments)
            # Normalize the MCP result (text content blocks + structuredContent)
            # into a single JSON string for the OpenAI tool-message content.
            parts: list[str] = []
            if res.content:
                for block in res.content:
                    txt = getattr(block, "text", None)
                    if txt is not None:
                        parts.append(str(txt))
            if res.structuredContent:
                parts.append(json.dumps(res.structuredContent, ensure_ascii=False))
            if not parts:
                parts.append(json.dumps({"result": "ok"}))
            return "\n".join(parts)

    try:
        return anyio.run(_run)
    except Exception as exc:  # noqa: BLE001
        log.error("remote mcp call failed", exc_info=True,
                  extra={"fields": {"url": url, "tool": name}})
        msg = _friendly_error(exc)
        return json.dumps({"error": f"Remote tool {name} failed: {msg}"})


def _friendly_error(exc: Exception) -> str:
    """Turn transport/HTTP exceptions into a short, actionable message."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "401" in low or "unauthorized" in low:
        return "Token invalid or expired — please register a new one via /el/register or /dt/register."
    if "403" in low or "forbidden" in low:
        return "Not authorized (403) — this token does not allow the action."
    if "404" in low or "not found" in low:
        return "MCP endpoint not found (404)."
    if "timed out" in low or "timeout" in low or "connection refused" in low:
        return "Server unreachable (timeout/connection)."
    # ExceptionGroup nests the real HTTP error in its sub-exceptions.
    # Walk the tree (unpack .exceptions recursively) and re-check.
    for sub in getattr(exc, "exceptions", []) or []:
        nested = _friendly_error(sub)
        if nested != text[:200]:
            return nested
    # Strip the ExceptionGroup noise: keep the first meaningful line.
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(("|", "+", "File", "raise", "httpx", "mcp", "for more")):
            return line[:200]
    return text[:200]


# ---------------------------------------------------------------------------
# Public API used by openai_tools.py / app.py
# ---------------------------------------------------------------------------

class RemoteClients:
    """Hold per-user tokens and fetch/call tools on the remote servers."""

    def __init__(self, elab_token: str | None = None, dt_token: str | None = None):
        self.elab_token = elab_token
        self.dt_token = dt_token

    @property
    def active(self) -> list[tuple[str, str, str, str]]:
        """(prefix, url, token, auth_mode) for each server the user has a token for."""
        out: list[tuple[str, str, str, str]] = []
        if self.elab_token:
            out.append(("elab", ELAB_MCP_URL, self.elab_token, AUTH_MODES["elab"]))
        if self.dt_token:
            out.append(("dt", DATATAGGER_MCP_URL, self.dt_token, AUTH_MODES["dt"]))
        return out

    def build_openai_tools(self) -> list[dict[str, Any]]:
        """Fetch remote tools and return OpenAI schemas (namespaced)."""
        schemas: list[dict[str, Any]] = []
        for prefix, url, token, mode in self.active:
            try:
                import anyio
                tools = anyio.run(_fetch_tools, url, token, mode)
            except Exception as exc:  # noqa: BLE001
                log.error("list remote tools failed", exc_info=True,
                          extra={"fields": {"url": url}})
                # One failing server must not break the chat; expose a clear
                # status instead of the raw exception.
                schemas.append(_error_tool(prefix, _friendly_error(exc)))
                continue
            for t in tools:
                name = f"{prefix}_{t.get('name', '')}"
                schemas.append(_to_openai_schema(name, t))
        return schemas

    def call(self, prefixed_name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a namespaced tool name to the right remote server."""
        prefix, _, raw_name = prefixed_name.partition("_")
        url = PREFIXES.get(prefix)
        mode = AUTH_MODES.get(prefix)
        token = self.elab_token if prefix == "elab" else self.dt_token if prefix == "dt" else None
        if not url or not token or not mode or not raw_name:
            return json.dumps({"error": f"Unknown remote tool: {prefixed_name}"})
        return _call_tool(url, token, mode, raw_name, arguments)


def _error_tool(prefix: str, err: str) -> dict[str, Any]:
    """A placeholder tool so the LLM sees why a source is unavailable."""
    name = f"{prefix}___unavailable__"
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Source {prefix} is currently unavailable. Error: {err}",
            "parameters": {"type": "object", "properties": {}},
        },
    }
