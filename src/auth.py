"""Sessions and identity for the web app.

A session is a server-side record keyed by a random cookie. It holds the
conversation, the model choice and the user's bring-your-own tokens for the
remote MCP sources, so tokens are scoped to one browser session and never to
the process.

Identity is separate from the session and deliberately thin: ``get_identity``
is the single hook where a real login (OIDC, NOMAD SSO) plugs in later. Today
it reads an optional proxy header (nginx ``auth_request`` deployments can
forward one) and otherwise yields an anonymous identity.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, HTTPException, Request, Response

import telemetry

COOKIE_NAME = "econverse_sid"
SESSION_TTL_S = int(os.environ.get("SESSION_TTL_S", str(12 * 3600)))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "5000"))
# Header carrying the proxy-authenticated user, if the deployment forwards one.
AUTH_HEADER = os.environ.get("AUTH_HEADER", "X-Forwarded-User")
# Cached remote tool schemas are refreshed after this long (tokens expire; the
# proxies' tool selection can change).
REMOTE_SCHEMAS_TTL_S = 600


@dataclass
class Identity:
    subject: str | None = None
    display: str = ""

    @property
    def anonymous(self) -> bool:
        return not self.subject


@dataclass
class Session:
    id: str                     # telemetry id (short, anonymous) -- what goes into logs
    cookie_id: str              # the browser's secret -- never logged
    identity: Identity
    created: float
    last_seen: float
    messages: list[dict] = field(default_factory=list)
    provider_name: str | None = None
    model_name: str | None = None
    elab_token: str | None = None
    dt_token: str | None = None
    elab_tools: int = 0
    dt_tools: int = 0
    remote: Any = None                          # mcp_clients.RemoteClients or None
    remote_schemas: list[dict] | None = None    # cached remote.build_openai_tools()
    remote_schemas_at: float = 0.0
    turns: int = 0
    tool_calls_total: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    turn_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    cancel: threading.Event | None = field(default=None, repr=False)

    def invalidate_remote_schemas(self) -> None:
        self.remote_schemas = None
        self.remote_schemas_at = 0.0

    def remote_schemas_fresh(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.remote_schemas is not None and now - self.remote_schemas_at < REMOTE_SCHEMAS_TTL_S


class SessionStore:
    """In-memory session table. One process, one store; swap for Redis behind this API."""

    def __init__(self, ttl_s: float = SESSION_TTL_S, max_sessions: int = MAX_SESSIONS,
                 clock=time.monotonic) -> None:
        self._by_cookie: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._clock = clock

    def __len__(self) -> int:
        return len(self._by_cookie)

    def _sweep(self, now: float) -> None:
        expired = [k for k, s in self._by_cookie.items() if now - s.last_seen > self.ttl_s]
        for k in expired:
            del self._by_cookie[k]
        if len(self._by_cookie) > self.max_sessions:
            # evict least recently seen
            for k, _ in sorted(self._by_cookie.items(), key=lambda kv: kv[1].last_seen)[
                    : len(self._by_cookie) - self.max_sessions]:
                del self._by_cookie[k]

    def get(self, cookie_id: str | None) -> Session | None:
        if not cookie_id:
            return None
        now = self._clock()
        with self._lock:
            self._sweep(now)
            s = self._by_cookie.get(cookie_id)
            if s is not None:
                s.last_seen = now
            return s

    def create(self, identity: Identity | None = None) -> Session:
        now = self._clock()
        s = Session(id=telemetry.new_session_id(), cookie_id=secrets.token_urlsafe(32),
                    identity=identity or Identity(), created=now, last_seen=now)
        with self._lock:
            self._sweep(now)
            self._by_cookie[s.cookie_id] = s
        return s

    def reset(self, session: Session) -> None:
        """Start a new conversation: clears history and mints a new telemetry id.

        Tokens and the model choice survive, like closing and reopening a chat
        while staying logged in to the sources.
        """
        with session.lock:
            session.messages = []
            session.turns = 0
            session.tool_calls_total = 0
            session.id = telemetry.new_session_id()


def get_identity(request: Request) -> Identity:
    """The future SSO hook. Proxy header today, anonymous otherwise."""
    subject = (request.headers.get(AUTH_HEADER) or "").strip() if AUTH_HEADER else ""
    return Identity(subject=subject or None, display=subject)


def _cookie_path(request: Request) -> str:
    return request.scope.get("root_path") or "/"


def _cookie_secure(request: Request) -> bool:
    if os.environ.get("COOKIE_SECURE", "").strip().lower() in ("1", "true", "yes"):
        return True
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto == "https"


def get_session(request: Request, response: Response,
                identity: Identity = Depends(get_identity)) -> Session:
    """The caller's session, created (and its cookie set) on first contact.

    Use on JSON endpoints only: FastAPI merges the dependency's cookie into a
    normal response but not into a StreamingResponse, so the streaming chat
    endpoint requires an existing session instead (``require_session``).
    """
    store: SessionStore = request.app.state.sessions
    session = store.get(request.cookies.get(COOKIE_NAME))
    if session is None:
        session = store.create(identity)
        response.set_cookie(
            COOKIE_NAME, session.cookie_id, max_age=int(store.ttl_s), httponly=True,
            samesite="lax", secure=_cookie_secure(request), path=_cookie_path(request),
        )
    elif identity.subject and session.identity.subject != identity.subject:
        session.identity = identity
    return session


def require_session(request: Request) -> Session:
    store: SessionStore = request.app.state.sessions
    session = store.get(request.cookies.get(COOKIE_NAME))
    if session is None:
        raise HTTPException(status_code=401, detail={"error": "session_required"})
    return session
