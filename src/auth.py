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


class TurnSlot:
    """The one in-flight chat turn of a session, and who is allowed to end it.

    Cancellation is cooperative -- the worker polls ``cancel`` between stream
    chunks -- so a turn stuck on a stalled model read or a slow remote tool can
    stay alive for minutes after a stop. The interface must not be hostage to
    that: ``start`` cancels the sitting turn, gives it a moment to hand over,
    and claims the slot regardless, while ``abandon`` (new chat) frees it
    outright. Each claim bumps ``epoch``; the worker that was displaced sees a
    stale epoch, winds down on its own, and never re-locks the session.
    """

    HANDOVER_S = 1.5

    def __init__(self) -> None:
        self._mu = threading.Lock()
        self._handed_over = threading.Event()
        self._handed_over.set()
        self.epoch = 0
        self.busy = False
        self.cancel: threading.Event | None = None

    def _claim(self) -> tuple[int, threading.Event]:
        self.epoch += 1
        self.busy = True
        self.cancel = threading.Event()
        self._handed_over.clear()
        return self.epoch, self.cancel

    def start(self, handover_s: float | None = None) -> tuple[int, threading.Event]:
        """Claim the slot for a new turn, displacing a running one."""
        with self._mu:
            running = self.cancel if self.busy else None
        if running is not None:
            running.set()
            self._handed_over.wait(self.HANDOVER_S if handover_s is None else handover_s)
        with self._mu:
            return self._claim()

    def finish(self, epoch: int) -> bool:
        """Release the slot. False when this turn was displaced meanwhile."""
        with self._mu:
            if epoch != self.epoch:
                return False
            self.busy = False
            self.cancel = None
            self._handed_over.set()
            return True

    def stop(self) -> bool:
        """Ask the running turn to stop. It may take a moment to notice."""
        with self._mu:
            cancel = self.cancel if self.busy else None
        if cancel is None:
            return False
        cancel.set()
        return True

    def abandon(self) -> bool:
        """Free the slot now; the displaced turn's result is dropped."""
        with self._mu:
            if not self.busy:
                return False
            if self.cancel is not None:
                self.cancel.set()
            self.epoch += 1
            self.busy = False
            self.cancel = None
            self._handed_over.set()
            return True


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
    # Per-session LLM parameter overrides (see llm.PARAM_SPEC); only the values
    # that differ from config.toml are stored.
    llm_params: dict[str, object] = field(default_factory=dict)
    elab_tools: int = 0
    dt_tools: int = 0
    remote: Any = None                          # mcp_clients.RemoteClients or None
    remote_schemas: list[dict] | None = None    # cached remote.build_openai_tools()
    remote_schemas_at: float = 0.0
    turns: int = 0
    tool_calls_total: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    turn: TurnSlot = field(default_factory=TurnSlot, repr=False)

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
