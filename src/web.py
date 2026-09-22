"""FastAPI application: JSON API, streaming chat over SSE, static frontend.

``create_app(state)`` is an app factory. ``AppState`` carries everything the
routes need (config, tool schemas, callables into the corpus) so tests can
inject fakes and never import ``server`` (which loads the caches). The real
state comes from ``AppState.from_server()``, called once from ``main.py`` in
the process main thread -- see the note there about the torch preload.

The chat turn runs in a worker thread (see ``agent``); this module only
bridges its events to the HTTP response and owns the session bookkeeping
(history, telemetry line) around it.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
import auth
import llm
import mcp_clients
import prompts
import telemetry
from config import Config, get_config
from logging_config import configure_logging, get_logger

log = get_logger("web")

APP_TITLE = "e-Converse"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CHAT_WORKERS = int(os.environ.get("CHAT_WORKERS", "8"))
SSE_PING_S = 15
STATS_TTL_S = 30
MAX_PROMPT_CHARS = 20_000
MAX_FEEDBACK_CHARS = 5_000
FEEDBACK_CATEGORIES = ("Bug report", "General feedback")

# Clickable starters shown while the conversation is empty. Sourced from the
# "Questions for MCP.md" test-question list (Obsidian, Meeting eConversion);
# a random subset of N_EXAMPLES is served per /api/config call.
EXAMPLE_QUESTIONS = [
    "How many publications does the e-conversion database contain?",
    "Which groups in e-conversion are working with electronic structure theory?",
    "What is the research focus of the group by Prof Rinke?",
    "Which paper has the most co-authors from within e-conversion?",
    "How many papers were published in Nature journals?",
    "Which groups are working on similar topics to Prof. Rinke, based on their publication abstracts?",
    "What are the main open scientific challenges mentioned across e-conversion papers?",
    "Which papers would be most relevant to someone new to the field wanting to understand the role of interfaces in energy conversion?",
    "Are there papers that explicitly mention machine learning being applied to experimental data, and if so which experimental techniques appear most often in that context?",
    "Which two groups have the most complementary research — i.e., one produces data/methods the other could use?",
    "Are there topics that appear in multiple group descriptions on the website but have few or no corresponding papers?",
    "Which experimental groups have published jointly with theory/simulation groups?",
    "How has the share of ML-related publications changed over the years in e-conversion?",
    "Which topics appear in early papers (pre-2020) but not recent ones, and vice versa?",
    "Has the collaboration density within the consortium increased over time?",
    "Which PI has the highest h-index?",
    "Can you create a connecting graph of the PIs in e-conversion based on collaborative papers?",
    'Who acts as the most central "bridge" between otherwise disconnected groups?',
    "Are there clusters of groups that only collaborate internally and rarely with others?",
    "Which PI has collaborated with the most different groups, vs. the most papers with a single group?",
]
N_EXAMPLES = 4

SOURCES = {
    "elab": {
        "label": "eLabFTW",
        # internal: the app registers the key for the user (POST /api/session/register)
        "register_api": os.environ.get("ELAB_REGISTER_API", "http://elabmcp-proxy:8081/register"),
        "base_url_default": "https://elntest.ub.tum.de",
        "key_label": "elabFTW API key",
        "profiles": {"h": "Hybrid (recommended)", "r": "Read-only", "f": "Full"},
        "test_token_env": "ELABFTW_TEST_TOKEN",
    },
    "dt": {
        "label": "DataTagger",
        "register_api": os.environ.get("DT_REGISTER_API", "http://datatagger-proxy:8000/register"),
        "base_url_default": "https://datatagger.ub.tum.de",
        "key_label": "DataTagger API token",
        "profiles": {},
        "test_token_env": "DATATAGGER_TEST_TOKEN",
    },
}

# Fields a source may expose to the browser (never tokens or internal URLs).
# "profiles" is built explicitly below -- it must always serialise as a list.
_SOURCE_UI_KEYS = ("label", "base_url_default", "key_label")

# The proxies answer /register with an HTML page whose only machine-readable part
# is the MCP URL; pull the token out of it.
_TOKEN_RE = re.compile(r"[?&]token=([A-Za-z0-9._\-]+)")
_REGISTER_MESSAGES = {
    400: "Registration rejected — check the base URL.",
    401: "The API key was rejected by the service.",
    403: "The key is valid but has no access.",
    500: "The registration service is misconfigured.",
}


def _post_registration(url: str, data: dict) -> tuple[int, str]:
    """Blocking POST to a proxy's /register page (always called in a worker thread)."""
    with httpx.Client(timeout=REGISTER_TIMEOUT_S, follow_redirects=True) as client:
        response = client.post(url, data=data)
        return response.status_code, response.text


def _test_token(src: dict) -> str:
    """The deployment's test-account token, read server-side only (never logged)."""
    return (os.environ.get(src.get("test_token_env", "")) or "").strip()


def source_public(src: dict) -> dict:
    """What the browser may see about a source -- tokens and internal URLs stay here."""
    out = {k: src[k] for k in _SOURCE_UI_KEYS if k in src}
    # always a list: the browser iterates it (a {} here broke the connect dialog)
    out["profiles"] = [{"value": value, "label": label}
                       for value, label in (src.get("profiles") or {}).items()]
    if _test_token(src):
        out["test_user"] = {"label": "test account"}
    return out

# How long _post_registration waits on a proxy's /register endpoint.
REGISTER_TIMEOUT_S = 20.0

_PIPELINE_STAGES = [
    ("papers", "papers (DOI csv)", "count"),
    ("abstracts", "abstracts", "count"),
    ("fulltext", "full texts", "count"),
    ("pis", "PI profiles", "count"),
    ("embeddings", "embeddings", "flag"),
    ("graph", "collaboration graph", "flag"),
    ("proposal", "proposal", "flag"),
]


def _empty_summary() -> dict:
    return {"turns": 0, "sessions": 0, "error_turns": 0, "avg_latency_ms": 0,
            "models": {}, "providers": {}, "tools": {}, "feedback": 0}


@dataclass
class AppState:
    cfg: Config
    system_prompt: str
    n_papers: int
    n_pis: int
    local_tools: list[dict]
    call_tool: Callable[..., str]                      # (name, args, remote) -> json str
    client_factory: Callable[[str, str], Any] = llm.make_client
    remote_factory: Callable[..., Any] = mcp_clients.RemoteClients
    corpus_map: Callable[[int], dict] | None = None
    corpus_map_available: Callable[[], bool] = lambda: False
    collab_graph: Callable[[], dict | None] = lambda: None
    coverage: Callable[[], dict] = lambda: {}
    summarize: Callable[[], dict] = _empty_summary
    sessions: auth.SessionStore = field(default_factory=auth.SessionStore)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(CHAT_WORKERS, thread_name_prefix="chat"))
    example_pool: list[str] = field(default_factory=lambda: list(EXAMPLE_QUESTIONS))
    sources: dict = field(default_factory=lambda: {k: dict(v) for k, v in SOURCES.items()})
    stats_cache: dict = field(default_factory=lambda: {"at": 0.0, "value": None})
    stats_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_server(cls) -> "AppState":
        """The production state. Imports ``server`` (loads every cache) -- call
        from the main thread at process start only."""
        import openai_tools
        import server
        import views

        cfg = get_config()
        local_tools = openai_tools.build_openai_tools()
        telemetry.log_startup(tools_local=len(local_tools), tools_remote=0,
                              providers=list(cfg.providers or {}), models=list(cfg.llm.models))
        return cls(
            cfg=cfg,
            system_prompt=prompts.build_system_prompt(cfg, len(server.papers), len(server._PIS)),
            n_papers=len(server.papers), n_pis=len(server._PIS),
            local_tools=local_tools, call_tool=openai_tools.call_tool,
            corpus_map=views.corpus_map_payload, corpus_map_available=views.corpus_map_available,
            collab_graph=views.collab_graph_payload,
            coverage=telemetry.coverage, summarize=telemetry.summarize,
        )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)


class ModelRequest(BaseModel):
    provider: str
    model: str = ""     # "" = provider default (OpenRouter: cheapest eligible)
    sort: str = ""      # OpenRouter route: price | throughput | latency


class RegisterRequest(BaseModel):
    base_url: str = ""
    api_key: str = ""
    profile: str = ""


class ParamsRequest(BaseModel):
    # Only the values that differ from the defaults; "" clears one back to them.
    params: dict[str, object] = Field(default_factory=dict)


class TokenRequest(BaseModel):
    token: str = ""     # "" disconnects


class FeedbackRequest(BaseModel):
    category: str
    text: str = Field(min_length=1, max_length=MAX_FEEDBACK_CHARS)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _selection(state: AppState, session: auth.Session) -> dict:
    """Provider, model and parameters as chosen -- no network (no auto-pick)."""
    providers = state.cfg.providers or {}
    name = session.provider_name if session.provider_name in providers else llm.default_provider_name(state.cfg)
    prov = providers.get(name)
    if prov is None:
        out = {"provider": name, "model": state.cfg.llm.default_model, "auto": False}
    else:
        models = list(prov.models) or [prov.default_model]
        if session.model_name in models:
            out = {"provider": name, "model": session.model_name, "auto": False}
        elif "openrouter" in prov.base_url:
            out = {"provider": name, "model": "", "auto": True}
        else:
            out = {"provider": name, "model": prov.default_model, "auto": False}
    out["params"] = llm.effective_params(state.cfg, session.llm_params)
    if prov is not None and "openrouter" in prov.base_url:
        route = out["params"]["provider_sort"]
        out["sort"] = route
        out["route_label"] = llm.route_label(route)
        # the concrete model the route resolves to, from the cached catalogue
        # (never fetched here: _selection must not do network I/O)
        out["auto_model"] = True
        out["resolved_model"] = llm.cached_cheapest_model(
            api_key=llm.provider_api_key(prov.api_key_env))
    return out


def _tools_available(state: AppState, session: auth.Session) -> dict:
    out = {
        "local": len(state.local_tools),
        "elab": session.elab_tools if session.elab_token else 0,
        "dt": session.dt_tools if session.dt_token else 0,
    }
    out["total"] = sum(out.values())
    return out


def _session_view(state: AppState, session: auth.Session) -> dict:
    with session.lock:
        messages = [dict(m) for m in session.messages]
        turns = session.turns
    sel = _selection(state, session)
    return {
        "session": session.id,
        "user": session.identity.display or None,
        "provider": sel["provider"], "model": sel["model"], "auto_model": sel["auto"],
        "sort": sel.get("sort"), "route_label": sel.get("route_label"),
        "resolved_model": sel.get("resolved_model"),
        "params": sel["params"],
        "connected": {
            kind: {"active": bool(getattr(session, f"{kind}_token")),
                   "tools": getattr(session, f"{kind}_tools") if getattr(session, f"{kind}_token") else 0}
            for kind in state.sources
        },
        "tools": _tools_available(state, session),
        "turns": turns,
        "busy": session.turn.busy,
        "messages": messages,
    }


def _rebuild_remote(state: AppState, session: auth.Session) -> None:
    if session.elab_token or session.dt_token:
        session.remote = state.remote_factory(elab_token=session.elab_token or None,
                                              dt_token=session.dt_token or None)
    else:
        session.remote = None
    session.invalidate_remote_schemas()


def _remote_schemas(session: auth.Session) -> list[dict]:
    """The session's remote tool schemas, fetched at most once per TTL."""
    if session.remote is None:
        return []
    if session.remote_schemas_fresh():
        return session.remote_schemas or []
    schemas = session.remote.build_openai_tools()
    session.remote_schemas = schemas
    session.remote_schemas_at = time.monotonic()
    return schemas


def _count_source_tools(schemas: list[dict], kind: str) -> tuple[int, str | None]:
    """(usable tool count, error text) for one source in a schema list."""
    prefix = kind + "_"
    found = 0
    error = None
    for t in schemas:
        name = t["function"]["name"]
        if not name.startswith(prefix):
            continue
        if "__unavailable__" in name:
            desc = t["function"].get("description") or ""
            error = desc.partition("Error:")[2].strip() or desc
        else:
            found += 1
    return found, error


# ---------------------------------------------------------------------------
# The chat turn (worker thread)
# ---------------------------------------------------------------------------

def _run_turn_worker(state: AppState, session: auth.Session, resolved: dict, prompt: str,
                     turn: int, api_msgs: list[dict], put: Callable[[Any], None],
                     cancel: threading.Event, epoch: int, convo: list[dict],
                     user_msg: dict) -> None:
    """Own one turn end to end: stream events, then commit history + telemetry.

    ``convo`` is the message list this turn was started on and ``user_msg`` the
    question it answers -- the answer is filed directly after it, by identity,
    so a displaced turn (the user sent another prompt before this one finished)
    lands under its own question however the list grew meanwhile. A turn whose
    conversation was reset finds ``convo`` detached and files nothing.
    """
    started = time.perf_counter()
    final: dict | None = None
    try:
        tools = state.local_tools + _remote_schemas(session)
        system_prompt = state.system_prompt + prompts.remote_system_note(session.remote)
        client = state.client_factory(resolved["api_key"], resolved["base_url"])
        put({"type": "start", "turn": turn, "session": session.id,
             "provider": resolved["provider"], "model": resolved["model"]})
        for ev in agent.run_turn(
            client, resolved["model"], api_msgs, system_prompt=system_prompt, tools=tools,
            call_tool=lambda name, args: state.call_tool(name, args, session.remote),
            extra=resolved.get("extra"), cancel=cancel, base_url=resolved["base_url"],
        ):
            put(ev)
            if ev["type"] in ("done", "error"):
                final = ev
    except Exception as exc:  # noqa: BLE001 -- setup failed before the loop ran (client, remote tools)
        log.error("turn setup failed", exc_info=True, extra={"fields": {"session": session.id, "turn": turn}})
        final = {"answer": f"Error: {exc}", "elapsed": time.perf_counter() - started, "rounds": 0,
                 "tools": [], "usage": {}, "tool_calls": [], "error": type(exc).__name__}
        put({"type": "error", "message": str(exc), "error_type": type(exc).__name__})
    finally:
        if final is None:
            final = {"answer": "", "elapsed": time.perf_counter() - started, "rounds": 0,
                     "tools": [], "usage": {}, "tool_calls": [], "error": "no_result"}
        meta = {"model": resolved["model"], "elapsed": final["elapsed"], "tool_calls": final["tool_calls"]}
        with session.lock:
            # False once "new chat" replaced the list this turn was writing to
            at = _index_of(convo, user_msg) if convo is session.messages else None
            if at is not None:
                convo.insert(at + 1, {"role": "assistant", "content": final["answer"], "meta": meta})
                session.turns = max(session.turns, turn)
                session.tool_calls_total += len(final["tool_calls"])
            tools_available = _tools_available(state, session)
        # Telemetry: exactly one privacy-safe line per turn (hashes and lengths
        # only). Raw prompt/answer text never enters the log.
        telemetry.log_turn(
            session=session.id, turn=turn, provider=resolved["provider"], model=resolved["model"],
            base_url=resolved["base_url"], tools_available=tools_available,
            rounds=final["rounds"], tools=final["tools"], prompt=prompt, answer=final["answer"],
            usage=final["usage"], latency_ms=(final["elapsed"] or 0) * 1000, error=final["error"],
        )
        session.turn.finish(epoch)
        put(None)


def _index_of(messages: list[dict], msg: dict) -> int | None:
    """Position of ``msg`` by identity -- indices shift when a displaced turn
    files its answer between the messages of a later one."""
    for i, m in enumerate(messages):
        if m is msg:
            return i
    return None


def _sse_frame(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


class RevalidatingStatic(StaticFiles):
    """Static assets with ``Cache-Control: no-cache``.

    The frontend carries no version in its URLs, so a heuristically cached
    app.js survives a deploy and the new code silently does not run. ``no-cache``
    means "revalidate", not "do not cache": with the ETag that costs one 304.
    """

    def file_response(self, *args, **kwargs) -> Any:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(state: AppState | None = None, root_path: str | None = None) -> FastAPI:
    """Build the app. ``root_path`` (or ``ROOT_PATH`` env) mounts everything under
    a URL prefix for deployments where the reverse proxy does not strip it."""
    llm.load_dotenv()
    configure_logging()
    if state is None:
        state = AppState.from_server()
    if root_path is None:
        root_path = os.environ.get("ROOT_PATH", "").strip()
    root_path = root_path.rstrip("/")

    app = FastAPI(title=APP_TITLE, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.sessions = state.sessions
    app.state.app_state = state
    get_session = auth.get_session
    require_session = auth.require_session

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        return {"ok": True, **telemetry.version_info(), "coverage": state.coverage()}

    @app.get("/api/config")
    def config():
        providers = {
            name: {"models": list(p.models), "default_model": p.default_model,
                   "base_url": p.base_url, "note": p.note,
                   "openrouter": "openrouter" in p.base_url}
            for name, p in (state.cfg.providers or {}).items()
        }
        return {
            "title": APP_TITLE,
            "cluster": {"name": state.cfg.cluster.name, "display_name": state.cfg.cluster.display_name,
                        "website": state.cfg.cluster.website},
            "placeholder": f"Ask about {state.n_papers} papers across {state.n_pis} PIs/groups in the cluster…",
            "examples": random.sample(state.example_pool, min(N_EXAMPLES, len(state.example_pool))),
            "providers": providers,
            "routes": [dict(route) for route in llm.ROUTE_OPTIONS],
            "parameters": llm.param_payload(state.cfg),
            "default_provider": llm.default_provider_name(state.cfg),
            "sources": {kind: source_public(src) for kind, src in state.sources.items()},
            "max_tool_rounds": agent.MAX_TOOL_ROUNDS,
            "version": telemetry.version_info(),
        }

    @app.get("/api/session")
    def session_info(session: auth.Session = Depends(get_session)):
        return _session_view(state, session)

    @app.post("/api/session/model")
    def set_model(req: ModelRequest, session: auth.Session = Depends(get_session)):
        prov = (state.cfg.providers or {}).get(req.provider)
        if prov is None:
            raise HTTPException(400, detail={"error": f"Unknown provider: {req.provider}"})
        model = req.model.strip()
        if model and model not in prov.models:
            raise HTTPException(400, detail={"error": f"Unknown model for {req.provider}: {model}"})
        session.provider_name = req.provider
        session.model_name = model or None
        if "openrouter" in prov.base_url and req.sort in llm.ROUTE_VALUES:
            session.llm_params = {**session.llm_params,
                                  "provider_sort": llm.normalize_param("provider_sort", req.sort)}
        return _selection(state, session)

    @app.post("/api/session/params")
    def set_params(req: ParamsRequest, session: auth.Session = Depends(get_session)):
        """Store this session's parameter overrides (only deviations are kept)."""
        defaults = llm.default_params(state.cfg)
        clean: dict[str, object] = {}
        for key, value in (req.params or {}).items():
            if key not in llm.PARAM_KEYS:
                raise HTTPException(400, detail={"error": f"Unknown parameter: {key}"})
            try:
                canonical = llm.normalize_param(key, value)
            except ValueError as exc:
                raise HTTPException(400, detail={"error": str(exc)}) from None
            if canonical != defaults[key]:
                clean[key] = canonical
        with session.lock:
            session.llm_params = clean
        return {"params": llm.effective_params(state.cfg, session.llm_params),
                "defaults": defaults}

    @app.delete("/api/session/params")
    def reset_params(session: auth.Session = Depends(get_session)):
        """Back to the config.toml defaults."""
        with session.lock:
            session.llm_params = {}
        return {"params": llm.effective_params(state.cfg, session.llm_params),
                "defaults": llm.default_params(state.cfg)}

    async def _connect_with_token(kind: str, token: str, session: auth.Session) -> dict:
        """Log one source in with a token: manual paste and test account share this."""
        setattr(session, f"{kind}_token", token or None)
        setattr(session, f"{kind}_tools", 0)
        _rebuild_remote(state, session)
        if not token:
            return {"kind": kind, "active": False, "tools": 0, "error": None}
        # build_openai_tools drives its own event loop (anyio.run): keep it off ours.
        schemas = await run_in_threadpool(session.remote.build_openai_tools)
        found, error = _count_source_tools(schemas, kind)
        if not found:
            setattr(session, f"{kind}_token", None)
            _rebuild_remote(state, session)
            return {"kind": kind, "active": False, "tools": 0,
                    "error": error or "Token invalid or expired — register a new one."}
        session.remote_schemas = schemas
        session.remote_schemas_at = time.monotonic()
        setattr(session, f"{kind}_tools", found)
        # the fetch covers every active source; refresh the other one's count too
        for other in state.sources:
            if other != kind and getattr(session, f"{other}_token"):
                setattr(session, f"{other}_tools", _count_source_tools(schemas, other)[0])
        return {"kind": kind, "active": True, "tools": found, "error": None}

    @app.post("/api/session/connect/{kind}")
    async def connect(kind: str, req: TokenRequest, session: auth.Session = Depends(get_session)):
        if kind not in state.sources:
            raise HTTPException(404, detail={"error": f"Unknown source: {kind}"})
        return await _connect_with_token(kind, req.token.strip(), session)

    @app.post("/api/session/connect/{kind}/test")
    async def connect_test_account(kind: str, session: auth.Session = Depends(get_session)):
        """Sign in with the test account this deployment is configured with.

        The token comes from an environment variable on the server, so the button
        only ever carries a label and the value never reaches the browser.
        """
        if kind not in state.sources:
            raise HTTPException(404, detail={"error": f"Unknown source: {kind}"})
        token = _test_token(state.sources[kind])
        if not token:
            raise HTTPException(404, detail={"error": f"No test account configured for {kind}"})
        return await _connect_with_token(kind, token, session)

    @app.post("/api/session/register/{kind}")
    async def register_source(kind: str, req: RegisterRequest,
                              session: auth.Session = Depends(get_session)):
        """Register an API key upstream and connect in one step.

        The dialog collects base URL + API key; the proxy validates the key and
        returns a signed MCP token, which is the only thing kept here. Neither the
        key nor the token is logged, and the key is never echoed back.
        """
        src = state.sources.get(kind)
        if src is None:
            raise HTTPException(404, detail={"error": f"Unknown source: {kind}"})
        api_key = req.api_key.strip()
        base_url = (req.base_url or src.get("base_url_default") or "").strip().rstrip("/")
        if not api_key or not base_url:
            raise HTTPException(400, detail={"error": "Base URL and API key are required."})
        form = {"api_key": api_key, "base_url": base_url, "validated": "1"}
        profiles = src.get("profiles") or {}
        if profiles:
            form["profile"] = req.profile if req.profile in profiles else next(iter(profiles))
        # Step 1 (validated=0) is the only step that checks the key against the
        # upstream API. Step 2 mints the token without looking at it again, so
        # skipping this would happily connect a typo. Same two steps as the form.
        try:
            status, body = await run_in_threadpool(
                _post_registration, src["register_api"], {**form, "validated": "0"})
        except Exception as exc:  # noqa: BLE001 -- proxy down, DNS, timeout
            raise HTTPException(502, detail={
                "error": f"Registration service unreachable ({type(exc).__name__})."}) from None
        if status >= 400:
            raise HTTPException(400, detail={"error": _REGISTER_MESSAGES.get(
                status, "Registration failed — check the base URL and the key.")})
        try:
            status, body = await run_in_threadpool(_post_registration, src["register_api"], form)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, detail={
                "error": f"Registration service unreachable ({type(exc).__name__})."}) from None
        match = _TOKEN_RE.search(body or "")
        if status >= 400 or match is None:
            raise HTTPException(400, detail={"error": _REGISTER_MESSAGES.get(
                status, "Registration failed — check the base URL and the key.")})
        return await _connect_with_token(kind, match.group(1), session)

    @app.delete("/api/session/connect/{kind}")
    def disconnect(kind: str, session: auth.Session = Depends(get_session)):
        if kind not in state.sources:
            raise HTTPException(404, detail={"error": f"Unknown source: {kind}"})
        setattr(session, f"{kind}_token", None)
        setattr(session, f"{kind}_tools", 0)
        _rebuild_remote(state, session)
        return {"kind": kind, "active": False, "tools": 0, "error": None}

    @app.post("/api/chat/reset")
    def reset(session: auth.Session = Depends(get_session)):
        # A running turn must never block this: it is cancelled and detached
        # from the conversation it was writing to (see _run_turn_worker).
        stopped = session.turn.abandon()
        state.sessions.reset(session)
        return {"ok": True, "session": session.id, "stopped": stopped}

    @app.post("/api/chat/stop")
    def stop(session: auth.Session = Depends(get_session)):
        return {"ok": True, "stopped": session.turn.stop()}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request,
                   session: auth.Session = Depends(require_session)):
        prompt = req.prompt.strip()
        if not prompt:
            return JSONResponse({"error": "Empty prompt."}, status_code=400)
        resolved = await run_in_threadpool(
            llm.resolve_llm, state.cfg, session.provider_name, session.model_name,
            dict(session.llm_params))
        session.provider_name = resolved["provider"]
        if not resolved["api_key"]:
            prov = (state.cfg.providers or {}).get(resolved["provider"])
            env = prov.api_key_env if prov else "API_KEY"
            return JSONResponse(
                {"error": f"No API key for provider '{resolved['provider']}' — set {env} in the "
                          f"environment or in .env at the repo root and restart the app."},
                status_code=400)
        # A turn already in flight is cancelled and displaced rather than
        # refused: cancellation is cooperative, so a stuck turn would otherwise
        # lock the user out of their own session until it happened to notice.
        epoch, cancel = await run_in_threadpool(session.turn.start)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        with session.lock:
            convo = session.messages
            user_msg = {"role": "user", "content": prompt}
            convo.append(user_msg)
            turn = session.turns + 1
            api_msgs = [{"role": m["role"], "content": m["content"]} for m in convo]

        def put(ev: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ev)
            except RuntimeError:  # loop closed: the client is long gone
                pass

        state.executor.submit(_run_turn_worker, state, session, resolved, prompt, turn,
                              api_msgs, put, cancel, epoch, convo, user_msg)

        async def stream():
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), SSE_PING_S)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
                        yield ": ping\n\n"
                        continue
                    if ev is None:
                        return
                    yield _sse_frame(ev)
            finally:
                cancel.set()  # client gone or stream finished; the worker sees it

        return StreamingResponse(stream(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest, session: auth.Session = Depends(get_session)):
        if req.category not in FEEDBACK_CATEGORIES:
            raise HTTPException(400, detail={"error": f"category must be one of {list(FEEDBACK_CATEGORIES)}"})
        text = req.text.strip()
        if not text:
            raise HTTPException(400, detail={"error": "Add a note before submitting."})
        with session.lock:
            msgs = list(session.messages)
        last = msgs[-1] if msgs else {}
        if last.get("role") == "assistant":
            answer = last.get("content", "")
            question = msgs[-2]["content"] if len(msgs) > 1 else ""
            model = (last.get("meta") or {}).get("model", "")
        else:
            answer, question, model = "", "", _selection(state, session)["model"]
        transcript = [{"role": m["role"], "content": m["content"]} for m in msgs]
        telemetry.record_feedback(
            question=question, answer=answer, model=model,
            provider=_selection(state, session)["provider"], session=session.id,
            category=req.category, text=text, messages=transcript,
        )
        return {"ok": True}

    @app.get("/api/stats")
    def stats():
        with state.stats_lock:
            now = time.monotonic()
            if state.stats_cache["value"] is None or now - state.stats_cache["at"] > STATS_TTL_S:
                state.stats_cache["value"] = _build_stats(state)
                state.stats_cache["at"] = now
            return state.stats_cache["value"]

    @app.get("/api/corpus-map")
    async def corpus_map(clusters: int = Query(8, ge=1, le=100)):
        if not state.corpus_map_available() or state.corpus_map is None:
            return {"available": False,
                    "hint": "Embeddings cache not built. Run: python build.py embeddings"}
        payload = await run_in_threadpool(state.corpus_map, clusters)
        return {"available": True, **payload}

    @app.get("/api/collaboration-graph")
    def collaboration_graph():
        payload = state.collab_graph()
        if payload is None:
            return JSONResponse({"error": "Collaboration graph not built.",
                                 "hint": "Run: python build.py graph"}, status_code=404)
        return payload

    app.mount("/static", RevalidatingStatic(directory=str(STATIC_DIR)), name="static")

    if root_path:
        outer = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        outer.mount(root_path, app)
        return outer
    return app


def _build_stats(state: AppState) -> dict:
    cov = state.coverage() or {}
    summary = state.summarize() or _empty_summary()
    ver = telemetry.version_info()
    pipeline = []
    for key, label, kind in _PIPELINE_STAGES:
        entry = cov.get(key, {}) or {}
        pipeline.append({
            "stage": label, "key": key, "available": bool(entry.get("available")),
            "entries": entry.get("count") if kind == "count" else None,
            "built": (entry.get("last_built") or "")[:19].replace("T", " ") or None,
        })
    tools = [{"name": name, **info} for name, info in list((summary.get("tools") or {}).items())[:12]]
    models = [{"name": name, "turns": n} for name, n in (summary.get("models") or {}).items()]
    return {
        "build": {"git_sha": ver["git_sha"], "build_time": ver["build_time"]},
        "providers": list(state.cfg.providers or {}),
        "default_model": state.cfg.llm.default_model,
        "pipeline": pipeline,
        "usage": {k: summary.get(k, 0) for k in ("turns", "sessions", "error_turns", "avg_latency_ms", "feedback")},
        "tools": tools,
        "models": models,
        "tools_local": len(state.local_tools),
    }
