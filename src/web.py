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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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

# Clickable starters shown while the conversation is empty.
EXAMPLES = [
    "Which papers cover perovskite stability?",
    "Who works on electrocatalysis?",
    "Which PIs bridge institutions?",
    "NOMAD data for battery materials",
]

SOURCES = {
    "elab": {"label": "eLabFTW", "register_url": "https://researchmcp.duckdns.org/el/register"},
    "dt": {"label": "DataTagger", "register_url": "https://researchmcp.duckdns.org/dt/register"},
}

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
    examples: list[str] = field(default_factory=lambda: list(EXAMPLES))
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


class TokenRequest(BaseModel):
    token: str = ""     # "" disconnects


class FeedbackRequest(BaseModel):
    category: str
    text: str = Field(min_length=1, max_length=MAX_FEEDBACK_CHARS)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _selection(state: AppState, session: auth.Session) -> dict:
    """Provider/model as chosen, without any network (no OpenRouter auto-pick)."""
    providers = state.cfg.providers or {}
    name = session.provider_name if session.provider_name in providers else llm.default_provider_name(state.cfg)
    prov = providers.get(name)
    if prov is None:
        return {"provider": name, "model": state.cfg.llm.default_model, "auto": False}
    models = list(prov.models) or [prov.default_model]
    openrouter = "openrouter" in prov.base_url
    if session.model_name in models:
        return {"provider": name, "model": session.model_name, "auto": False}
    if openrouter:
        return {"provider": name, "model": "", "auto": True}
    return {"provider": name, "model": prov.default_model, "auto": False}


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
        "connected": {
            kind: {"active": bool(getattr(session, f"{kind}_token")),
                   "tools": getattr(session, f"{kind}_tools") if getattr(session, f"{kind}_token") else 0}
            for kind in state.sources
        },
        "tools": _tools_available(state, session),
        "turns": turns,
        "busy": session.turn_lock.locked(),
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
                     cancel: threading.Event) -> None:
    """Own one turn end to end: stream events, then commit history + telemetry."""
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
            session.messages.append({"role": "assistant", "content": final["answer"], "meta": meta})
            session.turns = turn
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
        session.cancel = None
        session.turn_lock.release()
        put(None)


def _sse_frame(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


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
                   "base_url": p.base_url, "openrouter": "openrouter" in p.base_url}
            for name, p in (state.cfg.providers or {}).items()
        }
        return {
            "title": APP_TITLE,
            "cluster": {"name": state.cfg.cluster.name, "display_name": state.cfg.cluster.display_name,
                        "website": state.cfg.cluster.website},
            "placeholder": f"Ask about {state.n_papers} papers across {state.n_pis} PIs/groups in the cluster…",
            "examples": state.examples,
            "providers": providers,
            "default_provider": llm.default_provider_name(state.cfg),
            "sources": state.sources,
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
        return _selection(state, session)

    @app.post("/api/session/connect/{kind}")
    async def connect(kind: str, req: TokenRequest, session: auth.Session = Depends(get_session)):
        if kind not in state.sources:
            raise HTTPException(404, detail={"error": f"Unknown source: {kind}"})
        token = req.token.strip()
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
        if session.turn_lock.locked():
            raise HTTPException(409, detail={"error": "A response is still being generated; stop it first."})
        state.sessions.reset(session)
        return {"ok": True, "session": session.id}

    @app.post("/api/chat/stop")
    def stop(session: auth.Session = Depends(get_session)):
        cancel = session.cancel
        if cancel is not None:
            cancel.set()
        return {"ok": True, "stopped": cancel is not None}

    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request,
                   session: auth.Session = Depends(require_session)):
        prompt = req.prompt.strip()
        if not prompt:
            return JSONResponse({"error": "Empty prompt."}, status_code=400)
        resolved = await run_in_threadpool(
            llm.resolve_llm, state.cfg, session.provider_name, session.model_name)
        session.provider_name = resolved["provider"]
        if not resolved["api_key"]:
            prov = (state.cfg.providers or {}).get(resolved["provider"])
            env = prov.api_key_env if prov else "API_KEY"
            return JSONResponse(
                {"error": f"No API key for provider '{resolved['provider']}' — set {env} in the "
                          f"environment or in .env at the repo root and restart the app."},
                status_code=400)
        if not session.turn_lock.acquire(blocking=False):
            return JSONResponse({"error": "A response is already being generated for this session."},
                                status_code=409)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()
        session.cancel = cancel
        with session.lock:
            session.messages.append({"role": "user", "content": prompt})
            turn = session.turns + 1
            api_msgs = [{"role": m["role"], "content": m["content"]} for m in session.messages]

        def put(ev: Any) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ev)
            except RuntimeError:  # loop closed: the client is long gone
                pass

        state.executor.submit(_run_turn_worker, state, session, resolved, prompt, turn,
                              api_msgs, put, cancel)

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
        telemetry.record_feedback(
            question=question, answer=answer, model=model,
            provider=_selection(state, session)["provider"], session=session.id,
            category=req.category, text=text,
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

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
