# Porting map: this demo → `cluster-research-assist` (CRA)

This repository holds the interim e-verse frontend (FastAPI + SSE + vanilla JS,
September 2026). `AGENTS.md` / the redesign handover ask for the MVP to be built in
the new `cluster-research-assist` repo on Quart with an async single event loop, an
own tool registry, a `Corpus` object and a pydantic `config.yaml`. This file says,
per file, what moves as is, what changes shape, and what was learned that the port
must not lose. Module names follow the design doc §3.2 (no package prefixes,
unique basenames).

## Carries over as is (copy, adjust imports)

| here | CRA | notes |
|---|---|---|
| `src/static/index.html`, `css/app.css`, `js/*.js`, `js/views/*.js`, `logo.svg` | `src/cra/web/static/` | Framework-free. All URLs are relative, so it runs under any prefix. Theme toggle, SSE client, markdown render, live tool steps, stop button, session reload. |
| `src/static/collaboration_map.html`, `pipeline_map.html` | `src/cra/web/static/` | Self-contained; fetch `../api/collaboration-graph`. Pipeline lives inside the "Stats for nerds" dialog. |
| `src/agent.py` event protocol (`start`, `round`, `text_delta`, `reasoning_delta`, `tool_call_start/_end`, `done`, `error`) | `chat/orchestrator.py` | Keep the payloads; the frontend depends on them. |
| `src/llm.py` OpenRouter selection (`fetch_openrouter_models`, price/agentic/tools filters) | `llm/selection.py` | Pure; add a TTL cache as here. |
| `src/prompts.py` | `chat/prompt.py` | Parametrise by config instead of the module constant. |
| `src/views.py` (corpus-map payload with palette + legend, collaboration payload) | `viz/…` | Unchanged logic. |
| `src/telemetry.py` (privacy rule, `turn_record`, feedback JSONL) | `telemetry.py` (history replaces feedback JSONL in M4) | Keep the "hashes and lengths only" invariant and its tests. |
| `tests/fakes.py`, `tests/test_agent.py` | `tests/` | Chunk builders mirror the SDK shapes; make the fake stream an async iterator. |
| `tests/test_web.py`, `tests/test_web_live.py` | `tests/` | Same cases (frame order, 401/409, cancel on disconnect, two-session isolation, connect success/failure) on Quart's test client / a Hypercorn thread. |
| `Dockerfile`, `docker/entrypoint.sh` | deployment repo | Swap uvicorn for Hypercorn; keep `PORT=8501`, `ROOT_PATH`, the health check. |

## Changes shape

| here | CRA | what changes |
|---|---|---|
| `src/web.py` (FastAPI app factory, worker-thread SSE bridge) | `web/factory.py`, `web/routes/chat.py` | Quart `async` handlers; the turn runs on the loop, no thread, no queue. Keep the endpoint list and JSON shapes (`/api/session`, `/api/config`, `/api/chat`, `/api/chat/stop`, `/api/chat/reset`, `/api/session/model`, `/api/session/connect/{kind}`, `/api/feedback`, `/api/stats`, `/api/corpus-map`, `/api/collaboration-graph`, `/api/health`). |
| `src/agent.py` `run_turn` (sync generator) | `chat/orchestrator.py` (`async` generator on `AsyncOpenAI`) | Same logic: per-index tool-call accumulation, `<think>` splitter, cancel between chunks and tools, all rounds' text joined into the stored answer, `error` event that keeps the rounds/tools already run. CPU-bound tools (BM25, sentence-transformers) go through `asyncio.to_thread`. |
| `src/auth.py` (cookie sessions, `get_identity` header hook) | `auth/…` + a per-session store | Same fields. Key the store by the NOMAD SSO header when present (§4.1), cookie otherwise. The nginx in the NOMAD stack forwards **no** user header today; add `auth_request_set` + `proxy_set_header X-Forwarded-User` there. |
| `src/openai_tools.py` (schemas from `server.mcp._tool_manager`) | `tools/registry.py` with `@tool(name, tier, description)` | Derive the schema from signature + type hints once; drop the private FastMCP access and the `mcp==1.28.1` pin. Every tool declares `Tier.PUBLIC` / `Tier.INTERNAL` (`get_paper_fulltext`, `get_proposal_fulltext` internal). |
| `src/server.py` (loads caches at import, torch preload thread) | `corpus/loader.py` `Corpus.load()` | Explicit load at startup from the configured directory; no import-time side effects, no `__file__`-relative paths. |
| `src/mcp_clients.py` (`anyio.run()` per call) | `mcpclient/host.py` | `await` the MCP client directly on the app loop; per-session tokens, one handshake per session (pooling is out of MVP scope). |
| `config.toml` + `src/config.py` dataclasses | `config.yaml` + pydantic models | Same keys: cluster identity, `[llm]` providers/models, `api_key_env`. |

## Learned the hard way (keep these)

- **GWDG `stream_options={"include_usage": true}` returns HTTP 500**, not a 400. Treat any status error on the first attempt as "unsupported", retry plainly, remember per base URL. GWDG sends a usage chunk anyway, so nothing is lost.
- **GWDG answers a large share of requests with an instant 500** (5 of 8 trivial requests, all models, 2026-09-21) and occasionally hangs. Use SDK retries with backoff (5) and a per-read timeout (120 s). Show a readable message, not `Error code: 500`.
- Reasoning arrives as `delta.reasoning` on GWDG (vLLM); some models put `<think>` inline in `content`. Handle both; never store or resend reasoning.
- Tool-call fragments: `index` present on GWDG; ids look like `chatcmpl-tool-…`. Fragments without `index` need the "id or name opens a new call" heuristic.
- The deployment proxies `/nomad-oasis/api/everse/` **without stripping the prefix**: mount the app under `ROOT_PATH`, scope the cookie to it, keep all frontend URLs relative. nginx buffers by default: send `X-Accel-Buffering: no` on the SSE response; Caddy already has `flush_interval -1`.
- Client disconnect must cancel the turn (a `threading.Event` here; a `asyncio.Event`/task cancel there) and the turn must still be logged once.
- Frontend: after a stop, the fetch rejects before the server commits the assistant message; wait for `busy == false` before refreshing history, or the live DOM gets replaced. Store subscribers should receive the changed keys so unrelated updates do not re-render the chat.
- The compose `.env` names the GWDG key `ECONVERSION_API_KEY` (mapped to `API_KEY` in the container); accept both.

## Out of scope here, per AGENTS.md §5

History/database, the outward `/mcp` surface, tier enforcement, OIDC, MCP connection pooling, ingestion changes, retrieval changes.
