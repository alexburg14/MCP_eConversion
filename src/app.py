"""
eConversion Knowledge Assistant — Streamlit chat interface.

Run with:
    streamlit run src/app.py

Backed by the GWDG SAIA / Academic Cloud Chat AI endpoint (OpenAI-compatible).
Requires API_KEY in the environment or in a .env file at the repo root.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import server  # loads all caches at import time
import corpus_map
from config import get_config
from logging_config import configure_logging, get_logger
import openai_tools
import mcp_clients
import telemetry

configure_logging()  # idempotent; server import already did this
log = get_logger("app")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CFG = get_config()

BASE_URL = _CFG.llm.base_url

# OpenRouter: dynamically fetch models that satisfy the account's guardrails
# and cost under 1 EUR / M tokens (prompt AND completion). Falls back to the
# static config list if the fetch fails or no key is configured.
_OPENROUTER_MAX_PRICE_PER_MTOK = 1.0  # EUR-equivalent cap (approx. USD 1.0)
_PRICE_PER_TOKEN_CAP = _OPENROUTER_MAX_PRICE_PER_MTOK / 1_000_000


def _fetch_openrouter_models(api_key: str) -> tuple[list[str], list[str]]:
    """Return (cheapest_model_ids, all_model_ids) allowed by the user's guardrails,
    under the price cap, and with an agentic index >= 35.

    The guardrail-filtered list comes from /models/user; benchmark data (agentic
    index) only exists in the full /models list, so we intersect both.
    """
    import requests

    def _price_ok(m) -> bool:
        pricing = m.get("pricing", {}) or {}
        try:
            prompt = float(pricing.get("prompt") or 0)
            completion = float(pricing.get("completion") or 0)
        except (TypeError, ValueError):
            return False
        if prompt < 0 or completion < 0:
            return False
        return prompt <= _PRICE_PER_TOKEN_CAP and completion <= _PRICE_PER_TOKEN_CAP

    def _agentic_ok(m) -> bool:
        b = m.get("benchmarks", {}) or {}
        aa = b.get("artificial_analysis", {}) or {}
        try:
            return float(aa.get("agentic_index") or 0) >= 35.0
        except (TypeError, ValueError):
            return False

    def _tool_calling_ok(m) -> bool:
        # the assistant depends on tool calling for paper search; require the
        # model to advertise tool support
        return "tools" in (m.get("supported_parameters") or [])

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r_user = requests.get("https://openrouter.ai/api/v1/models/user", headers=headers, timeout=15)
        r_user.raise_for_status()
        user_ids = {m.get("id", "") for m in r_user.json().get("data", [])}
    except Exception:
        return [], []

    try:
        r_all = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        r_all.raise_for_status()
        all_models = r_all.json().get("data", [])
    except Exception:
        return [], []

    eligible: list[tuple[str, float]] = []  # (model_id, prompt_price_per_token)
    all_ids: list[str] = []
    for m in all_models:
        mid = m.get("id", "")
        if mid not in user_ids:
            continue
        # skip :free endpoints — weak data guarantees (ZDR is enforced per
        # request via provider.zdr, but free hosts are often unstable)
        if mid.endswith(":free") or ":free" in mid:
            continue
        if not _price_ok(m) or not _agentic_ok(m) or not _tool_calling_ok(m):
            continue
        all_ids.append(mid)
        try:
            prompt = float((m.get("pricing", {}) or {}).get("prompt") or 0)
        except (TypeError, ValueError):
            prompt = float("inf")
        eligible.append((mid, prompt))

    # sort by prompt price, cheapest first
    eligible.sort(key=lambda x: x[1])
    cheapest = [mid for mid, _ in eligible]
    return cheapest, all_ids


def _load_dotenv() -> None:
    """Set vars from the repo-root .env if not already in the environment."""
    env_file = _REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Tools: schemas and dispatch are derived from the MCP registry (server.py),
# the single source of truth. See openai_tools.py.
# ---------------------------------------------------------------------------

_TOOLS = openai_tools.build_openai_tools()

# One startup line per process: which build, which cache state, which tools.
telemetry.log_startup(
    tools_local=len(_TOOLS),
    tools_remote=0,
    providers=list((_CFG.providers or {})),
    models=list(_CFG.llm.models),
)


def _cluster_identity_line() -> str:
    """One-line cluster identity (funder, ID, institutions) if configured, else empty."""
    parts = []
    if _CFG.cluster.cluster_id:
        parts.append(f"Cluster ID {_CFG.cluster.cluster_id}")
    if _CFG.cluster.funding_body:
        parts.append(f"funded by {_CFG.cluster.funding_body}")
    if _CFG.cluster.host_institutions:
        parts.append("hosted at " + " and ".join(_CFG.cluster.host_institutions))
    if not parts:
        return ""
    return "\n\n" + "; ".join(parts) + "."


_BASE_SYSTEM = f"""\
You are a research assistant for {_CFG.cluster.description}.\
{_cluster_identity_line()}

You have access to a local database of {len(server.papers)} cluster publications (with \
abstracts and some full texts) and profiles of {len(server._PIS)} PIs. Use the tools to \
retrieve relevant information before answering. Always cite papers by title and DOI. If \
information is missing from the database, say so clearly — do not invent facts.

Two paper-search tools complement each other:
- search_papers (BM25, lexical): exact terms, acronyms, formulas, author names.
- semantic_search_papers (embeddings): conceptual queries where vocabulary may
  differ from abstracts. Run both when you're unsure which will hit — the
  union of results gives wider recall before you synthesize.

Use get_similar_papers(doi) for "what else is like this paper?" — it compares a
specific paper's own embedding to every other paper's, rather than taking a text query.

Use list_papers (exact metadata filter: author, year, journal) for exhaustive \
listings — "every paper by X", "papers in Nature", "what the cluster published in \
2022" — where the top-5 relevance results of the search tools are not enough.

Four collaboration-graph tools answer network questions that search cannot:
- get_collaborators(pi_query): who publishes with a given PI?
- joint_papers(pi_a, pi_b): which papers did two specific PIs co-author?
- collaboration_centrality(): which PIs bridge otherwise-separate groups?
- collaboration_communities(): which clusters of PIs work closely together?

When a question needs more than the abstract — specific methods, results, experimental \
details, or exact numbers — call get_paper_fulltext(doi) on the most relevant paper(s) \
from your search results before answering. Full text is cached for ~99% of the corpus; \
if it comes back missing, say so and answer from the abstract.

search_nomad queries the public NOMAD materials repository — data EXTERNAL to the \
cluster, not e-conversion papers. Use it when asked whether computed or measured data \
exists for a material, or what a PI has deposited. Report total_matches first; the \
entries it returns are a sample of a much larger set. NOMAD entries carry no DOI link \
back to cluster publications, so never present them as "the data behind" a paper.

For corpus-wide questions ("main open challenges", "trends over time", \
"complementary groups"), issue several complementary queries before answering: \
one tool call returns at most 5 papers, but a synthesis question needs evidence \
from many. Iterate with different phrasings, then summarize.

Answer in the same language as the question (German or English).\
"""

_PROPOSAL_SUMMARY_PATH = _REPO_ROOT / "data" / "cache" / "proposal_summary.md"
_FEEDBACK_PATH = _REPO_ROOT / "data" / "feedback" / "feedback.jsonl"


def _build_system_prompt() -> str:
    """Compose the system prompt, appending the e-conversion 2.0 proposal summary if present."""
    if _PROPOSAL_SUMMARY_PATH.exists():
        summary = _PROPOSAL_SUMMARY_PATH.read_text(encoding="utf-8")
        return (
            _BASE_SYSTEM
            + "\n\nThe following is Section 2 of the e-conversion 2.0 DFG proposal "
            + "(\"Summary of the Proposal\"), describing the cluster's scope, motivation, "
            + "and research approach. Use it as background context.\n\n"
            + "<proposal_summary>\n"
            + summary
            + "\n</proposal_summary>"
        )
    return _BASE_SYSTEM


_SYSTEM = _build_system_prompt()

_MAX_TOOL_ROUNDS = 10


def _remote_system_note() -> str:
    """One or two lines telling the model which live data sources are attached.

    Deliberately minimal: only appended when the user has connected a remote
    source, so the base system prompt stays untouched for everyone else.
    """
    rc = openai_tools.get_remote_clients()
    if rc is None:
        return ""
    parts = []
    if rc.elab_token:
        parts.append(
            "eLabFTW (ELN): tools prefixed elab_ read the lab notebook of the "
            "connected account - use them for questions about experiments, "
            "items, entries or inventory."
        )
    if rc.dt_token:
        parts.append(
            "DataTagger: tools prefixed dt_ search the research data repository "
            "- use them for questions about datasets, versions or deposited "
            "research data."
        )
    if not parts:
        return ""
    return (
        "\n\nLive data sources are attached in this session; use their tools "
        "when the question concerns them:\n- " + "\n- ".join(parts)
    )


def _answer(client: OpenAI, model: str, messages: list[dict], extra: dict | None = None) -> tuple[str, list[str], float, dict]:
    """Run the tool-use loop.

    Returns (answer_text, tool_calls_summary, elapsed_seconds, meta); meta carries
    rounds, per-tool-call metadata, token usage and the error flag the per-turn
    telemetry line needs.
    """
    tool_log: list[str] = []
    tool_meta: list[dict] = []
    usage_totals = {"prompt": 0, "completion": 0, "total": 0}
    rounds = 0
    error: str | None = None
    sys_content = _SYSTEM + _remote_system_note()
    msgs = [{"role": "system", "content": sys_content}] + list(messages)
    start = time.perf_counter()
    # Tools are rebuilt per call: local + (session) remote tools. A user who
    # connects eLabFTW/DataTagger gets those tools; without a token this is
    # exactly the local-only list from before.
    kwargs = dict(tools=openai_tools.build_chat_tools())
    extra_body: dict | None = None
    if extra:
        # OpenRouter-specific fields (provider routing, zdr, ...) must go in
        # extra_body; the OpenAI SDK rejects unknown top-level kwargs.
        if "provider" in extra:
            extra_body = {"provider": extra["provider"]}
        else:
            kwargs.update(extra)

    for round_num in range(_MAX_TOOL_ROUNDS):
        rounds = round_num + 1
        call_start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            # Headroom for long list answers (e.g. "all papers by X" can be
            # 30-40 items); 2048 truncated those mid-list.
            max_tokens=8192,
            messages=msgs,
            extra_body=extra_body,
            **kwargs,
        )
        log.info("llm_call", extra={"fields": {
            "model": model, "round": round_num,
            "duration_s": round(time.perf_counter() - call_start, 3),
        }})
        msg = response.choices[0].message
        _usage = getattr(response, "usage", None)
        if _usage is not None:
            usage_totals["prompt"] += int(getattr(_usage, "prompt_tokens", 0) or 0)
            usage_totals["completion"] += int(getattr(_usage, "completion_tokens", 0) or 0)
            usage_totals["total"] += int(getattr(_usage, "total_tokens", 0) or 0)

        if not msg.tool_calls:
            elapsed = time.perf_counter() - start
            log.info("answer_complete", extra={"fields": {
                "model": model, "rounds": round_num + 1, "duration_s": round(elapsed, 3),
            }})
            meta = {"rounds": rounds, "tools": tool_meta, "usage": usage_totals, "error": error}
            return msg.content or "", tool_log, elapsed, meta

        msgs.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            _call_t0 = time.perf_counter()
            result = openai_tools.call_tool(tc.function.name, args)
            _call_ok = not (isinstance(result, str) and result.lstrip().startswith('{"error"'))
            tool_meta.append(telemetry.tool_call_meta(
                tc.function.name, args, (time.perf_counter() - _call_t0) * 1000, _call_ok))
            tool_log.append(f"`{tc.function.name}({(tc.function.arguments or '')[:80]})`")
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    elapsed = time.perf_counter() - start
    meta = {
        "rounds": rounds,
        "tools": tool_meta,
        "usage": usage_totals,
        "error": "tool_call_limit_reached",
    }
    return (
        "Tool-call limit reached without a final answer — try rephrasing the question.",
        tool_log,
        elapsed,
        meta,
    )


def _record_feedback(question: str, answer: str, model: str, category: str, text: str) -> None:
    """Append one feedback record (bug report or general note on a question/answer pair) as JSONL.

    Provenance fields (session/model/build) make a report traceable to the turn it
    came from; the log line itself stays metadata-only.
    """
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ver = telemetry.version_info()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": st.session_state.get("provider_label", ""),
        "session": st.session_state.get("session_id", ""),
        "git_sha": _ver["git_sha"],
        "build_time": _ver["build_time"],
        "question": question,
        "answer": answer,
        "category": category,
        "text": text,
    }
    with open(_FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    telemetry.log_feedback(
        category=category,
        session=st.session_state.get("session_id", ""),
        model=model,
        text_len=len(text),
        question_len=len(question or ""),
        answer_len=len(answer or ""),
    )

def _tool_availability() -> dict:
    """How many tools the model is offered in this session (local + remote)."""
    try:
        tools = openai_tools.build_chat_tools()
    except Exception:  # noqa: BLE001 -- stats must never break the chat
        return {"local": 0, "elab": 0, "dt": 0, "total": 0}
    names = [t["function"]["name"] for t in tools]
    return {
        "local": sum(1 for x in names if not x.startswith(("elab_", "dt_"))),
        "elab": sum(1 for x in names if x.startswith("elab_")),
        "dt": sum(1 for x in names if x.startswith("dt_")),
        "total": len(names),
    }


@st.cache_data(ttl=60, show_spinner=False)
def _coverage_cached() -> dict:
    """Data coverage (cache availability/count/mtime) -- rarely changes."""
    return telemetry.coverage()


@st.cache_data(ttl=30, show_spinner=False)
def _stats_cached() -> dict:
    """All-time usage aggregated from the log files (current + rotations)."""
    return telemetry.summarize()


def _coverage_table(snapshot: dict) -> pd.DataFrame:
    rows = []
    for name, entry in snapshot.items():
        rows.append({
            "cache": name,
            "available": "yes" if entry.get("available") else "no",
            "entries": "\u2014" if entry.get("count") is None else str(entry["count"]),
            "last built": (entry.get("last_built") or "\u2014")[:19].replace("T", " "),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Computing corpus map (UMAP + clustering)...")
def _build_corpus_map(n_clusters: int) -> list[dict]:
    return corpus_map.build_map(server.papers_by_doi, n_clusters=n_clusters)


_COLLAB_GRAPH_PATH = _REPO_ROOT / "data" / "cache" / "collaboration_graph.json"


@st.cache_data(show_spinner=False)
def _build_collab_graph() -> dict | None:
    """PI co-authorship graph shaped for the d3 force map: nodes with degree +
    a short surname label, and undirected weighted links. None if not built."""
    if not _COLLAB_GRAPH_PATH.exists():
        return None
    g = json.loads(_COLLAB_GRAPH_PATH.read_text(encoding="utf-8"))
    deg: dict[str, int] = {}
    for link in g["links"]:
        deg[link["source"]] = deg.get(link["source"], 0) + 1
        deg[link["target"]] = deg.get(link["target"], 0) + 1
    nodes = [
        {
            "id": n["id"], "name": n["name"], "label": (n["name"].split() or [n["name"]])[-1],
            "group": n.get("group", ""), "inst": n.get("institution", ""),
            "papers": n.get("paper_count", 0), "deg": deg.get(n["id"], 0),
        }
        for n in g["nodes"]
    ]
    links = [{"source": l["source"], "target": l["target"], "weight": l["weight"]} for l in g["links"]]
    return {"nodes": nodes, "links": links}


# Corpus-map cluster colors: Tableau-20, saturated hues first so the common
# small-cluster-count case gets the most distinguishable set. RGB triples.
_CLUSTER_PALETTE = [
    tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    for h in (
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
        "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#79706E",
        "#9ECAE9", "#FFBF79", "#88D27A", "#FF9D98", "#83BCB6",
        "#F2CF5B", "#D6A5C9", "#D67195", "#D8B5A5", "#BAB0AC",
    )
]

# deck.gl scatter for the corpus map, rendered in a components.html iframe.
# Streamlit's bundled pydeck can't drive an OrthographicView (its JSON path is
# built for geospatial views and asserts), so we load deck.gl directly and get
# GPU-crisp points with smooth scroll-zoom / drag-pan. __DATA__ / __MATCHES__
# are filled per render. Uses OrthographicView so the UMAP plane maps 1:1 to
# screen. __MATCHES__ is the list of titles matching the search box.
_CORPUS_MAP_TEMPLATE = """
<style>
  html, body { margin: 0; height: 100%; background: transparent; }
  #wrap { position: relative; width: 100%; height: 100%; }
  #deck-canvas { width: 100%; height: 100%; }
  .dk-tip {
    background: #1b1e26; color: #e9ebf0;
    font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 6px 9px; border-radius: 6px; max-width: 340px;
    box-shadow: 0 4px 16px rgba(0,0,0,.45);
  }
</style>
<div id="wrap"><canvas id="deck-canvas"></canvas></div>
<script src="https://cdn.jsdelivr.net/npm/deck.gl@9.0.38/dist.min.js"></script>
<script>
  const DATA = __DATA__;
  const MATCHES = new Set(__MATCHES__);
  const { Deck, OrthographicView, ScatterplotLayer } = deck;
  const wrap = document.getElementById("wrap");
  const xs = DATA.map(d => d.x), ys = DATA.map(d => d.y);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const W = wrap.clientWidth || 700, H = wrap.clientHeight || 540;
  const zoom = Math.log2(0.85 * Math.min(
    W / Math.max(maxx - minx, 1e-6), H / Math.max(maxy - miny, 1e-6)));

  // A search rings every matching paper and frames the hits: a single hit gets
  // its ~30-neighbor surroundings; several get a bounding-box fit. An empty
  // search (or no matches) shows the whole corpus. All clamps keep us from
  // zooming out past the full view or in absurdly far.
  let viewTarget = [(minx + maxx) / 2, (miny + maxy) / 2, 0];
  let viewZoom = zoom;
  const hits = DATA.filter(d => MATCHES.has(d.title));
  if (hits.length === 1) {
    const sel = hits[0];
    const dists = DATA
      .map(d => Math.hypot(d.x - sel.x, d.y - sel.y))
      .sort((a, b) => a - b);
    const R = dists[Math.min(30, dists.length - 1)] || 1;
    const zin = Math.log2(0.35 * Math.min(W, H) / Math.max(R, 1e-6));
    viewTarget = [sel.x, sel.y, 0];
    viewZoom = Math.min(Math.max(zin, zoom + 1), zoom + 5);
  } else if (hits.length > 1) {
    const hx = hits.map(d => d.x), hy = hits.map(d => d.y);
    const nx = Math.min(...hx), Xx = Math.max(...hx);
    const ny = Math.min(...hy), Xy = Math.max(...hy);
    viewTarget = [(nx + Xx) / 2, (ny + Xy) / 2, 0];
    const zfit = Math.log2(0.8 * Math.min(
      W / Math.max(Xx - nx, 1e-6), H / Math.max(Xy - ny, 1e-6)));
    viewZoom = Math.min(Math.max(zfit, zoom), zoom + 6);
  }

  function layers() {
    const L = [new ScatterplotLayer({
      id: "points", data: DATA,
      getPosition: d => [d.x, d.y], getFillColor: d => d.color,
      getRadius: 4, radiusUnits: "pixels", radiusMinPixels: 2.5, radiusMaxPixels: 9,
      opacity: 0.85, pickable: true,
      autoHighlight: true, highlightColor: [255, 255, 255, 140],
    })];
    if (MATCHES.size) {
      L.push(new ScatterplotLayer({
        id: "matched", data: hits,
        getPosition: d => [d.x, d.y],
        filled: false, stroked: true,
        getLineColor: [255, 255, 255],
        lineWidthUnits: "pixels", getLineWidth: 2,
        lineWidthMinPixels: 2, lineWidthMaxPixels: 2,
        getRadius: 12, radiusUnits: "pixels", radiusMinPixels: 12, radiusMaxPixels: 12,
        pickable: false,
      }));
    }
    return L;
  }

  new Deck({
    canvas: "deck-canvas",
    views: new OrthographicView({}),
    controller: { scrollZoom: true, dragPan: true, doubleClickZoom: true },
    initialViewState: { target: viewTarget, zoom: viewZoom },
    layers: layers(),
    getTooltip: ({ object }) => object && {
      html: "<b>" + object.title + "</b><br/>" + object.year + " \\u00b7 " + object.cluster,
      className: "dk-tip",
    },
  });
</script>
"""


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


APP_TITLE = "e-Converse"
_PAGE_ICON = "⚡"

# Clickable starters shown while the conversation is empty.
_EXAMPLES = [
    "Which papers cover perovskite stability?",
    "Who works on electrocatalysis?",
    "Which PIs bridge institutions?",
    "NOMAD data for battery materials",
]


def _stats_for_nerds() -> str:
    """Everything technical, behind the ⋮ menu -> About (nothing on the chat page)."""
    cov = telemetry.coverage()
    stats = telemetry.summarize()
    ver = telemetry.version_info()

    def n(name: str) -> object:
        return cov.get(name, {}).get("count", 0)

    def built(name: str) -> str:
        return str(cov.get(name, {}).get("last_built") or "—")[:19].replace("T", " ")

    def ok(name: str) -> str:
        return "yes" if cov.get(name, {}).get("available") else "no"

    build = ver["git_sha"] + (f" ({ver['build_time']})" if ver.get("build_time") else "")
    tool_rows = "".join(
        f"\n| {name} | {info['calls']} | {info['errors']} | {info['avg_ms']} ms |"
        for name, info in list(stats["tools"].items())[:12]
    ) or "\n| _none yet_ | | | |"
    model_rows = "".join(
        f"\n| {name} | {count} |" for name, count in stats["models"].items()
    ) or "\n| _none yet_ | |"
    return "\n".join([
        f"### Stats for nerds",
        "",
        f"**Build** `{build}` · providers: {', '.join(_CFG.providers or {})} · "
        f"default model `{_CFG.llm.default_model}`",
        "",
        "**Pipeline** (sources → caches → tools)",
        "",
        "| stage | entries | built |",
        "| --- | --- | --- |",
        f"| papers (DOI csv) | {n('papers')} | {built('papers')} |",
        f"| abstracts | {n('abstracts')} | {built('abstracts')} |",
        f"| full texts | {n('fulltext')} | {built('fulltext')} |",
        f"| PI profiles | {n('pis')} | {built('pis')} |",
        f"| embeddings | {ok('embeddings')} | {built('embeddings')} |",
        f"| collaboration graph | {ok('graph')} | {built('graph')} |",
        "",
        "**Usage** (from the server logs)",
        "",
        f"{stats['turns']} turns · {stats['sessions']} sessions · {stats['error_turns']} errors · "
        f"⌀ {stats['avg_latency_ms']} ms · {stats['feedback']} feedback",
        "",
        "| tool | calls | errors | avg |",
        "| --- | --- | --- | --- |" + tool_rows,
        "",
        "| model | turns |",
        "| --- | --- |" + model_rows,
        "",
        f"Full pipeline map: the **Pipeline** page · full text search tools: {_tool_inventory_text()}",
    ])


def _tool_inventory_text() -> str:
    local = len(openai_tools.build_openai_tools())
    bits = [f"{local} local"]
    if st.session_state.get("elab_token"):
        bits.append(f"{st.session_state.get('elab_tools', 0)} eLabFTW")
    if st.session_state.get("dt_token"):
        bits.append(f"{st.session_state.get('dt_tools', 0)} DataTagger")
    return " · ".join(bits)


def _page_setup() -> None:
    """Load .env, then configure the page. Numbers live in ⋮ → About, not in the chat."""
    _load_dotenv()
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=_PAGE_ICON,
        layout="centered",
        menu_items={"About": _stats_for_nerds()},
    )


def _view_corpus_map() -> None:
    """UMAP layout of the paper embeddings (deck.gl, searchable)."""
    st.caption(
        "UMAP layout of the paper embeddings; KMeans clusters (computed in the "
        "full 384-d space) labeled with their top title keywords. Scroll to zoom, "
        "drag to pan, hover a point for its title — a visual answer to 'which "
        "papers are near the one I'm reading?'"
    )
    if not corpus_map.is_available():
        st.info("Embeddings cache not built. Run: `python src/scripts/build_embeddings_cache.py`")
    else:
        n_clusters = st.slider("Clusters", min_value=2, max_value=20, value=8)
        df = pd.DataFrame(_build_corpus_map(n_clusters))

        # One bar that is both a scrollable list and a search: st.selectbox
        # shows the full paper list on click and filters it as you type.
        highlight = st.selectbox(
            "Find a paper",
            options=sorted(df["title"].tolist()),
            index=None,
            placeholder="Search or scroll the paper list…",
        )
        matches = [highlight] if highlight else []

        # Map each keyword-labeled cluster to a palette color. deck.gl's
        # OrthographicView has y pointing down, so negate y for a conventional
        # (y-up) orientation, then hand the points to the deck.gl iframe.
        clusters = sorted(df["cluster"].unique())
        cmap = {c: _CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i, c in enumerate(clusters)}
        records = [
            {
                "x": float(row.x), "y": -float(row.y),
                "title": row.title, "year": str(row.year),
                "cluster": row.cluster, "color": list(cmap[row.cluster]),
            }
            for row in df.itertuples()
        ]
        html = (
            _CORPUS_MAP_TEMPLATE
            .replace("__DATA__", json.dumps(records))
            .replace("__MATCHES__", json.dumps(matches))
        )
        components.html(html, height=560, scrolling=False)

        # deck.gl has no legend of its own — render one from the same cmap.
        swatches = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin:0 16px 6px 0;font-size:12px">'
            f'<span style="width:11px;height:11px;border-radius:3px;'
            f'background:rgb({r},{g},{b});display:inline-block"></span>{c}</span>'
            for c, (r, g, b) in cmap.items()
        )
        st.markdown(
            '<div style="opacity:.55;font-size:11px;margin-bottom:4px">'
            "Cluster (top title keywords)</div>"
            f'<div style="display:flex;flex-wrap:wrap">{swatches}</div>',
            unsafe_allow_html=True,
        )

# Collaboration network: PI co-authorship graph (data/cache/collaboration_graph.json)
# as a d3 force map. Same cache the collaboration_* tools query; no API key needed.


def _view_collaboration() -> None:
    """PI co-authorship network (interactive force graph)."""
    st.caption(
        "PI co-authorship network from the cluster's publications — nodes are PIs "
        "(sized by number of collaborators, colored by institution), edges are shared "
        "papers (thicker = more). Click a PI to isolate who they publish with."
    )
    _collab = _build_collab_graph()
    if _collab is None:
        st.info("Collaboration graph not built. Run: `python src/scripts/build_graph_cache.py`")
    else:
        _collab_html = (
            (Path(__file__).parent / "collaboration_map.html").read_text(encoding="utf-8")
            .replace("__NODES__", json.dumps(_collab["nodes"]))
            .replace("__LINKS__", json.dumps(_collab["links"]))
        )
        components.html(_collab_html, height=640, scrolling=False)

# Pipeline map: a static, self-contained HTML lineage diagram of the whole
# sources → build → caches → tools → registry → delivery pipeline. Rendered in
# an iframe so its own click-to-trace interactivity works independently of
# Streamlit; no API key needed, so render it before the chat tab's st.stop().


def _view_pipeline() -> None:
    """Cluster research pipeline map."""
    st.caption(
        "Every source, script, cache and tool behind the assistant, top to bottom "
        "in the order data moves through them. Click a box to trace what it's built "
        "from — all the way back to the raw sources — plus the one thing it directly produces."
    )
    _pipeline_html = (Path(__file__).parent / "pipeline_map.html").read_text(encoding="utf-8")
    components.html(_pipeline_html, height=2000, scrolling=True)


_SOURCES = {
    "elab": ("eLabFTW", "https://researchmcp.duckdns.org/el/register"),
    "dt": ("DataTagger", "https://researchmcp.duckdns.org/dt/register"),
}


def _chat_placeholder() -> str:
    return (f"Ask about {len(server.papers)} papers across {len(server._PIS)} PIs/groups "
            f"in the cluster…")


def _resolve_llm() -> dict:
    """Provider + model from the session (chosen in the bottom bar). No widgets here."""
    providers = _CFG.providers or {}
    names = list(providers)
    default_provider = next(
        (x for x, p in providers.items() if p.base_url == _CFG.llm.base_url), names[0] if names else "default"
    )
    provider_name = st.session_state.get("provider_name") or default_provider
    if provider_name not in providers:
        provider_name = default_provider
    st.session_state["provider_name"] = provider_name
    prov = providers.get(provider_name)
    if prov is None:
        return {"provider": "default", "base_url": BASE_URL, "model": _CFG.llm.default_model,
                "api_key": os.environ.get("API_KEY", ""), "extra": None}
    models = list(prov.models) or [prov.default_model]
    api_key = os.environ.get(prov.api_key_env, "")
    chosen = st.session_state.get("model_name")
    if "openrouter" in prov.base_url:
        if chosen in models:
            model = chosen
        else:
            cheapest, _all = _fetch_openrouter_models(api_key) if api_key else ([], [])
            model = cheapest[0] if cheapest else prov.default_model
    else:
        model = chosen if chosen in models else prov.default_model
    return {"provider": provider_name, "base_url": prov.base_url, "model": model,
            "api_key": api_key, "extra": None}


def _sync_remote_clients() -> None:
    """Hand the session's tokens to the running MCP proxies (no tokens -> local tools only)."""
    rc = mcp_clients.RemoteClients(
        elab_token=st.session_state.get("elab_token") or None,
        dt_token=st.session_state.get("dt_token") or None,
    )
    openai_tools.set_remote_clients(rc)


def _connect_body(kind: str) -> None:
    """Registration page embedded + token field: configure without leaving the app."""
    label, url = _SOURCES[kind]
    st.caption(f"Register {label} below — you stay inside the app. Then paste the token.")
    try:
        st.iframe(url, height=430)
    except Exception:  # noqa: BLE001 -- older/newer API fallback
        components.iframe(url, height=430, scrolling=True)
    token = st.text_input(f"{label} token", type="password", key=f"{kind}_token_input",
                          placeholder="Paste the token from the form above")
    c1, c2 = st.columns(2)
    if c1.button("Connect", type="primary", key=f"connect_{kind}"):
        st.session_state[f"{kind}_token"] = (token or "").strip()
        _sync_remote_clients()
        try:
            found = sum(1 for t in openai_tools.build_chat_tools()
                        if t["function"]["name"].startswith(kind + "_")
                        and "unavailable" not in t["function"]["name"])
        except Exception:  # noqa: BLE001
            found = 0
        st.session_state[f"{kind}_tools"] = found
        if found:
            st.toast(f"{label}: connected ({found} tools)", icon="🟢")
        else:
            st.error(f"{label}: token invalid or expired — register a new one above.")
        st.rerun()
    if c2.button("Disconnect", key=f"disconnect_{kind}"):
        st.session_state[f"{kind}_token"] = ""
        st.session_state[f"{kind}_tools"] = 0
        _sync_remote_clients()
        st.toast(f"{label} disconnected", icon="⚪")
        st.rerun()


@st.dialog("Connect eLabFTW", width="large")
def _dialog_elab() -> None:
    _connect_body("elab")


@st.dialog("Connect DataTagger", width="large")
def _dialog_dt() -> None:
    _connect_body("dt")


def _source_chip(kind: str) -> None:
    """Grey when not connected, green when the source is live. Click opens its setup."""
    label, _url = _SOURCES[kind]
    active = bool(st.session_state.get(f"{kind}_token"))
    count = st.session_state.get(f"{kind}_tools", 0)
    dot = "🟢" if active else "⚪"
    hint = (f"connected — {count} tools available" if active
            else "not connected — click to register and paste a token")
    if st.button(f"{dot} {label}", key=f"chip_{kind}", help=hint, width="stretch"):
        (_dialog_elab if kind == "elab" else _dialog_dt)()


def _model_popover(llm: dict) -> None:
    """One line with the current choice; clicking lists the models grouped by provider."""
    with st.popover(f"{llm['provider']} / {llm['model']}", width="stretch",
                    help="Switch provider or model"):
        for name, prov in (_CFG.providers or {}).items():
            st.markdown(f"**{name}**")
            for m in (list(prov.models) or [prov.default_model]):
                picked = (name == st.session_state.get("provider_name")
                          and m == st.session_state.get("model_name"))
                if st.button(("✓ " if picked else "") + m, key=f"pick_{name}_{m}",
                             width="stretch"):
                    st.session_state["provider_name"] = name
                    st.session_state["model_name"] = m
                    st.rerun()
        if any("openrouter" in p.base_url for p in (_CFG.providers or {}).values()):
            st.caption("OpenRouter: without a pick, the cheapest model allowed by the account "
                       "guardrails is used automatically.")
        st.caption("Tools: " + _tool_inventory_text())


def _feedback_popover() -> None:
    """Always visible bottom-right: report a bug or leave a note about the last answer."""
    msgs = st.session_state.get("messages") or []
    last = msgs[-1] if msgs else {}
    if last.get("role") == "assistant":
        answer = last.get("content", "")
        question = msgs[-2]["content"] if len(msgs) > 1 else ""
        model = (last.get("meta") or {}).get("model", "")
    else:
        answer, question, model = "", "", st.session_state.get("model_name", "")
    with st.popover("💬 Feedback", help="Bug report or note about the last answer"):
        with st.form(key="feedback_form", clear_on_submit=True, border=False):
            category = st.radio("Type", ["Bug report", "General feedback"], horizontal=True)
            text = st.text_area("What happened, or what would you like to see?")
            if st.form_submit_button("Submit"):
                if text.strip():
                    _record_feedback(question, answer, model, category, text.strip())
                    st.success("Thanks — recorded.")
                else:
                    st.warning("Add a note before submitting.")


def _settings_row() -> None:
    """Everything below the chat window: data sources, model, feedback."""
    _sync_remote_clients()
    left, mid, right = st.columns([1.6, 3.4, 1.4], vertical_alignment="center")
    with left:
        c1, c2 = st.columns(2)
        with c1:
            _source_chip("elab")
        with c2:
            _source_chip("dt")
    with mid:
        _model_popover(_resolve_llm())
    with right:
        _feedback_popover()


def _render_answer(msg: dict, key: str) -> None:
    """Tool calls + timing above the answer, copy button below — nothing else."""
    meta = msg.get("meta") or {}
    calls = meta.get("tool_calls") or []
    elapsed = meta.get("elapsed")
    timing = f"researched for {elapsed:.1f} s" if elapsed else ""
    if calls:
        label = f"⚙ {len(calls)} tool call" + ("s" if len(calls) != 1 else "")
        if timing:
            label += f" · {timing}"
        with st.expander(label, expanded=False):
            for call in calls:
                st.code(call, language=None)
    elif timing:
        st.caption(timing)
    st.markdown(msg.get("content", ""))
    _copy_button(msg.get("content", ""), key)


def _copy_button(text: str, key: str) -> None:
    """Copy the answer as markdown (Streamlit has no native clipboard button)."""
    if not text:
        return
    components.html(
        "<button id='copy' style='font:inherit;font-size:0.85rem;padding:0.2rem 0.6rem;"
        "border:1px solid rgba(128,128,128,.4);border-radius:0.5rem;background:transparent;"
        "color:inherit;cursor:pointer'>📋 Copy markdown</button>"
        "<script>const b=document.getElementById('copy');"
        "b.onclick=async()=>{try{await navigator.clipboard.writeText(" + json.dumps(text) + ");"
        "b.textContent='✓ copied';setTimeout(()=>b.textContent='📋 Copy markdown',1500);}"
        "catch(e){b.textContent='copy failed';}};</script>",
        height=38,
    )


def _run_turn(prompt: str) -> None:
    """Answer one question, log the privacy-safe turn record and render it."""
    llm = _resolve_llm()
    if not llm["api_key"]:
        st.error(f"No API key for provider '{llm['provider']}' — set it in the environment "
                 f"or .env at the repo root and restart the app.")
        st.stop()
    client = OpenAI(api_key=llm["api_key"], base_url=llm["base_url"])

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    local_tools = len(openai_tools.build_openai_tools())
    tools_available = {
        "local": local_tools,
        "elab": st.session_state.get("elab_tools", 0) if st.session_state.get("elab_token") else 0,
        "dt": st.session_state.get("dt_tools", 0) if st.session_state.get("dt_token") else 0,
    }
    tools_available["total"] = sum(tools_available.values())

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                answer, tool_calls, elapsed, turn_meta = _answer(
                    client, llm["model"], api_msgs, extra=llm.get("extra")
                )
            except Exception as exc:
                answer = f"Error: {exc}"
                tool_calls = []
                elapsed = None
                turn_meta = {"rounds": 0, "tools": [], "usage": {}, "error": type(exc).__name__}

        # Telemetry: exactly one privacy-safe line per turn (hashes and lengths
        # only). Raw prompt/answer text never enters the log — it is written
        # only when the user submits the 💬 Feedback form.
        st.session_state["turns"] = st.session_state.get("turns", 0) + 1
        st.session_state["tool_calls_total"] = (
            st.session_state.get("tool_calls_total", 0) + len(tool_calls or [])
        )
        telemetry.log_turn(
            session=st.session_state["session_id"],
            turn=st.session_state["turns"],
            provider=llm["provider"],
            model=llm["model"],
            base_url=llm["base_url"],
            tools_available=tools_available,
            rounds=turn_meta.get("rounds", 0),
            tools=turn_meta.get("tools") or [],
            prompt=prompt,
            answer=answer,
            usage=turn_meta.get("usage") or {},
            latency_ms=(elapsed or 0) * 1000,
            error=turn_meta.get("error"),
        )
        meta = {"model": llm["model"], "elapsed": elapsed, "tool_calls": tool_calls or []}
        _render_answer({"content": answer, "meta": meta},
                       key=f"copy_{len(st.session_state.messages)}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})


def _view_chat() -> None:
    """Chat page: clickable starters, history, chat bar with the settings below it."""
    st.markdown(f"##### {APP_TITLE}")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.setdefault("session_id", telemetry.new_session_id())
    st.session_state.setdefault("turns", 0)
    st.session_state.setdefault("tool_calls_total", 0)

    if not st.session_state.messages:
        st.caption("New here? Start with one of these:")
        for start in range(0, len(_EXAMPLES), 2):
            row = _EXAMPLES[start:start + 2]
            cols = st.columns(len(row))
            for col, question in zip(cols, row):
                with col:
                    if st.button(question, key=f"example_{abs(hash(question)) % 10 ** 8}",
                                 width="stretch"):
                        st.session_state["pending_prompt"] = question
                        st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_answer(msg, key=f"copy_{i}")
            else:
                st.markdown(msg["content"])

    with st.bottom:
        prompt = st.chat_input(_chat_placeholder())
        _settings_row()

    if prompt:
        _run_turn(prompt)


# ---------------------------------------------------------------------------
# Navigation: one page per view, in the Streamlit menu at the top (no sidebar)
# ---------------------------------------------------------------------------
_PAGES = [
    st.Page(_view_chat, title="Chat", icon="💬", url_path="chat", default=True),
    st.Page(_view_corpus_map, title="Corpus Map", icon="🗺️", url_path="corpus-map"),
    st.Page(_view_collaboration, title="Collaboration", icon="🤝", url_path="collaboration"),
    st.Page(_view_pipeline, title="Pipeline", icon="🔧", url_path="pipeline"),
]

_page_setup()
st.navigation(_PAGES, position="top").run()
