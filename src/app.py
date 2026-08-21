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
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import server  # loads all caches at import time
import corpus_map
from config import get_config
import openai_tools

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CFG = get_config()

BASE_URL = _CFG.llm.base_url

# Tool calling verified against the live endpoint on 2026-06-12; the meeting
# goal "mit welchen Modellen gut? Schlecht?" wants side-by-side comparison,
# so the model is a sidebar choice rather than a constant.
MODELS = list(_CFG.llm.models)


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

_BASE_SYSTEM = f"""\
You are a research assistant for {_CFG.cluster.description}.

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


def _answer(client: OpenAI, model: str, messages: list[dict]) -> tuple[str, list[str]]:
    """Run the tool-use loop; return (answer_text, list_of_tool_calls_summary)."""
    tool_log: list[str] = []
    msgs = [{"role": "system", "content": _SYSTEM}] + list(messages)

    for _ in range(_MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=msgs,
            tools=_TOOLS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or "", tool_log

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
            result = openai_tools.call_tool(tc.function.name, args)
            tool_log.append(f"`{tc.function.name}({(tc.function.arguments or '')[:80]})`")
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Tool-call limit reached without a final answer — try rephrasing the question.", tool_log


@st.cache_data(show_spinner="Computing corpus map (UMAP + clustering)...")
def _build_corpus_map(n_clusters: int) -> list[dict]:
    return corpus_map.build_map(server.papers_by_doi, n_clusters=n_clusters)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title=_CFG.cluster.display_name, page_icon="⚡", layout="centered")
st.title(f"⚡ {_CFG.cluster.display_name}")
st.caption(f"{len(server.papers)} publications · {len(server._PIS)} PIs")

tab_chat, tab_map = st.tabs(["💬 Chat", "🗺️ Corpus Map"])

# Corpus map only needs the embeddings cache, not the API key — render it
# before the chat tab's st.stop() so a missing key doesn't hide it too.
with tab_map:
    st.caption(
        "UMAP layout of the paper embeddings; KMeans clusters (computed in the "
        "full 384-d space) labeled with their top title keywords. Hover a point "
        "for title/year — a visual answer to 'which papers are near the one I'm reading?'"
    )
    if not corpus_map.is_available():
        st.info("Embeddings cache not built. Run: `python src/scripts/build_embeddings_cache.py`")
    else:
        n_clusters = st.slider("Clusters", min_value=2, max_value=20, value=8)
        df = pd.DataFrame(_build_corpus_map(n_clusters))

        highlight = st.selectbox(
            "Highlight a paper (the one you're reading)",
            options=[""] + sorted(df["title"].tolist()),
            format_func=lambda t: t if t else "— none —",
        )

        base = (
            alt.Chart(df)
            .mark_circle(size=45, opacity=0.65)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                color=alt.Color(
                    "cluster:N",
                    legend=alt.Legend(title="Cluster (top title keywords)", labelLimit=280),
                ),
                tooltip=["title:N", "year:N", "doi:N", "cluster:N"],
            )
        )
        if highlight:
            # Ring around the selected paper so its neighborhood is readable at a glance.
            marker = (
                alt.Chart(df[df["title"] == highlight])
                .mark_point(shape="circle", size=400, strokeWidth=3, filled=False, color="red")
                .encode(x="x:Q", y="y:Q", tooltip=["title:N", "year:N", "doi:N"])
            )
            chart = (base + marker).properties(height=600).interactive()
        else:
            chart = base.properties(height=600).interactive()
        st.altair_chart(chart, width="stretch")

with tab_chat:
    _load_dotenv()
    api_key = os.environ.get("API_KEY", "")
    if not api_key:
        st.error("Set `API_KEY` in your environment or in `.env` at the repo root and restart the app.")
        st.stop()

    _default_idx = MODELS.index(_CFG.llm.default_model) if _CFG.llm.default_model in MODELS else 0
    model = st.sidebar.selectbox("Model", MODELS, index=_default_idx)
    st.sidebar.caption(f"Endpoint: {BASE_URL}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about papers, PIs, or research topics..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build messages for API (text-only history)
        api_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    answer, tool_calls = _answer(client, model, api_msgs)
                except Exception as exc:
                    answer = f"Error: {exc}"
                    tool_calls = []

            st.markdown(answer)
            if tool_calls:
                with st.expander("Tools used", expanded=False):
                    for tc in tool_calls:
                        st.code(tc, language=None)

        st.session_state.messages.append({"role": "assistant", "content": answer})
