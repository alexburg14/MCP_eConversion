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

import streamlit as st
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
import server  # loads all caches at import time

_REPO_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "https://chat-ai.academiccloud.de/v1"

# Tool calling verified against the live endpoint on 2026-06-12; the meeting
# goal "mit welchen Modellen gut? Schlecht?" wants side-by-side comparison,
# so the model is a sidebar choice rather than a constant.
MODELS = [
    "qwen3.5-122b-a10b",
    "glm-4.7",
    "openai-gpt-oss-120b",
    "mistral-large-3-675b-instruct-2512",
]


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
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Lexical (BM25) search over e-conversion cluster publications. "
                "Best for exact terminology, acronyms, formulas, or author names — "
                "any query where the user's words are likely to appear verbatim. "
                "Returns the top 5 matching papers with titles, authors, abstracts, "
                "and any linked datasets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_papers",
            "description": (
                "Semantic (embedding) search over e-conversion cluster publications. "
                "Best for conceptual queries where the user's vocabulary may differ "
                "from the abstracts (synonyms, paraphrases, lay descriptions). "
                "Example: 'splitting water with sunlight' finds photocatalytic OER "
                "papers that never use those exact words. For exact terms, prefer search_papers. "
                "Returns the top 5 matching papers ranked by cosine similarity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Conceptual / semantic search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_by_doi",
            "description": "Return full metadata and abstract for a single paper given its DOI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "DOI of the paper, e.g. 10.1103/physrevb.110.125202"}
                },
                "required": ["doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_fulltext",
            "description": (
                "Return full-text markdown for a paper by DOI. "
                "Only available for the ~42% of open-access papers in the cache. "
                "Use when the abstract is insufficient and deeper content is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doi": {"type": "string", "description": "DOI of the paper"}
                },
                "required": ["doi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pis",
            "description": (
                "Search e-conversion principal investigators (PIs) by name, research area, "
                "or application field keyword. Returns the top 5 matching PIs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keyword query (name, topic, field)"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pi",
            "description": (
                "Return the full profile of a PI by name (last name or full name). "
                "Includes group, institution, research focus, application fields, website, "
                "and up to 10 of their publications from the abstract cache."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "PI last name or full name, e.g. 'Rinke' or 'Patrick Rinke'"}
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_collaborators",
            "description": (
                "Return all PIs who share at least one publication with the queried PI, "
                "ordered by number of shared papers. Use for 'who does X work with?' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pi_query": {"type": "string", "description": "PI last name or full name, e.g. 'Rinke'"}
                },
                "required": ["pi_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "joint_papers",
            "description": (
                "Return the DOIs of papers co-authored by two specific PIs. "
                "Returns count=0 and empty list if they have never co-published."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pi_a": {"type": "string", "description": "First PI last name or full name"},
                    "pi_b": {"type": "string", "description": "Second PI last name or full name"},
                },
                "required": ["pi_a", "pi_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collaboration_centrality",
            "description": (
                "Return PIs ranked by betweenness centrality — those who bridge otherwise "
                "separate research groups. Use for 'who are the connectors in the cluster?' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {"type": "integer", "description": "How many top PIs to return (default 10)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "collaboration_communities",
            "description": (
                "Return communities (clusters) of PIs who collaborate more internally than externally, "
                "detected via greedy modularity on the co-authorship graph. "
                "Use for 'which groups work closely together?' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

_BASE_SYSTEM = """\
You are a research assistant for the e-conversion energy research cluster, \
a consortium of ~42 research groups at TUM, LMU, FHI, and MPI focused on energy conversion.

You have access to a local database of 956 cluster publications (with abstracts and \
some full texts) and profiles of 42 PIs. Use the tools to retrieve relevant information \
before answering. Always cite papers by title and DOI. If information is missing from \
the database, say so clearly — do not invent facts.

Two paper-search tools complement each other:
- search_papers (BM25, lexical): exact terms, acronyms, formulas, author names.
- semantic_search_papers (embeddings): conceptual queries where vocabulary may
  differ from abstracts. Run both when you're unsure which will hit — the
  union of results gives wider recall before you synthesize.

Four collaboration-graph tools answer network questions that search cannot:
- get_collaborators(pi_query): who publishes with a given PI?
- joint_papers(pi_a, pi_b): which papers did two specific PIs co-author?
- collaboration_centrality(): which PIs bridge otherwise-separate groups?
- collaboration_communities(): which clusters of PIs work closely together?

For corpus-wide questions ("main open challenges", "trends over time", \
"complementary groups"), issue several complementary queries before answering: \
one tool call returns at most 5 papers, but a synthesis question needs evidence \
from many. Iterate with different phrasings, then summarize.

Answer in the same language as the question (German or English).\
"""

_PROPOSAL_SUMMARY_PATH = _REPO_ROOT / "data" / "proposal_summary.md"


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


def _dispatch(name: str, inputs: dict) -> str:
    dispatch = {
        "search_papers": lambda i: server.search_papers(i["query"]),
        "semantic_search_papers": lambda i: server.semantic_search_papers(i["query"]),
        "get_paper_by_doi": lambda i: server.get_paper_by_doi(i["doi"]),
        "get_paper_fulltext": lambda i: server.get_paper_fulltext(i["doi"]),
        "search_pis": lambda i: server.search_pis(i["query"]),
        "get_pi": lambda i: server.get_pi(i["name"]),
        "get_collaborators": lambda i: server.get_collaborators(i["pi_query"]),
        "joint_papers": lambda i: server.joint_papers(i["pi_a"], i["pi_b"]),
        "collaboration_centrality": lambda i: server.collaboration_centrality(i.get("top_k", 10)),
        "collaboration_communities": lambda i: server.collaboration_communities(),
    }
    fn = dispatch.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(inputs)
    except KeyError as exc:
        return json.dumps({"error": f"Missing argument for {name}: {exc}"})


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
            result = _dispatch(tc.function.name, args)
            tool_log.append(f"`{tc.function.name}({(tc.function.arguments or '')[:80]})`")
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "Tool-call limit reached without a final answer — try rephrasing the question.", tool_log


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="eConversion Assistant", page_icon="⚡", layout="centered")
st.title("⚡ eConversion Knowledge Assistant")
st.caption("956 publications · 42 PIs · TUM / LMU / FHI / MPI FKF")

_load_dotenv()
api_key = os.environ.get("API_KEY", "")
if not api_key:
    st.error("Set `API_KEY` in your environment or in `.env` at the repo root and restart the app.")
    st.stop()

model = st.sidebar.selectbox("Model", MODELS, index=0)
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
