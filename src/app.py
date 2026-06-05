"""
eConversion Knowledge Assistant — Streamlit chat interface.

Run with:
    streamlit run src/app.py

Requires ANTHROPIC_API_KEY in the environment.
"""
import json
import os
import sys
from pathlib import Path

import anthropic
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import server  # loads all caches at import time

# ---------------------------------------------------------------------------
# Claude tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "search_papers",
        "description": (
            "Search e-conversion cluster publications by keyword. "
            "Returns the top 5 matching papers with titles, authors, abstracts, "
            "and any linked datasets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_paper_by_doi",
        "description": "Return full metadata and abstract for a single paper given its DOI.",
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper, e.g. 10.1103/physrevb.110.125202"}
            },
            "required": ["doi"],
        },
    },
    {
        "name": "get_paper_fulltext",
        "description": (
            "Return full-text markdown for a paper by DOI. "
            "Only available for the ~42% of open-access papers in the cache. "
            "Use when the abstract is insufficient and deeper content is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "DOI of the paper"}
            },
            "required": ["doi"],
        },
    },
    {
        "name": "search_pis",
        "description": (
            "Search e-conversion principal investigators (PIs) by name, research area, "
            "or application field keyword. Returns the top 5 matching PIs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword query (name, topic, field)"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_pi",
        "description": (
            "Return the full profile of a PI by name (last name or full name). "
            "Includes group, institution, research focus, application fields, website, "
            "and up to 10 of their publications from the abstract cache."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "PI last name or full name, e.g. 'Rinke' or 'Patrick Rinke'"}
            },
            "required": ["name"],
        },
    },
]

_SYSTEM = """\
You are a research assistant for the e-conversion energy research cluster, \
a consortium of ~42 research groups at TUM, LMU, FHI, and MPI focused on energy conversion.

You have access to a local database of 956 cluster publications (with abstracts and \
some full texts) and profiles of 42 PIs. Use the tools to retrieve relevant information \
before answering. Always cite papers by title and DOI. If information is missing from \
the database, say so clearly — do not invent facts.

Answer in the same language as the question (German or English).\
"""


def _dispatch(name: str, inputs: dict) -> str:
    dispatch = {
        "search_papers": lambda i: server.search_papers(i["query"]),
        "get_paper_by_doi": lambda i: server.get_paper_by_doi(i["doi"]),
        "get_paper_fulltext": lambda i: server.get_paper_fulltext(i["doi"]),
        "search_pis": lambda i: server.search_pis(i["query"]),
        "get_pi": lambda i: server.get_pi(i["name"]),
    }
    fn = dispatch.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return fn(inputs)


def _answer(client: anthropic.Anthropic, messages: list[dict]) -> tuple[str, list[str]]:
    """Run the tool-use loop; return (answer_text, list_of_tool_calls_summary)."""
    tool_log: list[str] = []
    msgs = list(messages)

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_SYSTEM,
            tools=_TOOLS,
            messages=msgs,
        )

        # Collect tool calls
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if response.stop_reason == "end_turn" or not tool_uses:
            text = " ".join(b.text for b in text_blocks)
            return text, tool_log

        # Execute tools
        msgs.append({"role": "assistant", "content": response.content})
        tool_results = []
        for tu in tool_uses:
            result = _dispatch(tu.name, tu.input)
            tool_log.append(f"`{tu.name}({json.dumps(tu.input, ensure_ascii=False)[:80]})`")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })
        msgs.append({"role": "user", "content": tool_results})


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="eConversion Assistant", page_icon="⚡", layout="centered")
st.title("⚡ eConversion Knowledge Assistant")
st.caption("956 publications · 42 PIs · TUM / LMU / FHI / MPI FKF")

# API key check
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    st.error("Set `ANTHROPIC_API_KEY` in your environment and restart the app.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

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
                answer, tool_calls = _answer(client, api_msgs)
            except Exception as exc:
                answer = f"Error: {exc}"
                tool_calls = []

        st.markdown(answer)
        if tool_calls:
            with st.expander("Tools used", expanded=False):
                for tc in tool_calls:
                    st.code(tc, language=None)

    st.session_state.messages.append({"role": "assistant", "content": answer})
