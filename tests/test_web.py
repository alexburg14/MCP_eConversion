"""HTTP API tests against an injected AppState (no caches, no network, fake LLM).

The ``harness`` / ``state`` fixtures live in conftest.py (shared with test_web_live.py)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import auth  # noqa: E402
import web  # noqa: E402
from conftest import LOCAL_TOOL  # noqa: E402
from fakes import text_chunk, tool_chunk  # noqa: E402

@pytest.fixture()
def client(state):
    with TestClient(web.create_app(state, root_path="")) as c:
        yield c


def events(response):
    """Parse SSE frames from a streamed response into (event, data) pairs."""
    out = []
    for block in response.text.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln and not ln.startswith(":")]
        if not lines:
            continue
        ev = next(ln[len("event: "):] for ln in lines if ln.startswith("event: "))
        data = json.loads(next(ln[len("data: "):] for ln in lines if ln.startswith("data: ")))
        out.append((ev, data))
    return out


def chat(client, prompt):
    with client.stream("POST", "/api/chat", json={"prompt": prompt}) as r:
        r.read()
        return r


def test_config_lists_providers_and_examples(client):
    cfg = client.get("/api/config").json()
    assert cfg["title"] == web.APP_TITLE
    assert "gwdg" in cfg["providers"] and cfg["default_provider"] == "gwdg"
    assert cfg["examples"] and cfg["sources"]["elab"]["label"] == "eLabFTW"
    assert "3 papers" in cfg["placeholder"]


def test_session_endpoint_creates_cookie_and_empty_history(client):
    r = client.get("/api/session")
    assert r.status_code == 200 and auth.COOKIE_NAME in r.headers["set-cookie"]
    body = r.json()
    assert body["messages"] == [] and body["turns"] == 0 and body["busy"] is False
    assert body["provider"] == "gwdg" and body["model"] == "qwen3.8-27b"
    assert body["tools"] == {"local": 1, "elab": 0, "dt": 0, "total": 1}


def test_chat_requires_an_existing_session(client):
    r = client.post("/api/chat", json={"prompt": "hi"})
    assert r.status_code == 401


def test_chat_streams_events_and_commits_history(client, harness, tmp_path, monkeypatch):
    harness.scripts = [[tool_chunk(0, id="c", name="search_papers", arguments='{"q": 1}')],
                       [text_chunk("Hel"), text_chunk("lo")]]
    client.get("/api/session")
    r = chat(client, "  hello  ")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-accel-buffering"] == "no"
    evs = events(r)
    assert [e for e, _ in evs] == ["start", "round", "tool_call_start", "tool_call_end",
                                   "round", "text_delta", "text_delta", "done"]
    assert evs[0][1]["model"] == "qwen3.8-27b" and evs[0][1]["turn"] == 1
    assert evs[-1][1]["answer"] == "Hello"
    assert harness.tool_calls == [("search_papers", {"q": 1}, None)]
    # the fake client got the system prompt + the trimmed user message
    msgs = harness.clients[0].calls[0]["messages"]
    assert msgs[0]["content"] == "SYS" and msgs[1] == {"role": "user", "content": "hello"}
    assert harness.clients[0].calls[0]["tools"] == [LOCAL_TOOL]
    view = client.get("/api/session").json()
    assert view["turns"] == 1 and [m["role"] for m in view["messages"]] == ["user", "assistant"]
    assert view["messages"][1]["content"] == "Hello"
    assert view["messages"][1]["meta"]["tool_calls"] == ["`search_papers({\"q\": 1})`"]
    assert view["messages"][1]["meta"]["model"] == "qwen3.8-27b"


def test_second_turn_resends_the_whole_history(client, harness):
    harness.scripts = [[text_chunk("one")], [text_chunk("two")]]
    client.get("/api/session")
    chat(client, "q1")
    chat(client, "q2")
    msgs = harness.clients[1].calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "one" and "meta" not in msgs[2]


def test_llm_exception_yields_error_frame_and_error_history(client, harness):
    harness.scripts = [RuntimeError("upstream down")]
    client.get("/api/session")
    evs = events(chat(client, "q"))
    assert [e for e, _ in evs] == ["start", "round", "error"]
    assert evs[-1][1]["error_type"] == "RuntimeError" and evs[-1][1]["rounds"] == 1
    view = client.get("/api/session").json()
    assert view["messages"][-1]["content"].startswith("Error: upstream down")
    assert view["busy"] is False


def test_missing_api_key_is_a_400_not_a_stream(client, monkeypatch):
    monkeypatch.delenv("API_KEY")
    monkeypatch.delenv("ECONVERSION_API_KEY", raising=False)
    client.get("/api/session")
    r = client.post("/api/chat", json={"prompt": "q"})
    assert r.status_code == 400 and "API_KEY" in r.json()["error"]


def _session_of(client, state):
    return state.sessions.get(client.cookies[auth.COOKIE_NAME])


def test_busy_session_gets_409_and_reset_refuses(client, state):
    # The test client buffers streamed bodies, so hold the turn lock directly;
    # true interleaving is covered by test_web_live.py.
    client.get("/api/session")
    session = _session_of(client, state)
    assert session.turn_lock.acquire(blocking=False)
    try:
        assert client.post("/api/chat", json={"prompt": "again"}).status_code == 409
        assert client.post("/api/chat/reset").status_code == 409
        assert client.get("/api/session").json()["busy"] is True
    finally:
        session.turn_lock.release()
    assert client.post("/api/chat/reset").status_code == 200


def test_stop_sets_the_running_turns_cancel_flag(client, state):
    client.get("/api/session")
    assert client.post("/api/chat/stop").json()["stopped"] is False
    session = _session_of(client, state)
    session.cancel = threading.Event()
    assert client.post("/api/chat/stop").json()["stopped"] is True
    assert session.cancel.is_set()


def test_reset_starts_a_new_conversation_with_a_new_id(client, harness):
    sid = client.get("/api/session").json()["session"]
    chat(client, "q")
    r = client.post("/api/chat/reset").json()
    assert r["session"] != sid
    assert client.get("/api/session").json()["messages"] == []


def test_model_selection_is_validated_and_reflected_in_the_stream(client, harness):
    client.get("/api/session")
    assert client.post("/api/session/model", json={"provider": "nope", "model": ""}).status_code == 400
    assert client.post("/api/session/model", json={"provider": "gwdg", "model": "nope"}).status_code == 400
    r = client.post("/api/session/model", json={"provider": "gwdg", "model": "glm-5.3-flash"})
    assert r.json() == {"provider": "gwdg", "model": "glm-5.3-flash", "auto": False}
    evs = events(chat(client, "q"))
    assert evs[0][1]["model"] == "glm-5.3-flash"
    assert client.get("/api/session").json()["messages"][-1]["meta"]["model"] == "glm-5.3-flash"


def test_openrouter_without_pick_reports_auto(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    client.get("/api/session")
    r = client.post("/api/session/model", json={"provider": "openrouter", "model": ""}).json()
    assert r == {"provider": "openrouter", "model": "", "auto": True}


def test_feedback_records_the_last_turn(client, harness, tmp_path):
    harness.scripts = [[text_chunk("The answer")]]
    client.get("/api/session")
    chat(client, "The question")
    r = client.post("/api/feedback", json={"category": "Bug report", "text": " broken "})
    assert r.status_code == 200
    line = json.loads((tmp_path / "feedback.jsonl").read_text().strip())
    assert line["question"] == "The question" and line["answer"] == "The answer"
    assert line["model"] == "qwen3.8-27b" and line["text"] == "broken" and line["provider"] == "gwdg"
    assert client.post("/api/feedback", json={"category": "Praise", "text": "x"}).status_code == 400
    assert client.post("/api/feedback", json={"category": "Bug report", "text": "   "}).status_code == 400


def test_stats_are_structured(client):
    s = client.get("/api/stats").json()
    assert s["usage"]["turns"] == 1 and s["tools"][0]["name"] == "search_papers"
    assert [p["key"] for p in s["pipeline"]][:2] == ["papers", "abstracts"]
    assert s["pipeline"][0]["entries"] == 3 and s["tools_local"] == 1


def test_corpus_map_and_collaboration_graph_endpoints(client):
    m = client.get("/api/corpus-map", params={"clusters": 3}).json()
    assert m["available"] is True and len(m["points"]) == 3 and m["legend"]
    assert client.get("/api/collaboration-graph").json() == {"nodes": [], "links": []}


def test_connect_success_and_failure(client, harness, monkeypatch):
    class FakeRemote:
        def __init__(self, elab_token=None, dt_token=None):
            self.elab_token, self.dt_token = elab_token, dt_token

        def build_openai_tools(self):
            if self.elab_token == "good":
                return [{"type": "function", "function": {"name": "elab_list", "description": "d",
                                                          "parameters": {"type": "object", "properties": {}}}}]
            return [{"type": "function", "function": {
                "name": "elab___unavailable__", "description": "Source elab is currently unavailable. Error: Token invalid or expired",
                "parameters": {"type": "object", "properties": {}}}}]

        def call(self, name, arguments):
            return json.dumps({"remote": name})

    monkeypatch.setattr(client.app.state.app_state, "remote_factory", FakeRemote)
    client.get("/api/session")
    bad = client.post("/api/session/connect/elab", json={"token": "bad"}).json()
    assert bad["active"] is False and "invalid" in bad["error"].lower()
    assert client.get("/api/session").json()["connected"]["elab"]["active"] is False

    good = client.post("/api/session/connect/elab", json={"token": "good"}).json()
    assert good == {"kind": "elab", "active": True, "tools": 1, "error": None}
    view = client.get("/api/session").json()
    assert view["connected"]["elab"] == {"active": True, "tools": 1}
    assert view["tools"]["total"] == 2

    # the remote tool reaches the model and the remote note reaches the system prompt
    harness.scripts = [[text_chunk("x")]]
    chat(client, "q")
    call = harness.clients[0].calls[0]
    assert [t["function"]["name"] for t in call["tools"]] == ["search_papers", "elab_list"]
    assert "eLabFTW" in call["messages"][0]["content"]

    assert client.delete("/api/session/connect/elab").json()["active"] is False
    assert client.get("/api/session").json()["tools"]["total"] == 1
    assert client.post("/api/session/connect/nope", json={"token": "x"}).status_code == 404


def test_two_sessions_are_isolated(state, harness):
    app = web.create_app(state, root_path="")
    with TestClient(app) as a, TestClient(app) as b:
        harness.scripts = [[text_chunk("A")], [text_chunk("B")]]
        a.get("/api/session"); b.get("/api/session")
        a.post("/api/session/model", json={"provider": "gwdg", "model": "glm-5.3-flash"})
        chat(a, "from a")
        chat(b, "from b")
        va, vb = a.get("/api/session").json(), b.get("/api/session").json()
        assert va["session"] != vb["session"]
        assert va["messages"][0]["content"] == "from a" and vb["messages"][0]["content"] == "from b"
        assert va["model"] == "glm-5.3-flash" and vb["model"] == "qwen3.8-27b"


def test_root_path_mounts_everything_under_the_prefix(state):
    app = web.create_app(state, root_path="/nomad-oasis/api/everse")
    with TestClient(app) as c:
        assert c.get("/api/session").status_code == 404
        r = c.get("/nomad-oasis/api/everse/api/session")
        assert r.status_code == 200 and "Path=/nomad-oasis/api/everse" in r.headers["set-cookie"]
        assert c.get("/nomad-oasis/api/everse/").status_code == 200
