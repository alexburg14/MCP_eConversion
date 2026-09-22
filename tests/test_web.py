"""HTTP API tests against an injected AppState (no caches, no network, fake LLM).

The ``harness`` / ``state`` fixtures live in conftest.py (shared with test_web_live.py)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import httpx
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


def test_reset_detaches_a_turn_that_will_not_stop(client, state, harness):
    # The test client buffers streamed bodies, so take the slot directly; true
    # interleaving is covered by test_web_live.py.
    client.get("/api/session")
    session = _session_of(client, state)
    epoch, cancel = session.turn.start()
    assert client.get("/api/session").json()["busy"] is True

    r = client.post("/api/chat/reset")
    assert r.status_code == 200 and r.json()["stopped"] is True
    assert cancel.is_set()
    assert client.get("/api/session").json()["busy"] is False
    # the stuck turn eventually gives up; it must not re-lock the fresh session
    assert session.turn.finish(epoch) is False
    harness.scripts = [[text_chunk("after")]]
    assert chat(client, "q").status_code == 200


def test_a_new_prompt_displaces_a_turn_that_will_not_stop(client, state, harness):
    client.get("/api/session")
    session = _session_of(client, state)
    _, cancel = session.turn.start()

    harness.scripts = [[text_chunk("second")]]
    assert chat(client, "q2").status_code == 200
    assert cancel.is_set()
    assert client.get("/api/session").json()["busy"] is False


def test_stop_sets_the_running_turns_cancel_flag(client, state):
    client.get("/api/session")
    assert client.post("/api/chat/stop").json()["stopped"] is False
    session = _session_of(client, state)
    _, cancel = session.turn.start()
    assert client.post("/api/chat/stop").json()["stopped"] is True
    assert cancel.is_set()


def test_reset_starts_a_new_conversation_with_a_new_id(client, harness):
    sid = client.get("/api/session").json()["session"]
    chat(client, "q")
    r = client.post("/api/chat/reset").json()
    assert r["session"] != sid
    assert client.get("/api/session").json()["messages"] == []


def test_gwdg_is_the_fallback_when_openrouter_has_no_key(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert client.get("/api/config").json()["default_provider"] == "gwdg"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert client.get("/api/config").json()["default_provider"] == "openrouter"
    # and a fresh session starts on it
    client.get("/api/session")
    assert client.get("/api/session").json()["provider"] == "openrouter"


def test_model_selection_is_validated_and_reflected_in_the_stream(client, harness):
    client.get("/api/session")
    assert client.post("/api/session/model", json={"provider": "nope", "model": ""}).status_code == 400
    assert client.post("/api/session/model", json={"provider": "gwdg", "model": "nope"}).status_code == 400
    # the gwdg list is deliberately short (one model), openrouter only offers auto
    r = client.post("/api/session/model", json={"provider": "gwdg", "model": "qwen3.8-27b"})
    assert r.json()["model"] == "qwen3.8-27b" and r.json()["auto"] is False
    evs = events(chat(client, "q"))
    assert evs[0][1]["model"] == "qwen3.8-27b"
    assert client.get("/api/session").json()["messages"][-1]["meta"]["model"] == "qwen3.8-27b"


def test_openrouter_without_pick_reports_auto(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    client.get("/api/session")
    r = client.post("/api/session/model", json={"provider": "openrouter", "model": ""}).json()
    assert r["provider"] == "openrouter" and r["model"] == "" and r["auto"] is True
    # no static OpenRouter models are offered any more - auto is the only choice
    assert client.get("/api/config").json()["providers"]["openrouter"]["models"] == []
    assert client.post("/api/session/model",
                       json={"provider": "openrouter", "model": "openai/gpt-4o"}).status_code == 400


def test_feedback_records_the_last_turn(client, harness, tmp_path):
    harness.scripts = [[text_chunk("The answer")]]
    client.get("/api/session")
    chat(client, "The question")
    r = client.post("/api/feedback", json={"category": "Bug report", "text": " broken "})
    assert r.status_code == 200
    line = json.loads((tmp_path / "feedback.jsonl").read_text().strip())
    assert line["question"] == "The question" and line["answer"] == "The answer"
    assert line["model"] == "qwen3.8-27b" and line["text"] == "broken" and line["provider"] == "gwdg"
    assert line["messages"] == [
        {"role": "user", "content": "The question"},
        {"role": "assistant", "content": "The answer"},
    ]
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


def test_test_account_is_offered_only_when_configured(client, monkeypatch):
    sources = client.get("/api/config").json()["sources"]
    assert "test_user" not in sources["dt"]
    # profiles is iterated by the UI -> a list, also when a source has none
    assert sources["dt"]["profiles"] == [] and isinstance(sources["dt"]["profiles"], list)
    assert [p["value"] for p in sources["elab"]["profiles"]] == ["h", "r", "f"]
    assert sources["elab"]["base_url_default"].startswith("https://")
    monkeypatch.setenv("DATATAGGER_TEST_TOKEN", "secret-token")
    body = client.get("/api/config")
    assert body.json()["sources"]["dt"]["test_user"]["label"] == "test account"
    # neither the token nor the env var name is exposed
    assert "test_token_env" not in body.json()["sources"]["dt"]
    assert "secret-token" not in body.text


def test_test_account_endpoint_signs_in(client, monkeypatch):
    monkeypatch.setenv("DATATAGGER_TEST_TOKEN", "dt-good")

    class FakeRemote:
        def __init__(self, elab_token=None, dt_token=None):
            self.elab_token, self.dt_token = elab_token, dt_token

        def build_openai_tools(self):
            if self.dt_token == "dt-good":
                return [{"type": "function", "function": {"name": "dt_list_projects", "description": "d",
                                                          "parameters": {"type": "object", "properties": {}}}}]
            return [{"type": "function", "function": {
                "name": "dt___unavailable__", "description": "Source dt is unavailable. Error: bad token",
                "parameters": {"type": "object", "properties": {}}}}]

    monkeypatch.setattr(client.app.state.app_state, "remote_factory", FakeRemote)
    client.get("/api/session")
    r = client.post("/api/session/connect/dt/test", json={})
    assert r.status_code == 200 and r.json()["active"] is True and r.json()["tools"] == 1
    assert client.get("/api/session").json()["connected"]["dt"] == {"active": True, "tools": 1}
    # without a configured account the endpoint is a 404
    monkeypatch.delenv("DATATAGGER_TEST_TOKEN")
    assert client.post("/api/session/connect/dt/test", json={}).status_code == 404


def test_register_endpoint_creates_a_token_and_connects(client, monkeypatch):
    seen = {}

    class FakeRemote:
        def __init__(self, elab_token=None, dt_token=None):
            self.dt_token = dt_token

        def build_openai_tools(self):
            if self.dt_token == "tok-good":
                return [{"type": "function", "function": {"name": "dt_list_projects", "description": "d",
                                                          "parameters": {"type": "object", "properties": {}}}}]
            return [{"type": "function", "function": {
                "name": "dt___unavailable__", "description": "Source dt unavailable. Error: bad",
                "parameters": {"type": "object", "properties": {}}}}]

    calls = []

    def fake_post(url, data):
        calls.append(dict(data))
        if data.get("validated") == "0":        # validation step answers with a form
            return 200, "<h2>Select profile</h2>"
        return 200, f"<div class=\'url-box\'>{url}?token=tok-good</div>"

    monkeypatch.setattr(client.app.state.app_state, "remote_factory", FakeRemote)
    monkeypatch.setattr(web, "_post_registration", fake_post)
    client.get("/api/session")
    r = client.post("/api/session/register/dt",
                    json={"base_url": "https://datatagger.example", "api_key": "the-key"})
    assert r.status_code == 200 and r.json()["active"] is True and r.json()["tools"] == 1
    # the key is validated first, then the token is minted, and never echoed back
    assert [c["validated"] for c in calls] == ["0", "1"]
    assert all(c["api_key"] == "the-key" for c in calls)
    assert calls[0]["base_url"] == "https://datatagger.example"
    assert "the-key" not in r.text
    session = next(iter(client.app.state.app_state.sessions._by_cookie.values()))
    assert session.dt_token == "tok-good"


def test_register_endpoint_reports_bad_keys_and_missing_input(client, monkeypatch):
    calls = []

    def reject(url, data):
        calls.append(data.get("validated"))
        return 401, "<h2>Invalid Key</h2>"

    monkeypatch.setattr(web, "_post_registration", reject)
    client.get("/api/session")
    bad = client.post("/api/session/register/dt",
                      json={"base_url": "https://dt.example", "api_key": "nope"})
    assert bad.status_code == 400 and "rejected" in bad.json()["detail"]["error"].lower()
    # a key rejected in the validation step never reaches the token step
    assert calls == ["0"]
    assert client.post("/api/session/register/dt", json={"api_key": ""}).status_code == 400
    assert client.post("/api/session/register/nope", json={"api_key": "x"}).status_code == 404
    # profile defaults to the first configured one for eLabFTW
    seen = {}
    monkeypatch.setattr(web, "_post_registration",
                        lambda url, data: (seen.update(data), (400, "x"))[1])
    client.post("/api/session/register/elab", json={"api_key": "k", "profile": "nonsense"})
    assert seen["profile"] == "h"


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


class FakeUpstream:
    """Stands in for httpx.AsyncClient: records the one request, replays a canned answer."""

    def __init__(self, status=200, content_type="text/html; charset=utf-8",
                 text="<html>form</html>", raise_exc=None):
        self.status, self.content_type, self.text, self.raise_exc = status, content_type, text, raise_exc
        self.seen = []

    def __call__(self, *a, **kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def request(self, method, url, content=None, headers=None):
        self.seen.append({"method": method, "url": url, "content": content, "headers": headers})
        if self.raise_exc:
            raise self.raise_exc
        return httpx.Response(self.status, headers={"content-type": self.content_type,
                                                    "x-frame-options": "SAMEORIGIN",
                                                    "set-cookie": "upstream=1"}, text=self.text)


def test_register_proxy_mirrors_the_page_and_drops_the_framing_headers(client, monkeypatch):
    fake = FakeUpstream(text="<html>register form</html>")
    monkeypatch.setattr(web.httpx, "AsyncClient", fake)
    r = client.get("/api/register/elab")
    assert r.status_code == 200 and "register form" in r.text
    assert "x-frame-options" not in r.headers and "set-cookie" not in r.headers
    assert fake.seen[0]["url"] == web.SOURCES["elab"]["register_url"]


def test_register_proxy_forwards_the_submitted_form(client, monkeypatch):
    fake = FakeUpstream(status=401, text="<html>API key rejected</html>")
    monkeypatch.setattr(web.httpx, "AsyncClient", fake)
    r = client.post("/api/register/dt", content=b"api_key=secret",
                    headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 401 and "rejected" in r.text
    sent = fake.seen[0]
    assert sent["method"] == "POST" and sent["content"] == b"api_key=secret"
    assert sent["url"] == web.SOURCES["dt"]["register_url"]
    assert sent["headers"]["content-type"] == "application/x-www-form-urlencoded"


@pytest.mark.parametrize("fake", [
    FakeUpstream(raise_exc=httpx.ConnectError("down")),
    FakeUpstream(status=503, content_type="application/json", text="{}"),
])
def test_register_proxy_falls_back_to_a_new_tab_link(client, monkeypatch, fake):
    monkeypatch.setattr(web.httpx, "AsyncClient", fake)
    r = client.get("/api/register/elab")
    assert r.status_code == 502
    assert web.SOURCES["elab"]["register_url"] in r.text and "new tab" in r.text


def test_static_assets_must_revalidate(client):
    """The frontend has no versioned URLs: a cached app.js would outlive a deploy."""
    r = client.get("/static/js/app.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert r.headers.get("etag")


def test_register_proxy_rejects_an_unknown_source(client):
    assert client.get("/api/register/nope").status_code == 404


def test_two_sessions_are_isolated(state, harness):
    app = web.create_app(state, root_path="")
    with TestClient(app) as a, TestClient(app) as b:
        harness.scripts = [[text_chunk("A")], [text_chunk("B")]]
        a.get("/api/session"); b.get("/api/session")
        a.post("/api/session/model", json={"provider": "gwdg", "model": "qwen3.8-27b"})
        a.post("/api/session/params", json={"params": {"top_p": "0.9"}})
        chat(a, "from a")
        chat(b, "from b")
        va, vb = a.get("/api/session").json(), b.get("/api/session").json()
        assert va["session"] != vb["session"]
        assert va["messages"][0]["content"] == "from a" and vb["messages"][0]["content"] == "from b"
        assert va["model"] == "qwen3.8-27b" and vb["model"] == "qwen3.8-27b"
        # parameters are per session too
        assert va["params"]["top_p"] == "0.9" and vb["params"]["top_p"] == ""


def test_root_path_mounts_everything_under_the_prefix(state):
    app = web.create_app(state, root_path="/nomad-oasis/api/everse")
    with TestClient(app) as c:
        assert c.get("/api/session").status_code == 404
        r = c.get("/nomad-oasis/api/everse/api/session")
        assert r.status_code == 200 and "Path=/nomad-oasis/api/everse" in r.headers["set-cookie"]
        assert c.get("/nomad-oasis/api/everse/").status_code == 200
