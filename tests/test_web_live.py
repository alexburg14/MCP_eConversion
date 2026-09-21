"""Streaming behaviour that needs a real server: incremental SSE delivery, the
busy lock while a turn is in flight, and cancellation when the client leaves.

Runs uvicorn on a free localhost port in a thread with the same fake state as
test_web.py (no caches, no LLM).
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import auth  # noqa: E402
import web  # noqa: E402
from fakes import text_chunk, tool_chunk  # noqa: E402


@pytest.fixture()
def live(state):
    app = web.create_app(state, root_path="")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def _read_until(lines, prefix, limit=40):
    """Consume SSE lines until one starts with ``prefix``; return everything read."""
    seen = []
    for _ in range(limit):
        ln = next(lines)
        seen.append(ln)
        if ln.startswith(prefix):
            return seen
    raise AssertionError(f"{prefix!r} not seen in {seen}")


def _wait(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_events_arrive_before_the_turn_finishes_and_busy_is_409(live, harness):
    harness.gate = threading.Event()
    harness.scripts = [[text_chunk("pre "), tool_chunk(0, id="c", name="search_papers", arguments="{}")],
                       [text_chunk("post")]]
    with httpx.Client(base_url=live, timeout=10) as c:
        c.get("/api/session")
        with c.stream("POST", "/api/chat", json={"prompt": "q"}) as r:
            lines = r.iter_lines()
            assert next(lines) == "event: start"
            assert _wait(lambda: harness.tool_calls)  # the tool is now blocked on the gate
            # the text before the tool call has already been delivered
            seen = _read_until(lines, "event: tool_call_start")
            assert any(ln.startswith("event: text_delta") for ln in seen)
            assert c.post("/api/chat", json={"prompt": "again"}).status_code == 409
            assert c.get("/api/session").json()["busy"] is True
            harness.gate.set()
            rest = list(lines)
        assert any(ln.startswith("event: done") for ln in rest)
        view = c.get("/api/session").json()
        assert view["busy"] is False and view["messages"][-1]["content"] == "pre \n\npost"


def test_client_disconnect_cancels_the_turn(live, harness, state):
    harness.gate = threading.Event()
    harness.scripts = [[tool_chunk(0, id="c", name="search_papers", arguments="{}")], [text_chunk("never")]]
    with httpx.Client(base_url=live, timeout=10) as c:
        c.get("/api/session")
        session = state.sessions.get(c.cookies[auth.COOKIE_NAME])
        with c.stream("POST", "/api/chat", json={"prompt": "q"}) as r:
            next(r.iter_lines())
            assert _wait(lambda: harness.tool_calls)
            cancel = session.cancel
        # connection closed while the tool is still running: the server notices
        # the disconnect and flags the turn before the tool even returns
        assert _wait(lambda: cancel.is_set())
        harness.gate.set()
        assert _wait(lambda: not c.get("/api/session").json()["busy"])
        view = c.get("/api/session").json()
        assert [m["role"] for m in view["messages"]] == ["user", "assistant"]
        assert len(harness.clients[0].calls) == 1  # no second model round after the cancel
