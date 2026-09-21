"""Test fixtures shared across the suite.

Two kinds of tests live here:

* hermetic ones (agent, auth, web, telemetry, config): no caches, no network,
  a scripted fake LLM (tests/fakes.py) -- they run on a fresh clone;
* ``integration``-marked ones (search, PIs, graph, tool bridge): they import
  ``server`` and assert against the real built caches (see build.py).
  Deselect them with ``pytest -m "not integration"`` when data/ is absent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


import json
import logging
import threading

import pytest

from fakes import FakeOpenAI, text_chunk  # noqa: E402


@pytest.fixture()
def isolated_logs(tmp_path, monkeypatch):
    """Send the JSON log (and its telemetry lines) to a temp dir instead of logs/."""
    import logging_config
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(logging_config, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logging_config, "_configured", False)
    root = logging.getLogger("assistant")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    logging_config.configure_logging()
    yield log_dir


LOCAL_TOOL = {"type": "function", "function": {"name": "search_papers", "description": "d",
                                                "parameters": {"type": "object", "properties": {}}}}


class Harness:
    """Per-test knobs the fake state reads from."""

    def __init__(self):
        self.scripts = [[text_chunk("Hi")]]
        self.clients: list[FakeOpenAI] = []
        self.tool_calls: list[tuple] = []
        self.gate: threading.Event | None = None  # tool blocks until set (for busy tests)

    def client_factory(self, api_key, base_url):
        c = FakeOpenAI(self.scripts)
        self.clients.append(c)
        return c

    def call_tool(self, name, args, remote=None):
        self.tool_calls.append((name, args, remote))
        if self.gate is not None:
            self.gate.wait(5)
        return json.dumps({"ok": name})


@pytest.fixture()
def harness():
    return Harness()


@pytest.fixture()
def state(harness, tmp_path, monkeypatch, isolated_logs):
    """A fake AppState for the web tests: fake LLM, stub corpus callables, temp feedback file."""
    import telemetry
    import web
    from config import get_config

    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.delenv("ECONVERSION_API_KEY", raising=False)  # the deployment .env's name, a fallback
    monkeypatch.setattr(telemetry, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    cfg = get_config()
    return web.AppState(
        cfg=cfg, system_prompt="SYS", n_papers=3, n_pis=2, local_tools=[LOCAL_TOOL],
        call_tool=harness.call_tool, client_factory=harness.client_factory,
        corpus_map=lambda n: {"points": [{"x": 0, "y": 0, "title": "t", "cluster": "c", "color": [1, 2, 3]}] * n,
                              "legend": [{"cluster": "c", "color": [1, 2, 3]}]},
        corpus_map_available=lambda: True,
        collab_graph=lambda: {"nodes": [], "links": []},
        coverage=lambda: {"papers": {"available": True, "count": 3}},
        summarize=lambda: {"turns": 1, "sessions": 1, "error_turns": 0, "avg_latency_ms": 5,
                           "models": {"m": 1}, "providers": {"gwdg": 1},
                           "tools": {"search_papers": {"calls": 1, "errors": 0, "avg_ms": 3}}, "feedback": 0},
    )
