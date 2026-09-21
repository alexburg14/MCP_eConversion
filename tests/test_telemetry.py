"""Tests for the privacy-safe telemetry module.

The hard rule under test: a log record NEVER contains raw prompt/answer text --
only lengths and hashes. Raw content is written only to the report store, and
only on an explicit user action ("Report problem").
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import telemetry  # noqa: E402
import logging_config  # noqa: E402


SECRET_PROMPT = "SECRET-PROMPT-welche Paper zu Perowskiten?"
SECRET_ANSWER = "SECRET-ANSWER mit DOI 10.1000/xyz"


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point LOG_DIR + REPORT_DIR at a temp dir and reconfigure logging."""
    log_dir = tmp_path / "logs"
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(logging_config, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logging_config, "_configured", False)
    root = logging.getLogger("assistant")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    logging_config.configure_logging()
    monkeypatch.setattr(telemetry, "REPORT_DIR", report_dir)
    monkeypatch.setattr(telemetry, "REPORT_FILE", report_dir / "reports.jsonl")
    monkeypatch.setattr(telemetry, "LOG_TOOL_ARGS", False)
    yield log_dir, report_dir


def _log_text(log_dir: Path) -> str:
    path = log_dir / "app.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _turn(**overrides):
    kwargs = dict(
        session="abcd1234",
        turn=1,
        provider="gwdg",
        model="qwen3.5-122b-a10b",
        base_url="https://chat-ai.academiccloud.de/v1",
        tools_available={"local": 14, "elab": 0, "dt": 0, "total": 14},
        rounds=2,
        tools=[telemetry.tool_call_meta("semantic_search_papers", {"query": SECRET_PROMPT}, 412.0, True)],
        prompt=SECRET_PROMPT,
        answer=SECRET_ANSWER,
        usage={"prompt": 1234, "completion": 210, "total": 1444},
        latency_ms=5230.4,
        error=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_record_contains_no_raw_content():
    record = telemetry.turn_record(**_turn())
    blob = json.dumps(record)
    assert SECRET_PROMPT not in blob
    assert SECRET_ANSWER not in blob
    assert record["prompt_len"] == len(SECRET_PROMPT)
    assert record["prompt_hash"] == telemetry.digest(SECRET_PROMPT)
    assert record["answer_len"] == len(SECRET_ANSWER)
    tool_entry = record["tools"][0]
    assert tool_entry["args_hash"]
    assert SECRET_PROMPT not in json.dumps(tool_entry)  # no raw argument values
    assert tool_entry["args_keys"] == ["query"]  # argument names only


def test_log_turn_writes_one_json_line_without_content(isolated):
    log_dir, _ = isolated
    record = telemetry.log_turn(**_turn())
    text = _log_text(log_dir)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["msg"] == "chat turn"
    assert SECRET_PROMPT not in text and SECRET_ANSWER not in text
    for field in ("model", "provider", "latency_ms", "usage", "tools", "git_sha",
                  "prompt_hash", "prompt_len", "answer_hash", "answer_len", "tools_available"):
        assert field in parsed, field
    assert parsed["latency_ms"] == 5230
    assert parsed["git_sha"] == record["git_sha"]


def test_tool_meta_hashes_args_by_default(isolated):
    meta = telemetry.tool_call_meta("search_papers", {"query": SECRET_PROMPT}, 12.34, True)
    assert "query" not in meta and SECRET_PROMPT not in json.dumps(meta)
    assert meta["args_hash"] == telemetry.digest(json.dumps({"query": SECRET_PROMPT}, sort_keys=True))
    assert meta["args_len"] > 0
    assert meta["ms"] == 12


def test_write_report_stores_raw_content_and_keeps_log_clean(isolated):
    log_dir, report_dir = isolated
    report_id = telemetry.write_report(
        session="abcd1234", turn=1, comment="cites a paper that does not exist",
        prompt=SECRET_PROMPT, answer=SECRET_ANSWER,
        tools=[telemetry.tool_call_meta("search_papers", {"query": SECRET_PROMPT}, 5.0, True)],
        provider="gwdg", model="qwen3.5-122b-a10b", base_url="https://chat-ai.academiccloud.de/v1",
    )
    report_file = report_dir / "reports.jsonl"
    assert report_file.exists()
    stored = json.loads(report_file.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert stored["report_id"] == report_id
    assert stored["prompt"] == SECRET_PROMPT and stored["answer"] == SECRET_ANSWER
    assert stored["comment"] == "cites a paper that does not exist"
    assert "git_sha" in stored and "coverage" in stored

    log_text = _log_text(log_dir)
    assert SECRET_PROMPT not in log_text and SECRET_ANSWER not in log_text
    parsed = json.loads([ln for ln in log_text.splitlines() if ln.strip()][-1])
    assert parsed["msg"] == "problem reported"
    assert parsed["report_id"] == report_id
    assert parsed["comment_len"] == len("cites a paper that does not exist")


def test_summarize_aggregates_turns(isolated):
    telemetry.log_turn(**_turn(turn=1, latency_ms=1000))
    telemetry.log_turn(**_turn(turn=2, latency_ms=3000, error="tool_call_limit_reached",
                               tools=[telemetry.tool_call_meta("search_papers", {"query": "x"}, 100.0, False)]))
    summary = telemetry.summarize()
    assert summary["turns"] == 2
    assert summary["sessions"] == 1
    assert summary["error_turns"] == 1
    assert summary["avg_latency_ms"] == 2000
    assert summary["models"]["qwen3.5-122b-a10b"] == 2
    assert summary["tools"]["semantic_search_papers"]["calls"] == 1
    assert summary["tools"]["search_papers"]["errors"] == 1
    assert summary["reports"] == 0


def test_log_startup_logs_only_once(isolated):
    log_dir, _ = isolated
    telemetry._startup_logged = False
    telemetry.log_startup(tools_local=14, tools_remote=0, providers=["gwdg"], models=["m"])
    telemetry.log_startup(tools_local=14, tools_remote=0, providers=["gwdg"], models=["m"])
    starts = [ln for ln in _log_text(log_dir).splitlines() if '"app start"' in ln]
    assert len(starts) == 1
    assert json.loads(starts[0])["tools_local"] == 14


def test_arg_fields_hashes_values(isolated):
    """The pre-existing tool log must not carry raw argument values."""
    fields = telemetry.arg_fields({"query": SECRET_PROMPT, "limit": 5})
    assert fields["args_keys"] == ["limit", "query"]
    assert SECRET_PROMPT not in json.dumps(fields)
    assert fields["args_len"] > 0

