"""The streamed agent loop, driven by a scripted fake client (no network, no caches)."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent  # noqa: E402
import telemetry  # noqa: E402
from fakes import (FakeOpenAI, reasoning_chunk, text_chunk, tool_chunk,  # noqa: E402
                   usage_chunk)

TOOLS = [{"type": "function", "function": {"name": "search_papers", "description": "d",
                                            "parameters": {"type": "object", "properties": {}}}}]


def run(client, call_tool=None, **kw):
    calls = []

    def _call(name, args):
        calls.append((name, args))
        return call_tool(name, args) if call_tool else json.dumps({"ok": name})

    events = list(agent.run_turn(client, "m", [{"role": "user", "content": "q"}],
                                 system_prompt="sys", tools=TOOLS, call_tool=_call, **kw))
    return events, calls


def types(events):
    return [e["type"] for e in events]


@pytest.fixture(autouse=True)
def _clear_stream_options_memory():
    agent._no_stream_options.clear()


def test_plain_text_streams_deltas_and_requests_usage():
    client = FakeOpenAI([[text_chunk("Hel"), text_chunk("lo"), usage_chunk(10, 2, 12)]])
    events, calls = run(client)
    assert types(events) == ["round", "text_delta", "text_delta", "done"]
    done = events[-1]
    assert done["answer"] == "Hello" and done["error"] is None and done["rounds"] == 1
    assert done["usage"] == {"prompt": 10, "completion": 2, "total": 12}
    assert calls == []
    kw = client.calls[0]
    assert kw["stream"] is True and kw["stream_options"] == {"include_usage": True}
    assert kw["max_tokens"] == agent.MAX_TOKENS and kw["tools"] == TOOLS
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert client.streams[0].closed


def test_fragmented_interleaved_tool_calls_are_assembled_in_index_order():
    round1 = [
        tool_chunk(0, id="c0", name="search_papers", arguments='{"que'),
        tool_chunk(1, id="c1", name="get_pi", arguments='{"name": '),
        tool_chunk(0, arguments='ry": "x"}'),
        tool_chunk(1, arguments='"Bein"}'),
    ]
    client = FakeOpenAI([round1, [text_chunk("done")]])
    events, calls = run(client)
    assert calls == [("search_papers", {"query": "x"}), ("get_pi", {"name": "Bein"})]
    assert types(events) == ["round", "tool_call_start", "tool_call_end",
                             "tool_call_start", "tool_call_end", "round", "text_delta", "done"]
    done = events[-1]
    assert done["rounds"] == 2
    assert done["tool_calls"][0].startswith("`search_papers(")
    assert [t["name"] for t in done["tools"]] == ["search_papers", "get_pi"]
    # the second round saw the assistant tool_calls message and both tool results
    msgs = client.calls[1]["messages"]
    assert msgs[-3]["role"] == "assistant" and [t["id"] for t in msgs[-3]["tool_calls"]] == ["c0", "c1"]
    assert [m["tool_call_id"] for m in msgs[-2:]] == ["c0", "c1"]


def test_missing_index_and_id_get_synthesised_and_bad_json_becomes_empty_args():
    round1 = [
        tool_chunk(None, id="only-first", name="search_papers", arguments="{not json", omit_index=True),
        tool_chunk(None, arguments=" at all", omit_index=True),  # continues the last call
        tool_chunk(None, name="get_pi", arguments="{}", omit_index=True),  # no id at all
    ]
    client = FakeOpenAI([round1, [text_chunk("ok")]])
    events, calls = run(client)
    assert calls == [("search_papers", {}), ("get_pi", {})]
    ids = [e["id"] for e in events if e["type"] == "tool_call_start"]
    assert ids[0] == "only-first" and ids[1]  # synthesised, non-empty
    starts = [e for e in events if e["type"] == "tool_call_start"]
    assert starts[0]["args"] == {}


def test_tool_error_json_is_flagged_not_ok_and_previewed():
    client = FakeOpenAI([[tool_chunk(0, id="c", name="search_papers", arguments="{}")], [text_chunk("x")]])
    events, _ = run(client, call_tool=lambda n, a: json.dumps({"error": "boom"}))
    end = next(e for e in events if e["type"] == "tool_call_end")
    assert end["ok"] is False and end["preview"].startswith('{"error"')
    assert events[-1]["tools"][0]["ok"] is False


def test_round_limit_returns_the_limit_text_and_error():
    rounds = [[tool_chunk(0, id=f"c{i}", name="search_papers", arguments="{}")]
              for i in range(agent.MAX_TOOL_ROUNDS)]
    client = FakeOpenAI(rounds)
    events, calls = run(client)
    done = events[-1]
    assert done["error"] == "tool_call_limit_reached"
    assert done["answer"] == agent.LIMIT_REACHED_TEXT  # no round produced text
    assert done["rounds"] == agent.MAX_TOOL_ROUNDS and len(calls) == agent.MAX_TOOL_ROUNDS


def test_create_exception_becomes_an_error_event_that_keeps_progress():
    client = FakeOpenAI([[tool_chunk(0, id="c", name="search_papers", arguments="{}")],
                         RuntimeError("upstream down")])
    events, calls = run(client)
    err = events[-1]
    assert err["type"] == "error" and err["error_type"] == "RuntimeError" and "upstream down" in err["message"]
    assert agent.friendly_error(_status_error(500, "x")).startswith("The model endpoint failed with HTTP 500")
    assert err["rounds"] == 2 and [t["name"] for t in err["tools"]] == ["search_papers"]
    assert err["answer"].startswith("Error: upstream down")
    assert len(calls) == 1


def _status_error(status: int, message: str) -> openai.APIStatusError:
    resp = httpx.Response(status, request=httpx.Request("POST", "https://x/v1/chat/completions"))
    cls = {400: openai.BadRequestError, 500: openai.InternalServerError}[status]
    return cls(message, response=resp, body=None)


@pytest.mark.parametrize("status,message", [
    (400, "Unrecognized request argument: stream_options"),
    (500, "Internal Server Error"),  # what GWDG actually answers
])
def test_stream_options_rejection_falls_back_and_is_remembered(status, message):
    client = FakeOpenAI([_status_error(status, message), [text_chunk("a")], [text_chunk("b")]])
    run(client, base_url="https://gwdg/v1")
    run(client, base_url="https://gwdg/v1")
    assert "stream_options" in client.calls[0]
    assert "stream_options" not in client.calls[1]
    assert "stream_options" not in client.calls[2]  # second turn: no retry dance
    assert "https://gwdg/v1" in agent._no_stream_options


def test_error_on_the_plain_retry_is_reported():
    client = FakeOpenAI([_status_error(500, "down"), _status_error(500, "still down")])
    events, _ = run(client, base_url="https://gwdg/v1")
    assert events[-1]["type"] == "error" and events[-1]["error_type"] == "InternalServerError"
    assert len(client.calls) == 2


def test_cancel_during_tool_execution_ends_the_turn_with_partial_text():
    cancel = threading.Event()
    round1 = [text_chunk("partial "), tool_chunk(0, id="c", name="search_papers", arguments="{}")]
    client = FakeOpenAI([round1, [text_chunk("never")]])

    def _tool(name, args):
        cancel.set()
        return "{}"

    events, calls = run(client, call_tool=_tool, cancel=cancel)
    done = events[-1]
    assert done["error"] == "cancelled" and done["answer"] == "partial "
    assert len(client.calls) == 1  # no second round after cancellation


def test_cancel_mid_stream_closes_the_stream():
    cancel = threading.Event()
    chunks = [text_chunk("a"), text_chunk("b"), text_chunk("c")]
    client = FakeOpenAI([chunks])

    def _iter():
        events = []
        for ev in agent.run_turn(client, "m", [], system_prompt="s", tools=[], call_tool=lambda n, a: "{}",
                                 cancel=cancel):
            events.append(ev)
            if ev["type"] == "text_delta":
                cancel.set()
        return events

    events = _iter()
    assert events[-1]["error"] == "cancelled" and events[-1]["answer"] == "a"
    assert client.streams[0].closed


def test_reasoning_goes_to_its_own_channel_and_not_into_the_answer():
    chunks = [reasoning_chunk("thinking"), text_chunk("<thi"), text_chunk("nk>more</think>"),
              text_chunk("Answer <b>")]
    client = FakeOpenAI([chunks])
    events, _ = run(client)
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning_delta")
    assert reasoning == "thinkingmore"
    assert events[-1]["answer"] == "Answer <b>"


def test_text_before_a_tool_call_is_kept_in_the_final_answer():
    client = FakeOpenAI([[text_chunk("Let me check."), tool_chunk(0, id="c", name="search_papers", arguments="{}")],
                         [text_chunk("Found it.")]])
    events, _ = run(client)
    assert events[-1]["answer"] == "Let me check.\n\nFound it."


def test_openrouter_extra_provider_goes_to_extra_body():
    client = FakeOpenAI([[text_chunk("x")]])
    run(client, extra={"provider": {"zdr": True}})
    assert client.calls[0]["extra_body"] == {"provider": {"zdr": True}}
    assert "provider" not in client.calls[0]


def test_done_event_feeds_the_privacy_safe_turn_record():
    secret = "SECRET question"
    client = FakeOpenAI([[tool_chunk(0, id="c", name="search_papers", arguments=json.dumps({"query": secret}))],
                         [text_chunk("SECRET answer")]])
    events, _ = run(client)
    done = events[-1]
    record = telemetry.turn_record(
        session="s", turn=1, provider="p", model="m", base_url="u",
        tools_available={"local": 1, "elab": 0, "dt": 0, "total": 1},
        rounds=done["rounds"], tools=done["tools"], prompt=secret, answer=done["answer"],
        usage=done["usage"], latency_ms=done["elapsed"] * 1000, error=done["error"],
    )
    blob = json.dumps(record)
    assert "SECRET" not in blob
    assert record["tools_used"] == 1 and record["tool_rounds"] == 2
