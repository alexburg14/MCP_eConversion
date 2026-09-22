"""The chat agent loop: streamed tool calling over an OpenAI-compatible API.

``run_turn`` is a *synchronous generator* that yields plain-dict events while
the model streams and tools run, so any transport (SSE, a CLI, a test) can
consume it. It deliberately knows nothing about sessions, HTTP or the tool
registry: the caller passes the tool schemas and a ``call_tool`` callable.

Why synchronous: the local tools are CPU-bound (BM25, sentence-transformers,
networkx) and the remote MCP bridge drives its own event loop via
``anyio.run``; both want a plain worker thread, not the server's event loop.

Events (``{"type": ..., ...}``):

    round               {round, max}
    text_delta          {text}                      answer text fragment
    reasoning_delta     {text}                      model "thinking" (never stored)
    tool_call_start     {id, name, args, round}
    tool_call_end       {id, name, ok, ms, preview}
    done                {answer, elapsed, rounds, usage, tool_calls, tools, error}
    error               same fields as done plus {message, error_type}

Exactly one of ``done`` / ``error`` is the last event. ``done.error`` is None,
``tool_call_limit_reached`` or ``cancelled``; ``error`` carries an exception
(upstream failure, bad request) together with the rounds and tool calls that
had already run, so the caller can still account for them.
"""
from __future__ import annotations

import json
import threading
import time

import llm
from typing import Any, Callable, Iterator

import openai

import telemetry
from logging_config import get_logger

log = get_logger("agent")

MAX_TOOL_ROUNDS = 10
# Headroom for long list answers (e.g. "all papers by X" can be 30-40 items);
# 2048 truncated those mid-list.
MAX_TOKENS = 8192
LIMIT_REACHED_TEXT = "Tool-call limit reached without a final answer — try rephrasing the question."
RESULT_PREVIEW_CHARS = 200
_ARGS_LOG_CHARS = 80

# Endpoints that rejected ``stream_options`` (usage-in-stream) with a 400; we
# retry without it and remember so every later call skips the failed attempt.
_no_stream_options: set[str] = set()
_no_stream_options_lock = threading.Lock()

_THINK_OPEN, _THINK_CLOSE = "<think>", "</think>"


class _ThinkSplitter:
    """Route inline ``<think>…</think>`` blocks to the reasoning channel.

    Models served without a reasoning parser emit their thinking inline in
    ``content``. Tags can be split across stream chunks, so a suffix that could
    be the start of a tag is held back until the next chunk decides.
    """

    def __init__(self) -> None:
        self.in_think = False
        self._held = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        buf = self._held + text
        self._held = ""
        out: list[tuple[str, str]] = []
        while buf:
            tag = _THINK_CLOSE if self.in_think else _THINK_OPEN
            kind = "reasoning" if self.in_think else "text"
            pos = buf.find(tag)
            if pos >= 0:
                if pos:
                    out.append((kind, buf[:pos]))
                buf = buf[pos + len(tag):]
                self.in_think = not self.in_think
                continue
            # No full tag: hold back a trailing partial tag (e.g. "<thi"), emit the rest.
            hold = 0
            for n in range(min(len(tag) - 1, len(buf)), 0, -1):
                if tag.startswith(buf[-n:]):
                    hold = n
                    break
            if len(buf) - hold:
                out.append((kind, buf[:len(buf) - hold]))
            self._held = buf[len(buf) - hold:] if hold else ""
            buf = ""
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._held:
            return []
        out = [("reasoning" if self.in_think else "text", self._held)]
        self._held = ""
        return out


def _create_stream(client: Any, model: str, msgs: list[dict], kwargs: dict,
                   extra_body: dict | None, base_url: str):
    params = dict(model=model, max_tokens=MAX_TOKENS, messages=msgs,
                  extra_body=extra_body, stream=True, **kwargs)
    with _no_stream_options_lock:
        try_usage = base_url not in _no_stream_options
    if try_usage:
        try:
            return client.chat.completions.create(stream_options={"include_usage": True}, **params)
        except openai.APIStatusError as exc:
            # GWDG answers stream_options with a 500, others with a 400 naming the
            # field: any status error on this first attempt is treated as
            # "unsupported" and the call is retried plainly. A real outage then
            # fails on the retry and propagates from there.
            with _no_stream_options_lock:
                _no_stream_options.add(base_url)
            log.warning("stream_options rejected; retrying without usage", extra={"fields": {
                "base_url": base_url, "status": getattr(exc, "status_code", None)}})
    return client.chat.completions.create(**params)


def _accumulate_tool_delta(acc: dict[int, dict], tc: Any) -> None:
    """Merge one streamed tool-call fragment into the per-index accumulator.

    Providers differ: some send the whole call in one chunk, some fragment the
    arguments over many, some omit ``index`` (a fragment with an ``id`` opens a
    new call, one without continues the last) or ``id`` (synthesised later).
    """
    idx = getattr(tc, "index", None)
    fn = getattr(tc, "function", None)
    if idx is None:
        # Without an index, a fragment that carries an id or a name opens a new
        # call (names are never fragmented); anything else continues the last.
        opens = bool(getattr(tc, "id", None) or (fn is not None and getattr(fn, "name", None)))
        idx = len(acc) if opens or not acc else max(acc)
    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    if getattr(tc, "id", None):
        slot["id"] = tc.id
    if fn is not None:
        if getattr(fn, "name", None):
            slot["name"] = fn.name
        if getattr(fn, "arguments", None):
            slot["arguments"] += fn.arguments


def _add_usage(totals: dict, usage: Any) -> None:
    if usage is None:
        return
    totals["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
    totals["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
    totals["total"] += int(getattr(usage, "total_tokens", 0) or 0)


def _done(answer: str, start: float, rounds: int, usage: dict, tool_log: list[str],
          tool_meta: list[dict], error: str | None) -> dict:
    return {
        "type": "done", "answer": answer, "elapsed": time.perf_counter() - start,
        "rounds": rounds, "usage": usage, "tool_calls": tool_log, "tools": tool_meta,
        "error": error,
    }


class _Progress:
    """Accounting shared between the loop and its error handler."""

    def __init__(self) -> None:
        self.tool_log: list[str] = []
        self.tool_meta: list[dict] = []
        self.usage = {"prompt": 0, "completion": 0, "total": 0}
        self.rounds = 0
        self.round_texts: list[str] = []
        self.start = time.perf_counter()

    def answer_so_far(self, partial: str = "") -> str:
        return "\n\n".join(t for t in [*self.round_texts, partial] if t)


def run_turn(client: Any, model: str, messages: list[dict], *, system_prompt: str,
             tools: list[dict], call_tool: Callable[[str, dict], str],
             extra: dict | None = None, cancel: threading.Event | None = None,
             base_url: str = "", max_rounds: int | None = None) -> Iterator[dict]:
    """Stream one assistant turn, running tool calls between model rounds.

    ``max_rounds`` overrides ``MAX_TOOL_ROUNDS`` for this turn (a session's
    "Tool call limit" parameter); None uses the module default.

    Never raises: an exception becomes the terminal ``error`` event.
    """
    p = _Progress()
    try:
        yield from _run_rounds(client, model, messages, system_prompt, tools, call_tool,
                               extra, cancel, base_url, p, max_rounds or MAX_TOOL_ROUNDS)
    except Exception as exc:  # noqa: BLE001 -- reported to the caller as an event
        log.error("turn failed", exc_info=True, extra={"fields": {"model": model, "round": p.rounds}})
        message = friendly_error(exc)
        ev = _done(p.answer_so_far() or f"Error: {message}", p.start, p.rounds, p.usage,
                   p.tool_log, p.tool_meta, type(exc).__name__)
        ev.update({"type": "error", "message": message, "error_type": type(exc).__name__})
        yield ev


def friendly_error(exc: Exception) -> str:
    """Short, actionable text for the chat; the traceback stays in the log."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, openai.AuthenticationError):
        return "The model endpoint rejected the API key (401) — it may have expired."
    if isinstance(exc, openai.RateLimitError):
        return "The model endpoint is rate-limiting requests (429) — try again in a moment."
    if isinstance(exc, openai.APIStatusError) and status and status >= 500:
        return (f"The model endpoint failed with HTTP {status} even after retries — "
                "it is probably overloaded; try again or pick another model.")
    if isinstance(exc, openai.APITimeoutError):
        return "The model endpoint stopped responding (timeout) — try again or pick another model."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach the model endpoint — check the network or try again."
    text = str(exc).strip() or type(exc).__name__
    return text[:300]


def _run_rounds(client: Any, model: str, messages: list[dict], system_prompt: str,
                tools: list[dict], call_tool: Callable[[str, dict], str], extra: dict | None,
                cancel: threading.Event | None, base_url: str, p: _Progress,
                max_rounds: int) -> Iterator[dict]:
    msgs = [{"role": "system", "content": system_prompt}] + list(messages)
    kwargs: dict[str, Any] = {"tools": tools} if tools else {}
    extra_body: dict | None = None
    if extra:
        # OpenRouter-only fields (provider routing, reasoning, verbosity, ...) must
        # go in extra_body; the OpenAI SDK rejects unknown top-level kwargs. The
        # rest (temperature, top_p, max_tokens, parallel_tool_calls) is plain SDK
        # surface and goes in as a normal argument. See llm.split_extra.
        body, rest = llm.split_extra(extra)
        if body:
            extra_body = body
        kwargs.update(rest)

    tool_log, tool_meta, usage, start = p.tool_log, p.tool_meta, p.usage, p.start
    # Text the user has seen so far, one entry per round: the stored answer must
    # match what was streamed, including any preamble before a tool call.
    round_texts = p.round_texts
    answer_so_far = p.answer_so_far

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    for round_num in range(max_rounds):
        rounds = p.rounds = round_num + 1
        if cancelled():
            yield _done(answer_so_far(), start, rounds - 1, usage, tool_log, tool_meta, "cancelled")
            return
        yield {"type": "round", "round": rounds, "max": max_rounds}
        call_start = time.perf_counter()
        stream = _create_stream(client, model, msgs, kwargs, extra_body, base_url)
        text_parts: list[str] = []
        acc: dict[int, dict] = {}
        splitter = _ThinkSplitter()
        try:
            for chunk in stream:
                if cancelled():
                    yield _done(answer_so_far("".join(text_parts)), start, rounds, usage,
                                tool_log, tool_meta, "cancelled")
                    return
                _add_usage(usage, getattr(chunk, "usage", None))
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                reasoning = (getattr(delta, "reasoning_content", None)
                             or getattr(delta, "reasoning", None))
                if reasoning:
                    yield {"type": "reasoning_delta", "text": reasoning}
                content = getattr(delta, "content", None)
                if content:
                    for kind, piece in splitter.feed(content):
                        if kind == "text":
                            text_parts.append(piece)
                            yield {"type": "text_delta", "text": piece}
                        else:
                            yield {"type": "reasoning_delta", "text": piece}
                for tc in getattr(delta, "tool_calls", None) or []:
                    _accumulate_tool_delta(acc, tc)
            for kind, piece in splitter.flush():
                if kind == "text":
                    text_parts.append(piece)
                    yield {"type": "text_delta", "text": piece}
                else:
                    yield {"type": "reasoning_delta", "text": piece}
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        log.info("llm_call", extra={"fields": {
            "model": model, "round": round_num,
            "duration_s": round(time.perf_counter() - call_start, 3),
        }})

        content = "".join(text_parts)
        round_texts.append(content)
        if not acc:
            log.info("answer_complete", extra={"fields": {
                "model": model, "rounds": rounds,
                "duration_s": round(time.perf_counter() - start, 3),
            }})
            yield _done(answer_so_far(), start, rounds, usage, tool_log, tool_meta, None)
            return

        calls = [acc[i] for i in sorted(acc)]
        for i, c in enumerate(calls):
            if not c["id"]:
                c["id"] = f"call_{rounds}_{i}"
        msgs.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ],
        })
        for c in calls:
            if cancelled():
                yield _done(answer_so_far(), start, rounds, usage, tool_log, tool_meta, "cancelled")
                return
            try:
                args = json.loads(c["arguments"] or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool_call_start", "id": c["id"], "name": c["name"],
                   "args": args, "round": rounds}
            t0 = time.perf_counter()
            result = call_tool(c["name"], args)
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            ms = (time.perf_counter() - t0) * 1000
            ok = not result.lstrip().startswith('{"error"')
            tool_meta.append(telemetry.tool_call_meta(c["name"], args, ms, ok))
            tool_log.append(f"`{c['name']}({c['arguments'][:_ARGS_LOG_CHARS]})`")
            yield {"type": "tool_call_end", "id": c["id"], "name": c["name"], "ok": ok,
                   "ms": round(ms), "preview": result[:RESULT_PREVIEW_CHARS]}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result})

    yield _done(answer_so_far(LIMIT_REACHED_TEXT), start, rounds, usage, tool_log, tool_meta,
                "tool_call_limit_reached")
