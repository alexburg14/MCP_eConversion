"""The chat agent loop: streamed tool calling over an OpenAI-compatible API.

``run_turn`` is a *synchronous generator* that yields plain-dict events while
the model streams and tools run, so any transport (SSE, a CLI, a test) can
consume it. It deliberately knows nothing about sessions, HTTP or the tool
registry: the caller passes the tool schemas and a ``call_tool`` callable.

Why synchronous: the local tools are CPU-bound (BM25, sentence-transformers,
networkx) and the remote MCP bridge drives its own event loop via
``anyio.run``; both want a plain worker thread, not the server's event loop.

Events (``{"type": ..., ...}``):

    round               {round, max[, final]}
    text_delta          {text}                      answer text fragment
    reasoning_delta     {text}                      model "thinking" (never stored)
    tool_call_start     {id, name, args, round}
    tool_call_end       {id, name, ok, ms, preview}
    done                {answer, elapsed, rounds, usage, tool_calls, tools, error}
    error               same fields as done plus {message, error_type}

Exactly one of ``done`` / ``error`` is the last event. ``done.error`` is None,
``tool_call_limit_reached``, ``tool_calls_fruitless`` or ``cancelled``;
``error`` carries an exception (upstream failure, bad request) together with
the rounds and tool calls that had already run, so the caller can still
account for them.

The tool budget is a budget for searching, not for answering: when it is spent,
or when two rounds in a row produced nothing new, the model is asked once more
without tools to answer from what it has, so the person gets an answer built on
the evidence gathered rather than a note that the limit was hit.
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
REPEATED_NOTE = ("You already made this exact call earlier in this answer and have its result. "
                 "Use it, search for something different, or answer with what you have.")
FAILED_AGAIN_NOTE = ("This exact call already failed earlier in this answer with the same error. "
                     "Change the arguments or use a different tool.")
ANSWER_NOW = ("You have used the tool calls available for this answer. Do not call any more "
              "tools. Answer the question now from the results you already have, cite what "
              "you found, and say plainly what you could not find out.")
# rounds in a row in which every call failed or repeated an earlier one
FRUITLESS_ROUNDS = 2
LIMIT_REACHED_ERROR = "tool_call_limit_reached"
FRUITLESS_ERROR = "tool_calls_fruitless"
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
        # what the model said before each tool call: narration, not the answer
        self.round_texts: list[str] = []
        self.start = time.perf_counter()

    def answer_so_far(self, final: str = "") -> str:
        """The answer is what the model says once it stops calling tools. The
        text between tool calls ("let me check...") is used only when nothing
        else ever came, so a cut-off turn still shows what it had."""
        return final or "\n\n".join(t for t in self.round_texts if t)


class _Round:
    """What one model call produced, filled while its events stream out."""

    def __init__(self) -> None:
        self.text = ""
        self.calls: dict[int, dict] = {}
        self.cancelled = False


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


def _arguments(raw: str) -> dict:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def _error_of(result: str) -> str:
    try:
        return str(json.loads(result).get("error", result))[:300]
    except (json.JSONDecodeError, AttributeError):
        return result[:300]


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
    answer_so_far = p.answer_so_far

    def cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    # A model that repeats a call it already made would keep doing it until the
    # round limit. A repeat is answered rather than run: a successful call with
    # a note that its result is already there, a failed one with its error
    # again, so the model is never told it has a result it does not have.
    outcomes: dict[tuple[str, str], str | None] = {}
    # Some endpoints number their calls from zero in every round. The ids pair
    # each result with its call, in the conversation and in the interface, so
    # they have to be unique for the whole turn.
    used_ids: set[str] = set()
    fruitless = 0
    why_final: str | None = None

    for round_num in range(max_rounds):
        rounds = p.rounds = round_num + 1
        if cancelled():
            yield _done(answer_so_far(), start, rounds - 1, usage, tool_log, tool_meta, "cancelled")
            return
        yield {"type": "round", "round": rounds, "max": max_rounds}
        result = _Round()
        yield from _one_round(client, model, msgs, kwargs, extra_body, base_url, cancelled,
                              usage, p, result)
        if result.cancelled:
            yield _done(answer_so_far(result.text), start, rounds, usage, tool_log, tool_meta,
                        "cancelled")
            return
        if not result.calls:
            log.info("answer_complete", extra={"fields": {
                "model": model, "rounds": rounds,
                "duration_s": round(time.perf_counter() - start, 3),
            }})
            yield _done(answer_so_far(result.text), start, rounds, usage, tool_log, tool_meta, None)
            return
        p.round_texts.append(result.text)

        calls = [result.calls[i] for i in sorted(result.calls)]
        for i, c in enumerate(calls):
            if not c["id"] or c["id"] in used_ids:
                c["id"] = f"call_{rounds}_{i}"
            used_ids.add(c["id"])
        msgs.append({
            "role": "assistant",
            "content": result.text or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ],
        })
        useful = 0
        for c in calls:
            if cancelled():
                yield _done(answer_so_far(), start, rounds, usage, tool_log, tool_meta, "cancelled")
                return
            args = _arguments(c["arguments"])
            yield {"type": "tool_call_start", "id": c["id"], "name": c["name"],
                   "args": args, "round": rounds}
            t0 = time.perf_counter()
            signature = (c["name"], json.dumps(args, sort_keys=True))
            if signature in outcomes:
                earlier = outcomes[signature]
                result_text = json.dumps(
                    {"repeated": True, "note": REPEATED_NOTE} if earlier is None
                    else {"repeated": True, "error": earlier, "note": FAILED_AGAIN_NOTE})
                ok = False
            else:
                result_text = call_tool(c["name"], args)
                if not isinstance(result_text, str):
                    result_text = json.dumps(result_text, ensure_ascii=False)
                ok = not result_text.lstrip().startswith('{"error"')
                outcomes[signature] = None if ok else _error_of(result_text)
                useful += ok
            ms = (time.perf_counter() - t0) * 1000
            tool_meta.append(telemetry.tool_call_meta(c["name"], args, ms, ok))
            tool_log.append(f"`{c['name']}({c['arguments'][:_ARGS_LOG_CHARS]})`")
            yield {"type": "tool_call_end", "id": c["id"], "name": c["name"], "ok": ok,
                   "ms": round(ms), "preview": result_text[:RESULT_PREVIEW_CHARS]}
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": result_text})

        fruitless = 0 if useful else fruitless + 1
        if fruitless >= FRUITLESS_ROUNDS:
            why_final = FRUITLESS_ERROR
            break
    else:
        why_final = LIMIT_REACHED_ERROR

    log.info("answering without tools", extra={"fields": {
        "model": model, "reason": why_final, "rounds": p.rounds}})
    if cancelled():
        yield _done(answer_so_far(), start, p.rounds, usage, tool_log, tool_meta, "cancelled")
        return
    yield {"type": "round", "round": p.rounds + 1, "max": max_rounds, "final": True}
    msgs.append({"role": "user", "content": ANSWER_NOW})
    result = _Round()
    yield from _one_round(client, model, msgs, {**kwargs, "tool_choice": "none"}, extra_body,
                          base_url, cancelled, usage, p, result)
    if result.cancelled:
        yield _done(answer_so_far(result.text), start, p.rounds, usage, tool_log, tool_meta,
                    "cancelled")
        return
    yield _done(answer_so_far(result.text) or LIMIT_REACHED_TEXT, start, p.rounds, usage,
                tool_log, tool_meta, why_final)


def _one_round(client: Any, model: str, msgs: list[dict], kwargs: dict, extra_body: dict | None,
               base_url: str, cancelled: Callable[[], bool], usage: dict, p: _Progress,
               out: _Round) -> Iterator[dict]:
    """One model call, streamed. Text deltas go out as they arrive; the text
    and the tool calls it asked for are left in ``out``."""
    call_start = time.perf_counter()
    stream = _create_stream(client, model, msgs, kwargs, extra_body, base_url)
    text_parts: list[str] = []
    splitter = _ThinkSplitter()
    try:
        for chunk in stream:
            if cancelled():
                out.text = "".join(text_parts)
                out.cancelled = True
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
                _accumulate_tool_delta(out.calls, tc)
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
    out.text = "".join(text_parts)
    log.info("llm_call", extra={"fields": {
        "model": model, "round": p.rounds,
        "duration_s": round(time.perf_counter() - call_start, 3),
        "tool_calls": len(out.calls),
    }})
