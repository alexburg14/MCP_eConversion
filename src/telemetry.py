"""Usage telemetry for the e-conversion knowledge assistant.

Writes one JSON line per chat turn (``assistant.chat``) and turns those lines
back into the numbers shown in the stats panel.

Privacy rule (user-mandated, 2026-09-21): raw user prompts and assistant
answers are NEVER written to the log. Log records carry only lengths and
SHA-256 prefixes. Raw content is persisted exclusively in the feedback store
(``data/feedback/feedback.jsonl``), and only when the user submits the
feedback form.

Tool arguments are hashed by default (they contain the user's search queries,
i.e. prompt text); set ``LOG_TOOL_ARGS=1`` to log them verbatim for debugging.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from logging_config import get_logger, log_dir

log = get_logger("chat")

_REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = Path(os.environ.get("FEEDBACK_FILE") or
                     (_REPO_ROOT / "data" / "feedback" / "feedback.jsonl"))
LOG_TOOL_ARGS = os.environ.get("LOG_TOOL_ARGS", "0").strip().lower() not in ("", "0", "false", "no")

_startup_logged = False
_feedback_lock = threading.Lock()


def digest(text: str | None, n: int = 12) -> str:
    """Short, stable hash of user content -- what goes into the log instead of text."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def new_session_id() -> str:
    """Anonymous per-browser-session id (no user identity, no IP)."""
    return uuid.uuid4().hex[:8]


def version_info() -> dict:
    """Build provenance: git SHA / build time baked into the image (see Dockerfile)."""
    return {
        "git_sha": os.environ.get("GIT_SHA") or "unknown",
        "build_time": os.environ.get("BUILD_TIME") or "",
    }


def coverage() -> dict:
    """Data-coverage snapshot: per cache availability, count and last-built time."""
    import server  # local import: keeps this module importable without the caches

    paths = getattr(server, "_CACHE_PATHS", {}) or {}
    snap: dict = {}
    for name, state in (server.CACHE_STATUS or {}).items():
        entry = {"available": bool(state.get("available"))}
        if "count" in state:
            entry["count"] = state.get("count")
        path = paths.get(name)
        if path is not None:
            try:
                if Path(path).exists():
                    entry["last_built"] = datetime.fromtimestamp(
                        Path(path).stat().st_mtime, timezone.utc
                    ).isoformat(timespec="seconds")
            except OSError:
                pass
        snap[name] = entry
    return snap


def arg_fields(args) -> dict:
    """Privacy-safe stand-in for raw tool arguments.

    Arguments carry the user's query text (the model derives them from the
    prompt), so by default only a hash, the length and the argument *names* are
    logged. Set ``LOG_TOOL_ARGS=1`` to restore the raw values for debugging.
    """
    try:
        args_json = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        args_json = str(args)
    if LOG_TOOL_ARGS:
        return {"args": args}
    return {
        "args_hash": digest(args_json),
        "args_len": len(args_json),
        "args_keys": sorted(args.keys()) if isinstance(args, dict) else [],
    }


def tool_call_meta(name: str, args: dict, ms: float, ok: bool) -> dict:
    """Privacy-safe metadata for a single tool invocation."""
    return {"name": name, "ok": bool(ok), "ms": round(ms), **arg_fields(args)}


def log_startup(*, tools_local: int, tools_remote: int, providers, models) -> None:
    """One startup line per process."""
    global _startup_logged
    if _startup_logged:
        return
    _startup_logged = True
    log.info("app start", extra={"fields": {
        "tools_local": tools_local,
        "tools_remote": tools_remote,
        "providers": list(providers),
        "models": list(models),
        "pid": os.getpid(),
        "coverage": coverage(),
        **version_info(),
    }})


def turn_record(*, session, turn, provider, model, base_url, tools_available,
                rounds, tools, prompt, answer, usage, latency_ms, error=None) -> dict:
    """Build the privacy-safe per-turn record (hashes and lengths only)."""
    return {
        "session": session,
        "turn": turn,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "tools_available": tools_available,
        "tool_rounds": rounds,
        "tools_used": len(tools or []),
        "tools": tools or [],
        "prompt_len": len(prompt or ""),
        "prompt_hash": digest(prompt),
        "answer_len": len(answer or ""),
        "answer_hash": digest(answer),
        "usage": usage or {},
        "latency_ms": round(latency_ms or 0),
        "error": error,
        **version_info(),
    }


def log_turn(**kwargs) -> dict:
    """Emit one ``assistant.chat`` line and return the record (for the UI)."""
    record = turn_record(**kwargs)
    log.info("chat turn", extra={"fields": record})
    return record


def _log_files() -> list[Path]:
    """Current log plus its rotations, oldest first."""
    directory = log_dir()
    names = ["app.log.%d" % i for i in range(9, 0, -1)] + ["app.log"]
    return [directory / n for n in names if (directory / n).exists()]


def read_turns(limit: int = 20000) -> list[dict]:
    """All ``chat turn`` records from the current + rotated logs, oldest first."""
    out: list[dict] = []
    for path in _log_files():
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("msg") == "chat turn":
                        out.append(rec)
        except OSError:
            continue
    return out[-limit:]


def log_feedback(*, category: str, session: str, model: str,
                 text_len: int, question_len: int, answer_len: int, message_count: int) -> None:
    # The raw conversation lives in the feedback store; this line is metadata only.
    log.info("feedback", extra={"fields": {
        "category": category, "session": session, "model": model,
        "text_len": text_len, "question_len": question_len, "answer_len": answer_len,
        "message_count": message_count, **version_info()}})


def record_feedback(*, question: str, answer: str, model: str, provider: str,
                    session: str, category: str, text: str,
                    messages: list[dict] | None = None) -> dict:
    """Append one feedback record (bug report or general note) to the feedback
    store as JSONL and log a metadata-only line.

    ``question``/``answer`` are the last turn (kept for quick scanning);
    ``messages`` is the full conversation up to that point, so a report is
    reproducible without needing the reporter to re-explain earlier turns.
    Provenance fields (session/model/build) make a report traceable to the
    turn it came from. This is the only place raw conversation text is persisted.
    """
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    ver = version_info()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "provider": provider,
        "session": session,
        "git_sha": ver["git_sha"],
        "build_time": ver["build_time"],
        "question": question,
        "answer": answer,
        "messages": messages or [],
        "category": category,
        "text": text,
    }
    with _feedback_lock, FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_feedback(
        category=category, session=session, model=model, text_len=len(text),
        question_len=len(question or ""), answer_len=len(answer or ""),
        message_count=len(messages or []),
    )
    return entry


def count_feedback() -> int:
    if not FEEDBACK_FILE.exists():
        return 0
    try:
        with FEEDBACK_FILE.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def summarize(turns: list[dict] | None = None) -> dict:
    """Aggregate turns into the numbers shown in the stats panel."""
    turns = read_turns() if turns is None else turns
    tools: dict[str, dict] = {}
    models: dict[str, int] = {}
    providers: dict[str, int] = {}
    errors = 0
    latency = 0.0
    for t in turns:
        key = str(t.get("model"))
        models[key] = models.get(key, 0) + 1
        pkey = str(t.get("provider"))
        providers[pkey] = providers.get(pkey, 0) + 1
        if t.get("error"):
            errors += 1
        try:
            latency += float(t.get("latency_ms") or 0)
        except (TypeError, ValueError):
            pass
        for call in t.get("tools") or []:
            name = str(call.get("name"))
            entry = tools.setdefault(name, {"calls": 0, "errors": 0, "ms_total": 0.0})
            entry["calls"] += 1
            if not call.get("ok", True):
                entry["errors"] += 1
            try:
                entry["ms_total"] += float(call.get("ms") or 0)
            except (TypeError, ValueError):
                pass
    for entry in tools.values():
        entry["avg_ms"] = round(entry.pop("ms_total") / entry["calls"]) if entry["calls"] else 0
    return {
        "turns": len(turns),
        "sessions": len({t.get("session") for t in turns if t.get("session")}),
        "error_turns": errors,
        "avg_latency_ms": round(latency / len(turns)) if turns else 0,
        "models": models,
        "providers": providers,
        "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1]["calls"])),
        "feedback": count_feedback(),
    }
