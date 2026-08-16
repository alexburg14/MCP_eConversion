"""Structured (JSON-lines) logging for the assistant.

One JSON object per line, emitted to stderr and to a rotated ``logs/app.log``.
JSON so later usage/analytics tooling can read the log with a parser instead of
regexes. ``configure_logging()`` is idempotent and safe to call from any entry
point; ``get_logger(name)`` returns a logger under a shared namespace.

Attach structured fields with ``log.info("msg", extra={"fields": {...}})`` — they
are merged into the JSON line alongside the standard ts/level/logger/msg keys.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_NAMESPACE = "assistant"
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach stderr + rotating-file handlers to the shared namespace. Idempotent."""
    global _configured
    if _configured:
        return
    _LOG_DIR.mkdir(exist_ok=True)
    root = logging.getLogger(_NAMESPACE)
    root.setLevel(level)
    formatter = _JsonFormatter()

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file = logging.handlers.RotatingFileHandler(
        _LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(formatter)
    root.addHandler(file)

    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger under the shared namespace, e.g. get_logger('server') -> 'assistant.server'."""
    return logging.getLogger(f"{_NAMESPACE}.{name}")
