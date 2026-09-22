"""LLM provider selection and client construction (no UI dependencies).

Resolves which OpenAI-compatible endpoint and model a session talks to, from
``config.toml`` plus the session's picks, and handles the OpenRouter dynamic
model discovery that filters by price, agentic index and tool support.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from config import Config

_REPO_ROOT = Path(__file__).resolve().parent.parent

# OpenRouter: dynamically fetch models that satisfy the account's guardrails
# and cost under 1 EUR / M tokens (prompt AND completion). Falls back to the
# static config list if the fetch fails or no key is configured.
OPENROUTER_MAX_PRICE_PER_MTOK = 1.0  # EUR-equivalent cap (approx. USD 1.0)
_PRICE_PER_TOKEN_CAP = OPENROUTER_MAX_PRICE_PER_MTOK / 1_000_000
_MIN_AGENTIC_INDEX = 35.0

# The two OpenRouter catalogue calls take ~1 s and were made on every turn
# without a pinned model; cache the outcome per key for a while.
_OPENROUTER_CACHE_TTL_S = 600
_openrouter_cache: dict[str, tuple[float, tuple[list[str], list[str]]]] = {}
_openrouter_lock = threading.Lock()


def load_dotenv(env_file: Path | None = None) -> None:
    """Set vars from the repo-root .env if not already in the environment."""
    env_file = env_file or (_REPO_ROOT / ".env")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fetch_openrouter_models_uncached(api_key: str) -> tuple[list[str], list[str]]:
    """Return (cheapest_model_ids, all_model_ids) allowed by the user's guardrails,
    under the price cap, and with an agentic index >= 35.

    The guardrail-filtered list comes from /models/user; benchmark data (agentic
    index) only exists in the full /models list, so we intersect both.
    """
    import requests

    def _price_ok(m) -> bool:
        pricing = m.get("pricing", {}) or {}
        try:
            prompt = float(pricing.get("prompt") or 0)
            completion = float(pricing.get("completion") or 0)
        except (TypeError, ValueError):
            return False
        if prompt < 0 or completion < 0:
            return False
        return prompt <= _PRICE_PER_TOKEN_CAP and completion <= _PRICE_PER_TOKEN_CAP

    def _agentic_ok(m) -> bool:
        b = m.get("benchmarks", {}) or {}
        aa = b.get("artificial_analysis", {}) or {}
        try:
            return float(aa.get("agentic_index") or 0) >= _MIN_AGENTIC_INDEX
        except (TypeError, ValueError):
            return False

    def _tool_calling_ok(m) -> bool:
        # the assistant depends on tool calling for paper search; require the
        # model to advertise tool support
        return "tools" in (m.get("supported_parameters") or [])

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        r_user = requests.get("https://openrouter.ai/api/v1/models/user", headers=headers, timeout=15)
        r_user.raise_for_status()
        user_ids = {m.get("id", "") for m in r_user.json().get("data", [])}
    except Exception:  # noqa: BLE001 -- any failure falls back to the static list
        return [], []

    try:
        r_all = requests.get("https://openrouter.ai/api/v1/models", timeout=15)
        r_all.raise_for_status()
        all_models = r_all.json().get("data", [])
    except Exception:  # noqa: BLE001
        return [], []

    eligible: list[tuple[str, float]] = []  # (model_id, prompt_price_per_token)
    all_ids: list[str] = []
    for m in all_models:
        mid = m.get("id", "")
        if mid not in user_ids:
            continue
        # skip :free endpoints — weak data guarantees (ZDR is enforced per
        # request via provider.zdr, but free hosts are often unstable)
        if ":free" in mid:
            continue
        if not _price_ok(m) or not _agentic_ok(m) or not _tool_calling_ok(m):
            continue
        all_ids.append(mid)
        try:
            prompt = float((m.get("pricing", {}) or {}).get("prompt") or 0)
        except (TypeError, ValueError):
            prompt = float("inf")
        eligible.append((mid, prompt))

    eligible.sort(key=lambda x: x[1])
    cheapest = [mid for mid, _ in eligible]
    return cheapest, all_ids


def fetch_openrouter_models(api_key: str) -> tuple[list[str], list[str]]:
    """Cached wrapper around the two OpenRouter catalogue calls (per key, TTL)."""
    key = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    now = time.monotonic()
    with _openrouter_lock:
        hit = _openrouter_cache.get(key)
        if hit and now - hit[0] < _OPENROUTER_CACHE_TTL_S:
            return hit[1]
    result = _fetch_openrouter_models_uncached(api_key)
    if result != ([], []):  # a failed fetch is retried on the next call
        with _openrouter_lock:
            _openrouter_cache[key] = (now, result)
    return result


# The deployment's compose .env names the GWDG key ECONVERSION_API_KEY and maps
# it to API_KEY inside the container; accept the prefixed name too so the same
# .env file works for a local run.
_KEY_FALLBACKS = {"API_KEY": ("ECONVERSION_API_KEY",)}


def provider_api_key(env_name: str) -> str:
    for name in (env_name, *_KEY_FALLBACKS.get(env_name, ())):
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def default_provider_name(cfg: Config) -> str:
    """The provider whose base_url matches the top-level [llm] block, else the first."""
    providers = cfg.providers or {}
    names = list(providers)
    return next(
        (x for x, p in providers.items() if p.base_url == cfg.llm.base_url),
        names[0] if names else "default",
    )


def resolve_llm(cfg: Config, provider_name: str | None = None,
                model_name: str | None = None) -> dict[str, Any]:
    """Provider + model for a turn. Pure: unknown picks fall back to defaults.

    Returns ``{"provider", "base_url", "model", "api_key", "extra"}``; the caller
    may store ``provider`` back into the session to normalise an invalid pick.
    """
    providers = cfg.providers or {}
    default_provider = default_provider_name(cfg)
    if not provider_name or provider_name not in providers:
        provider_name = default_provider
    prov = providers.get(provider_name)
    if prov is None:
        return {"provider": "default", "base_url": cfg.llm.base_url,
                "model": cfg.llm.default_model,
                "api_key": provider_api_key("API_KEY"), "extra": None}
    models = list(prov.models) or [prov.default_model]
    api_key = provider_api_key(prov.api_key_env)
    if "openrouter" in prov.base_url:
        if model_name in models:
            model = model_name
        else:
            cheapest, _all = fetch_openrouter_models(api_key) if api_key else ([], [])
            model = cheapest[0] if cheapest else prov.default_model
    else:
        model = model_name if model_name in models else prov.default_model
    return {"provider": provider_name, "base_url": prov.base_url, "model": model,
            "api_key": api_key, "extra": None}


# Per-read timeout: a stream that produces no chunk for this long is a hung
# upstream (observed on the GWDG endpoint), not a slow answer.
_READ_TIMEOUT_S = 120.0
# The GWDG endpoint answers a sizeable share of requests with an immediate 500
# (measured 5 of 8 on 2026-09-21); the SDK retries those with exponential
# backoff (0.5 s .. 8 s), so a whole round rarely fails for that reason.
_MAX_RETRIES = 5


def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=_MAX_RETRIES,
                  timeout=httpx.Timeout(_READ_TIMEOUT_S, connect=15.0))
