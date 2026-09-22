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

from config import Config, Provider

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The OpenRouter choices offered in the model picker. All of them use the same
# (cheapest eligible) model and differ only in how the upstream provider is
# picked; a hard price cap keeps every route cheap.
ROUTE_OPTIONS: tuple[dict[str, str], ...] = (
    {"value": "price", "label": "cheapest",
     "help": "Cheapest eligible model at the lowest price — the default."},
    {"value": "throughput", "label": "fastest",
     "help": "Same model, routed to the provider with the most tokens/second."},
    {"value": "latency", "label": "lowest latency",
     "help": "Same model, routed to the provider with the lowest latency."},
)
ROUTE_VALUES = tuple(route["value"] for route in ROUTE_OPTIONS)


def route_label(value: str) -> str:
    return next((r["label"] for r in ROUTE_OPTIONS if r["value"] == value), value)


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


def _cache_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


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


# ---------------------------------------------------------------------------
# Per-session LLM parameters
# ---------------------------------------------------------------------------
# One source of truth: the keys the UI renders, the values config.toml may set,
# the values a session may override, and how they map onto the request.
PARAM_SPEC: tuple[dict[str, Any], ...] = (
    {"key": "reasoning_effort", "label": "Reasoning effort", "type": "enum",
     "options": ["", "minimal", "low", "medium", "high"],
     "option_labels": {"": "off (provider default)"},
     "help": "Thinking budget for models with reasoning (deepseek). Ignored elsewhere."},
    {"key": "top_p", "label": "Top P", "type": "enum",
     "options": ["", "0.8", "0.9", "0.95", "1.0"],
     "option_labels": {"": "default (1.0)"},
     "help": "Restrict sampling to the most likely tokens."},
    {"key": "max_tokens", "label": "Max output tokens", "type": "number",
     "min": 256, "max": 32768, "step": 256,
     "help": "Cap the answer length. Empty = the model's own default."},
    {"key": "max_tool_rounds", "label": "Tool call limit", "type": "number",
     "min": 1, "max": 100, "step": 1,
     "help": "Max tool-call rounds per turn before the agent gives up and answers with "
             "what it has. Empty = server default (10)."},
    # chosen in the model picker, not in the parameters panel
    {"key": "provider_sort", "label": "Provider routing", "type": "enum", "hidden": True,
     "options": ["price", "throughput", "latency"],
     "option_labels": {"price": "cheapest", "throughput": "fastest tokens/s",
                       "latency": "lowest latency"},
     "help": "OpenRouter picks the upstream provider by price, throughput or latency."},
)
PARAM_KEYS = tuple(spec["key"] for spec in PARAM_SPEC)
# Sent in OpenRouter's extra_body instead of a top-level request field.
EXTRA_BODY_KEYS = ("provider", "reasoning", "reasoning_effort", "verbosity",
                    "web_search_options", "models", "transforms", "route")


def normalize_param(key: str, value: Any) -> Any:
    """Canonical stored form for one parameter (raises on an invalid value).

    Enums and numbers are stored as strings ("" = unset), booleans as bools, so
    the value from config.toml, the session override and the UI all compare equal.
    """
    spec = next((s for s in PARAM_SPEC if s["key"] == key), None)
    if spec is None:
        raise ValueError(f"unknown parameter: {key}")
    if spec["type"] == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"{key} must be true or false")
    if value is None:
        return ""
    text = str(value).strip()
    if spec["type"] == "enum":
        if text not in spec["options"]:
            raise ValueError(f"{key} must be one of {', '.join(o or 'default' for o in spec['options'])}")
        return text
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{key} must be a number") from None
    if not spec["min"] <= number <= spec["max"]:
        raise ValueError(f"{key} must be between {spec['min']} and {spec['max']}")
    return str(int(number))


def default_params(cfg: Config) -> dict[str, Any]:
    """config.toml values for every known parameter, in canonical form."""
    out: dict[str, Any] = {}
    raw = dict(cfg.llm.params or {})
    for spec in PARAM_SPEC:
        key = spec["key"]
        fallback = "" if spec["type"] != "bool" else bool(spec.get("default", False))
        if key not in raw:
            out[key] = fallback
            continue
        try:
            out[key] = normalize_param(key, raw[key])
        except ValueError:  # a typo in config.toml must not break the app
            out[key] = fallback
    return out


def effective_params(cfg: Config, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Defaults + the session's overrides (invalid overrides are ignored)."""
    out = default_params(cfg)
    for key, value in (overrides or {}).items():
        if key not in PARAM_KEYS:
            continue
        try:
            out[key] = normalize_param(key, value)
        except ValueError:
            continue
    return out


def build_extra(cfg: Config, params: dict[str, Any] | None, provider_name: str | None) -> dict[str, Any] | None:
    """Request fields for one turn, or None when nothing is set.

    Sampling parameters go to every provider (they are plain OpenAI fields);
    OpenRouter-only controls (provider routing, reasoning) are sent only on the
    OpenRouter base_url, so a GWDG turn never sees a field its gateway could
    reject. Zero-data-retention routing and parallel tool calls are deliberately
    not part of the panel: ZDR is always requested, the rest stays at the
    provider's default.
    """
    effective = effective_params(cfg, params)
    providers = cfg.providers or {}
    prov = providers.get(provider_name or "")
    openrouter = bool(prov and "openrouter" in prov.base_url)
    extra: dict[str, Any] = {}
    if effective["top_p"] != "":
        extra["top_p"] = float(effective["top_p"])
    if effective["max_tokens"] != "":
        extra["max_tokens"] = int(effective["max_tokens"])
    if not openrouter:
        return extra or None
    # Prompts must not be retained upstream (privacy rule) -> always request ZDR.
    # max_price caps the routing: "fastest"/"lowest latency" may pick a pricier
    # provider, but never one above the same ceiling the model filter uses.
    extra["provider"] = {"sort": effective["provider_sort"], "zdr": True,
                         "data_collection": "deny",
                         "max_price": {"prompt": OPENROUTER_MAX_PRICE_PER_MTOK,
                                       "completion": OPENROUTER_MAX_PRICE_PER_MTOK}}
    if effective["reasoning_effort"]:
        extra["reasoning"] = {"effort": effective["reasoning_effort"]}
    return extra


def cached_cheapest_model(api_key: str) -> str:
    """The cheapest eligible OpenRouter model, from the cache only (no fetch)."""
    if not api_key:
        return ""
    with _openrouter_lock:
        hit = _openrouter_cache.get(_cache_key(api_key))
    if not hit:
        return ""
    cheapest, _all = hit[1]
    return cheapest[0] if cheapest else ""


def split_extra(extra: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split one turn's request fields into (extra_body, plain SDK kwargs)."""
    body = {k: v for k, v in (extra or {}).items() if k in EXTRA_BODY_KEYS}
    rest = {k: v for k, v in (extra or {}).items() if k not in EXTRA_BODY_KEYS}
    return body, rest


def param_payload(cfg: Config, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the UI needs: the field spec, the defaults and the effective values."""
    return {
        "spec": [dict(spec) for spec in PARAM_SPEC],
        "defaults": default_params(cfg),
        "effective": effective_params(cfg, overrides),
    }


def provider_has_key(prov: Provider) -> bool:
    """True when this provider's API key is present in the environment."""
    return bool(provider_api_key(prov.api_key_env))


def default_provider_name(cfg: Config) -> str:
    """Provider a session starts on.

    ``[llm] default_provider`` wins; without it (or without its API key) the
    provider matching the top-level [llm] block is used, then the first one that
    has a key at all, so a missing key never leaves the chat without a model.
    """
    providers = cfg.providers or {}
    wanted = (cfg.llm.default_provider or "").strip()
    if wanted and wanted in providers and provider_has_key(providers[wanted]):
        return wanted
    fallback = next(
        (x for x, p in providers.items() if p.base_url == cfg.llm.base_url),
        next(iter(providers), "default"),
    )
    if fallback in providers and provider_has_key(providers[fallback]):
        return fallback
    return next((x for x, p in providers.items() if provider_has_key(p)), fallback)


def resolve_llm(cfg: Config, provider_name: str | None = None,
                model_name: str | None = None,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provider + model + parameter overrides for a turn.

    Pure: unknown picks fall back to defaults. Returns ``{"provider",
    "base_url", "model", "api_key", "extra"}``; the caller may store ``provider``
    back into the session to normalise an invalid pick.
    """
    providers = cfg.providers or {}
    default_provider = default_provider_name(cfg)
    if not provider_name or provider_name not in providers:
        provider_name = default_provider
    prov = providers.get(provider_name)
    # Agent-loop control, not an LLM request field -- resolved here (not in
    # build_extra) so it never leaks into the OpenAI/OpenRouter request body.
    raw_max_rounds = effective_params(cfg, params)["max_tool_rounds"]
    max_tool_rounds = int(raw_max_rounds) if raw_max_rounds else None
    if prov is None:
        return {"provider": "default", "base_url": cfg.llm.base_url,
                "model": cfg.llm.default_model,
                "api_key": provider_api_key("API_KEY"),
                "extra": build_extra(cfg, params, None),
                "max_tool_rounds": max_tool_rounds}
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
            "api_key": api_key, "extra": build_extra(cfg, params, provider_name),
            "max_tool_rounds": max_tool_rounds}


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
