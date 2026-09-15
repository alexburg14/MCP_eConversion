"""Central configuration for the knowledge-assistant template.

All cluster-specific identity and endpoint settings live in ``config.toml`` at
the repo root, so adapting this server to a different research cluster is a
matter of editing one data file — no Python changes. This module loads that
file once (``tomllib`` is stdlib on 3.11+) and exposes it as a frozen, typed
object via ``get_config()``.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    display_name: str
    description: str
    website: str
    # Optional identity metadata — default empty so a minimal config.toml (e.g. a
    # fork for a different cluster) still loads without these.
    cluster_id: str = ""
    funding_body: str = ""
    host_institutions: tuple[str, ...] = ()
    participating_institutions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Provider:
    """One selectable LLM provider (OpenAI-compatible endpoint)."""
    name: str
    base_url: str
    default_model: str
    models: tuple[str, ...]
    # environment variable holding the API key for this provider
    api_key_env: str = "API_KEY"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    default_model: str
    models: tuple[str, ...]
    providers: dict[str, Provider] | None = None


@dataclass(frozen=True)
class Config:
    cluster: ClusterConfig
    llm: LLMConfig
    providers: dict[str, Provider] = field(default_factory=dict)


def _llm_from(raw_llm: dict, name: str = "llm") -> LLMConfig:
    """Build LLMConfig from a raw [llm] or [llm.providers.X] dict."""
    return LLMConfig(
        base_url=raw_llm["base_url"],
        default_model=raw_llm["default_model"],
        models=tuple(raw_llm["models"]),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache config.toml. Raises if the file or a required key is missing."""
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)
    cluster_raw = dict(raw["cluster"])
    for list_field in ("host_institutions", "participating_institutions"):
        if list_field in cluster_raw:
            cluster_raw[list_field] = tuple(cluster_raw[list_field])
    llm_raw = raw["llm"]
    providers_raw = llm_raw.get("providers", {})
    providers: dict[str, Provider] = {}
    for name, prov in providers_raw.items():
        providers[name] = Provider(
            name=name,
            base_url=prov["base_url"],
            default_model=prov["default_model"],
            models=tuple(prov["models"]),
            api_key_env=prov.get("api_key_env", "API_KEY"),
        )
    return Config(
        cluster=ClusterConfig(**cluster_raw),
        llm=_llm_from(llm_raw, "llm"),
        providers=providers,
    )
