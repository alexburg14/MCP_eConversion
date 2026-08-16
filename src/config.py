"""Central configuration for the knowledge-assistant template.

All cluster-specific identity and endpoint settings live in ``config.toml`` at
the repo root, so adapting this server to a different research cluster is a
matter of editing one data file — no Python changes. This module loads that
file once (``tomllib`` is stdlib on 3.11+) and exposes it as a frozen, typed
object via ``get_config()``.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class ClusterConfig:
    name: str
    display_name: str
    description: str
    website: str


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    default_model: str
    models: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    cluster: ClusterConfig
    llm: LLMConfig


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load and cache config.toml. Raises if the file or a required key is missing."""
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        cluster=ClusterConfig(**raw["cluster"]),
        llm=LLMConfig(
            base_url=raw["llm"]["base_url"],
            default_model=raw["llm"]["default_model"],
            models=tuple(raw["llm"]["models"]),
        ),
    )
