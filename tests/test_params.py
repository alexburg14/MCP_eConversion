"""Per-session LLM parameters: defaults, validation, request mapping, HTTP API.

The ``state`` fixture (fake AppState, real config.toml) lives in conftest.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm  # noqa: E402
import web  # noqa: E402
from config import get_config  # noqa: E402


@pytest.fixture()
def cfg():
    return get_config()


@pytest.fixture()
def client(state):
    with TestClient(web.create_app(state, root_path="")) as c:
        yield c


# --- defaults and validation ----------------------------------------------

def test_defaults_cover_every_parameter(cfg):
    defaults = llm.default_params(cfg)
    assert set(defaults) == set(llm.PARAM_KEYS)
    assert defaults["provider_sort"] == "price"
    assert defaults["top_p"] == ""


def test_overrides_win_and_invalid_ones_fall_back(cfg):
    effective = llm.effective_params(cfg, {"top_p": "0.9", "max_tokens": 2048})
    assert effective["top_p"] == "0.9"
    assert effective["max_tokens"] == "2048"          # normalised to its stored form
    assert llm.effective_params(cfg, {"top_p": "9"})["top_p"] == ""
    assert "nonsense" not in llm.effective_params(cfg, {"nonsense": 1})


@pytest.mark.parametrize("key, value", [("top_p", "9"), ("provider_sort", "hot"),
                                        ("max_tokens", "5")])
def test_normalize_rejects_bad_values(key, value):
    with pytest.raises(ValueError):
        llm.normalize_param(key, value)


def test_normalize_rejects_unknown_key():
    with pytest.raises(ValueError):
        llm.normalize_param("does_not_exist", "1")


# --- request mapping ------------------------------------------------------

def test_openrouter_gets_routing_and_privacy(cfg):
    extra = llm.build_extra(cfg, {}, "openrouter")
    assert extra["provider"] == {"sort": "price", "zdr": True, "data_collection": "deny",
                                "max_price": {"prompt": 1.0, "completion": 1.0}}
    assert "reasoning" not in extra                   # "" = not sent


def test_gwdg_never_sees_openrouter_fields(cfg):
    assert llm.build_extra(cfg, {"top_p": "0.9"}, "gwdg") == {"top_p": 0.9}


def test_reasoning_effort_becomes_the_reasoning_map(cfg):
    extra = llm.build_extra(cfg, {"reasoning_effort": "high"}, "openrouter")
    assert extra["reasoning"] == {"effort": "high"}


def test_split_extra_separates_extra_body(cfg):
    extra = llm.build_extra(cfg, {"reasoning_effort": "low"}, "openrouter")
    body, rest = llm.split_extra(extra)
    assert set(body) == {"provider", "reasoning"}
    assert rest == {}


def test_split_extra_handles_none():
    assert llm.split_extra(None) == ({}, {})


# --- HTTP API -------------------------------------------------------------

def test_params_endpoint_stores_and_resets(client):
    assert client.get("/api/session").json()["params"]["top_p"] == ""
    r = client.post("/api/session/params",
                    json={"params": {"top_p": "0.9", "reasoning_effort": "high"}})
    assert r.status_code == 200
    assert r.json()["params"]["top_p"] == "0.9"
    assert client.get("/api/session").json()["params"]["reasoning_effort"] == "high"
    assert client.delete("/api/session/params").json()["params"]["top_p"] == ""


def test_params_endpoint_rejects_bad_input(client):
    assert client.post("/api/session/params", json={"params": {"nope": 1}}).status_code == 400
    assert client.post("/api/session/params", json={"params": {"top_p": "9"}}).status_code == 400
    # ZDR is not a knob any more: the key is rejected outright
    assert client.post("/api/session/params", json={"params": {"zdr_only": False}}).status_code == 400


def test_only_deviations_from_the_defaults_are_stored(client, state):
    client.post("/api/session/params",
                json={"params": {"provider_sort": "price", "top_p": "0.9"}})
    session = next(iter(state.sessions._by_cookie.values()))
    assert session.llm_params == {"top_p": "0.9"}


def test_config_exposes_the_reduced_model_list_and_the_spec(client):
    cfg = client.get("/api/config").json()
    assert cfg["providers"]["gwdg"]["models"] == ["qwen3.8-27b"]
    assert cfg["providers"]["gwdg"]["note"] == "fallback"
    assert cfg["providers"]["openrouter"]["models"] == []
    assert [r["value"] for r in cfg["routes"]] == ["price", "throughput", "latency"]
    assert [p["key"] for p in cfg["parameters"]["spec"]] == list(llm.PARAM_KEYS)
    # routing is chosen in the picker, so it is not rendered in the panel
    hidden = [p["key"] for p in cfg["parameters"]["spec"] if p.get("hidden")]
    assert hidden == ["provider_sort"]


def test_model_picker_route_is_stored_and_capped(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    client.get("/api/session")
    r = client.post("/api/session/model",
                    json={"provider": "openrouter", "model": "", "sort": "throughput"}).json()
    assert r["auto"] is True and r["sort"] == "throughput" and r["route_label"] == "fastest"
    view = client.get("/api/session").json()
    assert view["params"]["provider_sort"] == "throughput"
    # every route keeps the cheapness ceiling
    extra = llm.build_extra(get_config(), view["params"], "openrouter")
    assert extra["provider"]["max_price"] == {"prompt": 1.0, "completion": 1.0}
    assert extra["provider"]["zdr"] is True
    # an unknown route is ignored (falls back to the stored default)
    r2 = client.post("/api/session/model",
                     json={"provider": "openrouter", "model": "", "sort": "warp"}).json()
    assert r2["sort"] == "throughput"
