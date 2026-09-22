"""Session store and cookie/identity dependencies (hermetic: no caches, no LLM)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import auth  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_store_creates_gets_and_expires_sessions():
    clock = Clock()
    store = auth.SessionStore(ttl_s=100, clock=clock)
    s = store.create()
    assert store.get(s.cookie_id) is s
    assert store.get("nope") is None and store.get(None) is None
    clock.t += 99
    assert store.get(s.cookie_id) is s  # touching keeps it alive
    clock.t += 101
    assert store.get(s.cookie_id) is None
    assert len(store) == 0


def test_store_evicts_least_recently_seen_beyond_capacity():
    clock = Clock()
    store = auth.SessionStore(ttl_s=10_000, max_sessions=2, clock=clock)
    a = store.create(); clock.t += 1
    b = store.create(); clock.t += 1
    store.get(a.cookie_id); clock.t += 1  # a is now more recent than b
    c = store.create()
    assert store.get(b.cookie_id) is None
    assert store.get(a.cookie_id) is a and store.get(c.cookie_id) is c


def test_reset_clears_history_and_mints_a_new_id_but_keeps_tokens():
    store = auth.SessionStore()
    s = store.create()
    s.messages.append({"role": "user", "content": "x"})
    s.turns = 3
    s.elab_token = "tok"
    s.model_name = "m"
    old_id = s.id
    store.reset(s)
    assert s.messages == [] and s.turns == 0
    assert s.id != old_id
    assert s.elab_token == "tok" and s.model_name == "m"
    assert store.get(s.cookie_id) is s


def test_remote_schema_cache_freshness():
    s = auth.SessionStore().create()
    assert not s.remote_schemas_fresh()
    s.remote_schemas, s.remote_schemas_at = [], 100.0
    assert s.remote_schemas_fresh(now=100.0 + auth.REMOTE_SCHEMAS_TTL_S - 1)
    assert not s.remote_schemas_fresh(now=100.0 + auth.REMOTE_SCHEMAS_TTL_S + 1)
    s.invalidate_remote_schemas()
    assert s.remote_schemas is None


@pytest.fixture()
def app():
    app = FastAPI()
    app.state.sessions = auth.SessionStore()

    @app.get("/whoami")
    def whoami(session: auth.Session = Depends(auth.get_session)):
        return {"id": session.id, "subject": session.identity.subject}

    @app.get("/strict")
    def strict(session: auth.Session = Depends(auth.require_session)):
        return {"id": session.id}

    return app


def test_first_request_sets_cookie_and_second_reuses_the_session(app):
    with TestClient(app) as client:
        r1 = client.get("/whoami")
        assert r1.status_code == 200
        cookie = r1.headers["set-cookie"]
        assert auth.COOKIE_NAME in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
        assert "Secure" not in cookie  # plain http test client
        r2 = client.get("/whoami")
        assert r2.json()["id"] == r1.json()["id"]
        assert "set-cookie" not in r2.headers


def test_unknown_cookie_gets_a_fresh_session(app):
    with TestClient(app) as client:
        r = client.get("/whoami", cookies={auth.COOKIE_NAME: "stale"})
        assert r.status_code == 200 and "set-cookie" in r.headers


def test_require_session_rejects_without_cookie_but_accepts_after_first_contact(app):
    with TestClient(app) as client:
        assert client.get("/strict").status_code == 401
        sid = client.get("/whoami").json()["id"]
        assert client.get("/strict").json()["id"] == sid


def test_cookie_is_scoped_to_the_root_path_and_secure_behind_https_proxy(app):
    with TestClient(app, root_path="/nomad-oasis/api/everse") as client:
        r = client.get("/whoami", headers={"x-forwarded-proto": "https"})
        cookie = r.headers["set-cookie"]
        assert "Path=/nomad-oasis/api/everse" in cookie and "Secure" in cookie


def test_proxy_header_becomes_the_identity(app, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_HEADER", "X-Forwarded-User")
    with TestClient(app) as client:
        r = client.get("/whoami", headers={"X-Forwarded-User": "alice"})
        assert r.json()["subject"] == "alice"
        assert client.get("/whoami").json()["subject"] == "alice"  # sticks to the session
