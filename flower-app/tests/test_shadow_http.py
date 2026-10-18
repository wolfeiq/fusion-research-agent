"""Shadow dashboard routes retain the same-origin boundary; no physics is run."""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dashboard import make_server
from fusion_agent.research.shadow import validate_config

IDENTIFIER = "a" * 32
CONFIG = {"change": "td3_actor_lr", "value": .00005, "seeds": [17], "rounds": 1}


class ShadowStub:
    """Observe route dispatch without constructing an RL runtime or subprocess."""

    def __init__(self):
        self.calls = []
        self.run = {"id": None, "status": "idle", "events": [], "comparison": None}

    def snapshot(self):
        return {"active": self.run["status"] == "running", "run": copy.deepcopy(self.run),
                "runtime": {"available": True, "reason": "test double"}, "catalog": {}}

    def prepare(self, body):
        config = validate_config(body)
        self.calls.append(("prepare", copy.deepcopy(config)))
        self.run.update(id=IDENTIFIER, status="frozen", config=config)
        return self.snapshot()

    def start(self, identifier):
        if identifier != IDENTIFIER or self.run["status"] != "frozen":
            raise ValueError("Select the current frozen trial")
        self.calls.append(("start", identifier))
        self.run["status"] = "running"
        return self.snapshot()

    def export(self, identifier):
        if identifier != IDENTIFIER or self.run["id"] is None:
            raise ValueError("Unknown shadow trial")
        return copy.deepcopy(self.run)


@pytest.fixture
def http_shadow():
    shadow = ShadowStub()

    def forbidden(*_args, **_kwargs):
        pytest.fail("A shadow HTTP request reached the existing Flower experiment")

    app = SimpleNamespace(shadow=shadow, start=forbidden, active=False)
    server = make_server(app, 0)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    worker.start()
    try:
        yield app, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def post(base, path, body=None, *, headers=None, raw=None):
    selected = {"Content-Type": "application/json", "X-Fusion-UI": "1", "Origin": base}
    selected.update(headers or {})
    selected = {key: value for key, value in selected.items() if value is not None}
    data = raw if raw is not None else json.dumps(body).encode()
    return urlopen(Request(base + path, method="POST", headers=selected, data=data), timeout=3)


def test_shadow_prepare_and_start_are_separate_explicit_requests(http_shadow):
    app, base = http_shadow
    with urlopen(base + "/api/shadow", timeout=3) as response:
        assert response.status == 200
        assert json.load(response)["run"]["status"] == "idle"
        assert response.headers["Cache-Control"] == "no-store"
        assert "Access-Control-Allow-Origin" not in response.headers
    assert app.shadow.calls == []
    with post(base, "/api/shadow/prepare", CONFIG) as response:
        assert response.status == 201
        frozen = json.load(response)
    assert frozen["run"]["status"] == "frozen" and not frozen["active"]
    assert app.shadow.calls == [("prepare", validate_config(CONFIG))]
    with urlopen(base + "/api/shadow/export?id=" + IDENTIFIER, timeout=3) as response:
        assert json.load(response) == frozen["run"]
    assert len(app.shadow.calls) == 1
    with post(base, "/api/shadow/start", {"id": IDENTIFIER}) as response:
        assert response.status == 202
        assert json.load(response)["active"] is True
    assert app.shadow.calls[-1] == ("start", IDENTIFIER)
    assert not app.active


@pytest.mark.parametrize("path", ["/api/shadow/prepare", "/api/shadow/start"])
@pytest.mark.parametrize("headers", [
    {"Origin": None}, {"Origin": "https://attacker.example"},
    {"Origin": "null"}, {"X-Fusion-UI": None}, {"X-Fusion-UI": "0"},
    {"Content-Type": "text/plain"}, {"Host": "attacker.example"},
])
def test_shadow_mutations_require_same_origin_and_ui_header(http_shadow, path, headers):
    app, base = http_shadow
    # Empty body avoids a Windows connection reset while rejecting headers.
    with pytest.raises(HTTPError) as error:
        post(base, path, headers=headers, raw=b"")
    assert error.value.code == 403
    assert app.shadow.calls == []


@pytest.mark.parametrize("path", ["/api/shadow", "/api/shadow/export?id=" + IDENTIFIER])
def test_shadow_read_routes_reject_untrusted_host(http_shadow, path):
    app, base = http_shadow
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + path, headers={"Host": "attacker.example"}), timeout=3)
    assert error.value.code == 403
    assert app.shadow.calls == []


@pytest.mark.parametrize("body", [
    {}, {"id": IDENTIFIER, "value": .00001}, {"id": IDENTIFIER, "command": "anything"},
    {"id": "../outside"}, {"id": None}, {"id": [IDENTIFIER]},
])
def test_shadow_start_accepts_only_current_frozen_id(http_shadow, body):
    app, base = http_shadow
    with pytest.raises(HTTPError) as error:
        post(base, "/api/shadow/start", body)
    assert error.value.code == 400
    assert app.shadow.calls == []


@pytest.mark.parametrize("raw", [b"", b"{", b"null", b"[]", b'"a"'])
@pytest.mark.parametrize("path", ["/api/shadow/prepare", "/api/shadow/start"])
def test_shadow_mutations_reject_missing_malformed_and_nonobject_json(http_shadow, path, raw):
    app, base = http_shadow
    with pytest.raises(HTTPError) as error:
        post(base, path, raw=raw)
    assert error.value.code == 400
    assert app.shadow.calls == []


@pytest.mark.parametrize("patch", [
    {"command": "anything"}, {"output": "../existing-run"}, {"value": True},
    {"value": .0001}, {"seeds": [17, 17]}, {"rounds": 3},
])
def test_shadow_prepare_preserves_typed_service_validation(http_shadow, patch):
    app, base = http_shadow
    with pytest.raises(HTTPError) as error:
        post(base, "/api/shadow/prepare", {**CONFIG, **patch})
    assert error.value.code == 400
    assert app.shadow.calls == []


@pytest.mark.parametrize("query", ["", "?id=..%2Foutside", "?id=" + "b" * 32])
def test_shadow_export_requires_a_known_trial(http_shadow, query):
    app, base = http_shadow
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/shadow/export" + query, timeout=3)
    assert error.value.code == 404
    assert app.shadow.calls == []


def test_shadow_duplicate_start_is_a_visible_error(http_shadow):
    app, base = http_shadow
    with post(base, "/api/shadow/prepare", CONFIG):
        pass
    with post(base, "/api/shadow/start", {"id": IDENTIFIER}):
        pass
    with pytest.raises(HTTPError) as error:
        post(base, "/api/shadow/start", {"id": IDENTIFIER})
    assert error.value.code == 400
    assert len([call for call in app.shadow.calls if call[0] == "start"]) == 1
