"""Exercise deployed HTTP trust boundaries without calling paid providers."""
import json
import threading
from email.message import Message
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from dashboard import Dashboard, make_server
from deployment import AccessPolicy

TOKEN = "unit-test-token-with-at-least-thirty-two-characters"
ORIGIN = "https://fusion.example.com"


@pytest.mark.parametrize("kwargs", [
    {"bind": "0.0.0.0"},
    {"bind": "192.168.1.2"},
    {"proxy_token": TOKEN},
    {"public_origin": ORIGIN},
    {"public_origin": ORIGIN, "proxy_token": "weak"},
    {"public_origin": "http://fusion.example.com", "proxy_token": TOKEN},
    {"public_origin": "https://fusion.example.com/", "proxy_token": TOKEN},
    {"public_origin": "https://user@fusion.example.com", "proxy_token": TOKEN},
    {"public_origin": "https://fusion.example.com?x=y", "proxy_token": TOKEN},
    {"public_origin": "https://fusion.example.com#x", "proxy_token": TOKEN},
    {"public_origin": "https://fusion.example.com:8787", "proxy_token": TOKEN},
    {"public_origin": "https://fusion.example.com:443", "proxy_token": TOKEN},
])
def test_unsafe_deployment_fails_closed(kwargs):
    with pytest.raises(ValueError):
        AccessPolicy(**kwargs)


def test_proxy_secret_is_not_in_config_repr():
    access = AccessPolicy(public_origin=ORIGIN, proxy_token=TOKEN)
    assert TOKEN not in repr(access)
    headers = Message()
    headers["X-Fusion-Proxy-Token"] = TOKEN
    assert access.authenticated(headers)
    headers["X-Fusion-Proxy-Token"] = TOKEN
    assert not access.authenticated(headers)


@pytest.fixture
def deployed(tmp_path):
    app = Dashboard(tmp_path)
    server = make_server(
        app, 0, access=AccessPolicy(public_origin=ORIGIN, proxy_token=TOKEN)
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield app, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def request(base, path, *, method="GET", auth=True, origin=ORIGIN, host="fusion.example.com"):
    headers = {"Host": host, "Origin": origin, "X-Fusion-UI": "1"}
    if auth:
        headers["X-Fusion-Proxy-Token"] = TOKEN
    if method == "POST":
        headers["Content-Type"] = "application/json"
    return Request(base + path, headers=headers, method=method,
                   data=b"{}" if method == "POST" else None)


@pytest.mark.parametrize("method,path", [
    ("GET", "/"), ("GET", "/api/research"), ("GET", "/api/shadow"),
    ("GET", "/api/research/export"), ("HEAD", "/healthz"),
    ("POST", "/api/research"), ("POST", "/api/shadow/start"),
])
def test_unauthenticated_requests_never_reach_app(deployed, method, path):
    app, base = deployed
    with pytest.raises(HTTPError) as error:
        urlopen(request(base, path, method=method, auth=False), timeout=3)
    assert error.value.code == 403
    assert not app.active and not app.research.active and not app.shadow.active


def test_authenticated_reads_and_loopback_liveness(deployed):
    _, base = deployed
    with urlopen(base + "/healthz", timeout=3) as response:
        assert json.load(response) == {"status": "ok"}
    with urlopen(request(base, "/"), timeout=3) as response:
        assert b"Fusion Investigator" in response.read()
    with urlopen(request(base, "/api/research"), timeout=3) as response:
        assert isinstance(json.load(response), dict)
    with urlopen(request(base, "/healthz", method="HEAD"), timeout=3) as response:
        assert response.status == 200


@pytest.mark.parametrize("origin,host", [
    ("https://evil.example", "fusion.example.com"),
    ("http://fusion.example.com", "fusion.example.com"),
    (ORIGIN, "evil.example"),
    (ORIGIN, "fusion.example.com:8787"),
])
def test_proxy_auth_does_not_bypass_csrf_or_host(deployed, origin, host):
    app, base = deployed
    with pytest.raises(HTTPError) as error:
        urlopen(request(base, "/api/research", method="POST", origin=origin, host=host), timeout=3)
    assert error.value.code == 403
    assert not app.research.active


def test_authenticated_post_reaches_only_existing_typed_route(deployed, monkeypatch):
    app, base = deployed
    calls = []
    monkeypatch.setattr(app.research, "start", lambda *args: calls.append(args) or {"ok": True})
    with urlopen(request(base, "/api/research", method="POST"), timeout=3) as response:
        assert response.status == 202
    assert calls == [(None, None, "live", False)]


def test_forwarded_headers_are_not_authentication(deployed):
    _, base = deployed
    req = request(base, "/api/research", auth=False)
    req.add_header("X-Forwarded-For", "127.0.0.1")
    req.add_header("X-Forwarded-Host", "fusion.example.com")
    with pytest.raises(HTTPError) as error:
        urlopen(req, timeout=3)
    assert error.value.code == 403
