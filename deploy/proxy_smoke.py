"""No-provider integration checks through the real Caddy proxy in CI.

The HTTP listener is bound to runner loopback only; deployment still requires
HTTPS. Dummy credentials are intentionally not usable on a real deployment.
"""
import argparse
import base64
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = "fusion.example.com"
LOGIN = base64.b64encode(b"demo:ci-only-dummy-password").decode("ascii")
WRONG_LOGIN = base64.b64encode(b"demo:wrong-ci-password").decode("ascii")


def check(base, path, status, *, login=None, extra=None, data=None):
    headers = {"Host": HOST, **(extra or {})}
    if login is not None:
        headers["Authorization"] = "Basic " + login
    request = Request(base + path, headers=headers, data=data)
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        assert response.status == status, (path, response.status, status)
        if status == 401:
            assert response.headers.get("WWW-Authenticate", "").startswith("Basic ")
        return response.read()


def wait_for_proxy(base):
    for attempt in range(20):
        try:
            check(base, "/", 401)
            return
        except (URLError, ConnectionError):
            if attempt == 19:
                raise
            time.sleep(.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18788")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    wait_for_proxy(args.base)
    check(args.base, "/", 401, login=WRONG_LOGIN)
    check(args.base, "/api/research", 401, extra={
        "X-Fusion-Proxy-Token": "ci-only-dummy-token-at-least-thirty-two-characters",
    })
    if args.probe:
        received = json.loads(check(args.base, "/", 200, login=LOGIN, extra={
            "X-Fusion-Proxy-Token": "client-spoof-must-be-overwritten",
        }))
        assert received == {
            "authorization_received": False, "proxy_token_valid": True,
            "host": HOST,
        }, received
        print("Caddy upstream probe passed: Authorization stripped; token overwritten.")
        return
    assert b"Fusion Investigator" in check(args.base, "/", 200, login=LOGIN)
    state = json.loads(check(args.base, "/api/research", 200, login=LOGIN, extra={
        "Origin": "https://" + HOST,
        "X-Fusion-Proxy-Token": "client-spoof-must-be-overwritten",
    }))
    assert isinstance(state, dict)
    check(args.base, "/healthz", 401)
    assert json.loads(check(args.base, "/healthz", 200, login=LOGIN)) == {"status": "ok"}
    check(args.base, "/api/research", 403, login=LOGIN, extra={
        "Origin": "https://evil.example", "X-Fusion-UI": "1",
        "Content-Type": "application/json",
    }, data=b"{}")
    print("Caddy/app integration passed: login, UI, reads, token overwrite and CSRF.")


if __name__ == "__main__":
    main()
