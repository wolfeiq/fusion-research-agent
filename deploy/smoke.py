"""Container boundary smoke. Requires no real credentials or provider calls."""
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

base = os.environ.get("SMOKE_BASE_URL", "http://127.0.0.1:18787")
token = os.environ["FUSION_PROXY_TOKEN"]
headers = {"Host": "fusion.example.com", "X-Fusion-Proxy-Token": token}


def check(path, status, *, request_headers=None, data=None):
    req = Request(base + path, headers=request_headers or {}, data=data)
    try:
        response = urlopen(req, timeout=5)
    except HTTPError as exc:
        response = exc
    with response:
        assert response.status == status, (path, response.status, status)
        return response.read()


check("/", 403)
check("/api/research", 403, request_headers={"Host": "fusion.example.com"})
assert b"Fusion Investigator" in check("/", 200, request_headers=headers)
assert isinstance(json.loads(check("/api/research", 200, request_headers=headers)), dict)
assert json.loads(check("/healthz", 200, request_headers=headers)) == {"status": "ok"}
check("/api/research", 403, request_headers={**headers, "Origin": "https://evil.example",
      "X-Fusion-UI": "1", "Content-Type": "application/json"}, data=b"{}")
print("Container smoke passed: static UI, read-only research, auth, health and CSRF.")
