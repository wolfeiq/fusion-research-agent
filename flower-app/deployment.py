"""Fail-closed boundary for a single-user HTTPS reverse-proxy deployment.

Local development keeps its existing loopback policy. Public mode requires both
an exact canonical HTTPS origin and a private token injected by the authenticated
proxy. Forwarded headers never establish trust by themselves.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class AccessPolicy:
    bind: str = "127.0.0.1"
    public_origin: str | None = None
    proxy_token: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.bind not in {"127.0.0.1", "0.0.0.0"}:
            raise ValueError("Bind must be 127.0.0.1 or 0.0.0.0")
        if not self.public_origin:
            if self.bind != "127.0.0.1" or self.proxy_token:
                raise ValueError("External binding requires HTTPS origin and proxy token")
            return
        parsed = urlsplit(self.public_origin)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", parsed.hostname)
            or "." not in parsed.hostname
            or self.public_origin != f"https://{parsed.netloc.lower()}"
            or parsed.port is not None
        ):
            raise ValueError("Public origin must be a canonical HTTPS domain without a path")
        if (
            not isinstance(self.proxy_token, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", self.proxy_token)
        ):
            raise ValueError("Public mode requires a generated 32–256 character proxy token")

    def valid_host(self, host, port):
        if self.public_origin:
            return host == urlsplit(self.public_origin).netloc
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def valid_origin(self, origin, port):
        if self.public_origin:
            return origin == self.public_origin
        return origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def authenticated(self, headers):
        if not self.public_origin:
            return True
        # Duplicate tokens are rejected, including proxies that append rather
        # than overwrite the user-supplied header.
        values = headers.get_all("X-Fusion-Proxy-Token", [])
        return len(values) == 1 and hmac.compare_digest(
            values[0].encode("utf-8"), self.proxy_token.encode("utf-8")
        )

    @staticmethod
    def local_health(host, port, address):
        return address == "127.0.0.1" and host in {
            f"127.0.0.1:{port}", f"localhost:{port}"
        }
