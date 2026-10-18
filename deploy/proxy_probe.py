"""Disposable CI upstream; reports only three harmless header checks."""
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class Probe(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        payload = json.dumps({
            "authorization_received": bool(self.headers.get_all("Authorization", [])),
            "proxy_token_valid": (
                len(self.headers.get_all("X-Fusion-Proxy-Token", [])) == 1
                and hmac.compare_digest(self.headers.get("X-Fusion-Proxy-Token", ""),
                                        os.environ["FUSION_PROXY_TOKEN"])
            ),
            "host": self.headers.get("Host"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8787), Probe).serve_forever()
