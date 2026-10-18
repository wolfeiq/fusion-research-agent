"""Local frontend for the Flower AgentApp. Run with this project's Python."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from deployment import AccessPolicy
from fusion_agent.sites import load_sites
from fusion_agent.tools import Toolbox

ROOT = Path(__file__).resolve().parent
# Submitting to a hardcoded federation breaks for every other account: the
# author's "@marykor/personal" is not joinable by a presenter logged in as
# someone else, and the CLI then reports success=false with no run ID. Leave it
# unset to use the logged-in account's default federation; override with
# FUSION_FEDERATION=@account/name when a shared federation exists.
FEDERATION = os.environ.get("FUSION_FEDERATION", "").strip() or None
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def decode_logs(logs):
    clean = ANSI.sub("", logs)
    audit = []
    report = clean
    if "FUSION_TOOL_AUDIT " in clean:
        report, tail = clean.split("FUSION_TOOL_AUDIT ", 1)
        try:
            audit, _ = json.JSONDecoder().raw_decode(tail.strip())
        except ValueError:
            pass
    # Keep raw logs separately; remove the runtime's installation preamble.
    lines = report.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("#")), None)
    if start is not None:
        report = "\n".join(lines[start:])
    return {"report": report.strip(), "audit": audit, "logs": clean[-180000:]}


class Dashboard:
    def __init__(self, state_dir=None):
        self.state_dir = state_dir or ROOT / ".dashboard"
        self.state_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.active = False
        self.run = {"status": "idle", "report": "", "audit": [], "logs": ""}
        path = self.state_dir / "run.json"
        if path.exists():
            self.run = json.loads(path.read_text(encoding="utf-8"))
            if self.run["status"] in ("submitting", "pending", "starting", "running"):
                self.run["status"] = "monitor interrupted"
        # Rehearsal state is separate from hosted-agent and remote facility ledgers.
        self.sites = load_sites(ledger_dir=self.state_dir / "rehearsal")
        self.thermal = {"mode": "local-rehearsal", "events": []}
        trace_path = self.state_dir / "thermal.json"
        if trace_path.exists():
            self.thermal = json.loads(trace_path.read_text(encoding="utf-8"))
        # This local research service reads immutable summaries; it never calls
        # the Flower submission path or the TORAX training runner.
        from fusion_agent.research.service import ResearchService
        from fusion_agent.research.shadow import ShadowService

        self.shadow = ShadowService(ROOT.parent, self.state_dir / "shadow")
        self.research = ResearchService(ROOT.parent, self.state_dir / "research", shadow_service=self.shadow)

    def thermal_snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.thermal))

    def release(self, site, kind):
        if site not in ("facility-a", "facility-b", "facility-c"):
            raise ValueError("Select a configured thermal facility")
        if kind not in ("context", "balance", "source_check", "raw_logs"):
            raise ValueError("Unknown evidence product")
        with self.lock:
            result = self.sites[site].call("request_evidence", {"kind": kind})
            if self.thermal.get("case_id") != result["case_id"]:
                self.thermal = {
                    "mode": "local-rehearsal",
                    "case_id": result["case_id"],
                    "events": [],
                }
            self.thermal["events"].append(result)
            self.save_thermal()
            return self.thermal_snapshot()

    def save_thermal(self):
        temporary = self.state_dir / "thermal.tmp"
        temporary.write_text(
            json.dumps(self.thermal, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(self.state_dir / "thermal.json")

    def replay(self):
        # This fixed rehearsal makes no model calls. Budgets are never reset.
        with self.lock:
            self.thermal["events"] = []
            for facility, kind in (
                ("a", "context"),
                ("a", "balance"),
                ("b", "context"),
                ("c", "context"),
                ("c", "balance"),
                ("b", "balance"),
                ("b", "source_check"),
                ("a", "source_check"),
                ("b", "raw_logs"),
                ("b", "source_check"),
            ):
                self.release("facility-" + facility, kind)
            return self.thermal_snapshot()

    def update(self, **values):
        with self.lock:
            self.run.update(values)
            temporary = self.state_dir / "run.tmp"
            temporary.write_text(json.dumps(self.run), encoding="utf-8")
            temporary.replace(self.state_dir / "run.json")

    def snapshot(self):
        with self.lock:
            return {
                **self.run,
                "active": self.active,
                "federation": FEDERATION or "account default",
            }

    def cli(self, args):
        executable = Path(sys.executable).with_name(
            "flwr.exe" if os.name == "nt" else "flwr"
        )
        result = subprocess.run(
            [str(executable), *args],
            check=False,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env={**os.environ, "PYTHONUTF8": "1", "NO_COLOR": "1"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout)[-4000:])
        return result.stdout

    def start(self, prompt):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 12000:
            raise ValueError("Enter an investigation prompt of 1–12000 characters.")
        with self.lock:
            if self.active:
                raise ValueError("An investigation is already running.")
            self.active = True
            self.update(
                status="submitting",
                prompt=prompt.strip(),
                run_id=None,
                report="",
                audit=[],
                logs="",
                error=None,
            )
        threading.Thread(
            target=self.worker, args=(prompt.strip(),), daemon=True
        ).start()
        return self.snapshot()

    def worker(self, prompt):
        try:
            # A file avoids shell quoting and preserves multiline prompts as TOML.
            import tomli_w

            config = self.state_dir / "request.toml"
            config.write_text(tomli_w.dumps({"agent.input": prompt}), encoding="utf-8")
            submitted = json.loads(
                self.cli(
                    [
                        "run",
                        ".",
                        "supergrid",
                        *(["--federation", FEDERATION] if FEDERATION else []),
                        "--run-config",
                        str(config),
                        "--format",
                        "json",
                    ]
                )
            )
            run_id = submitted.get("run-id")
            if not submitted.get("success") or not run_id or not run_id.isdecimal():
                raise RuntimeError("Flower did not return a run ID.")
            self.update(run_id=run_id, status="pending")
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                details = json.loads(
                    self.cli(
                        [
                            "ls",
                            "supergrid",
                            "--run-id",
                            run_id,
                            "--format",
                            "json",
                        ]
                    )
                )["runs"][0]
                status = details["status"]
                logs = self.cli(["log", run_id, "supergrid", "--show"])
                self.update(status=status, **decode_logs(logs))
                if "finished" in status.lower():
                    return
                time.sleep(5)
            self.update(
                status="monitor timed out", error="Open Flower to check this run."
            )
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            self.update(status="connection error", error=str(exc))
        finally:
            with self.lock:
                self.active = False


# The WebGL reactor assemblies are committed at the repository root; serve them
# under /twin/ through an explicit allowlist only (missing files 404 harmlessly
# in checkouts that do not carry the assets).
TWIN_PAGES = {
    f"/twin/{name}.html": ROOT.parent / "assets" / "3d" / "html" / f"{name}.html"
    for name in ("iter_like", "sparc_like", "diiid_like", "tcv_like")
}
TWIN_PAGES = {k: v for k, v in TWIN_PAGES.items() if v.exists()}


def make_server(app, port=8787, *, access=None):
    access = access or AccessPolicy()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, data, status=200, content_type="application/json", csp=None):
            payload = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            # Default policy: no remote scripts, top page never framed. The
            # only additions over the original are the interface's Google Fonts and
            # permission to frame our own /twin/ pages.
            self.send_header(
                "Content-Security-Policy",
                csp
                or (
                    "default-src 'self'; "
                    "script-src 'self'; "
                    "style-src 'self' https://fonts.googleapis.com; "
                    "font-src https://fonts.gstatic.com; "
                    "img-src 'self' data:; "
                    "frame-src 'self'; frame-ancestors 'none'"
                ),
            )
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            return len(self.headers.get_all("Host", [])) == 1 and access.valid_host(
                self.headers.get("Host"), self.server.server_port
            )

        def authorized(self):
            return self.valid_host() and access.authenticated(self.headers)

        def local_health(self):
            return (
                self.path == "/healthz"
                and len(self.headers.get_all("Host", [])) == 1
                and access.local_health(
                    self.headers.get("Host"), self.server.server_port,
                    self.client_address[0],
                )
            )

        def do_GET(self):
            if self.local_health():
                self.send({"status": "ok"})
                return
            if not self.authorized():
                self.send({"error": "Access denied"}, 403)
                return
            if self.path == "/healthz":
                self.send({"status": "ok"})
            elif self.path == "/api/run":
                self.send(app.snapshot())
            elif self.path == "/api/research":
                self.send(app.research.snapshot())
            elif self.path == "/api/shadow":
                self.send(app.shadow.snapshot())
            elif urlsplit(self.path).path == "/api/shadow/export":
                try:
                    query = parse_qs(urlsplit(self.path).query)
                    self.send(app.shadow.export(query.get("id", [None])[0]))
                except ValueError as exc:
                    self.send({"error": str(exc)}, 404)
            elif urlsplit(self.path).path in ("/api/research/source", "/api/research/export"):
                query = parse_qs(urlsplit(self.path).query)
                try:
                    run_id = query.get("run_id", [None])[0]
                    if urlsplit(self.path).path.endswith("/source"):
                        self.send(app.research.source(run_id, query.get("source_id", [None])[0]))
                    else:
                        self.send(app.research.export(run_id))
                except ValueError as exc:
                    self.send({"error": str(exc)}, 404)
            elif self.path == "/api/thermal":
                self.send(app.thermal_snapshot())
            elif self.path == "/api/evidence":
                box = Toolbox(app.sites)
                self.send(
                    {
                        "diagnoses": [
                            box.execute(
                                "diagnose_heating",
                                {"site": site, "case_id": "heating-response"},
                            )
                            for site in app.sites
                            if "diagnose_heating"
                            in app.sites[site].describe()["capabilities"]
                        ]
                    }
                )
            elif self.path == "/api/devices":
                self.send(
                    json.loads(
                        (
                            ROOT / "fusion_agent" / "thermal" / "_devices.json"
                        ).read_text()
                    )
                )
            elif self.path in TWIN_PAGES:
                # Fixed mapping, never the request path: the 3D assemblies live
                # outside frontend/ and nothing else there may be reachable.
                # These self-contained pages inline their scripts and pull
                # three.js from pinned CDNs, and may be framed by us alone.
                self.send(
                    TWIN_PAGES[self.path].read_text(encoding="utf-8").replace(
                        "</body>", '<script src="/twin-selector.js"></script></body>'
                    ).encode("utf-8"),
                    content_type="text/html; charset=utf-8",
                    csp=(
                        "default-src 'none'; "
                        "script-src 'self' 'unsafe-inline' "
                        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
                        "style-src 'unsafe-inline'; img-src data: blob:; "
                        "connect-src 'self'; frame-ancestors 'self'"
                    ),
                )
            elif self.path in ("/", "/app.js", "/thermal.js", "/physics.js", "/twin-selector.js", "/style.css", "/research.js", "/research.css"):
                name = "index.html" if self.path == "/" else self.path[1:]
                mime = {
                    "index.html": "text/html",
                    "app.js": "text/javascript",
                    "thermal.js": "text/javascript",
                    "physics.js": "text/javascript",
                    "twin-selector.js": "text/javascript",
                    "style.css": "text/css",
                    "research.js": "text/javascript",
                    "research.css": "text/css",
                }[name]
                self.send(
                    (ROOT / "frontend" / name).read_bytes(),
                    content_type=mime + "; charset=utf-8",
                )
            else:
                self.send({"error": "Not found"}, 404)

        def do_HEAD(self):
            # Availability check for the fixed reactor asset set; never reads a
            # user-supplied filesystem path or executes a tool.
            status = (200 if self.local_health() else 403 if not self.authorized()
                      else 200 if self.path in TWIN_PAGES or self.path == "/healthz"
                      else 404)
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (
                not self.authorized()
                or len(self.headers.get_all("Origin", [])) != 1
                or not access.valid_origin(origin, self.server.server_port)
                or self.headers.get("X-Fusion-UI") != "1"
                or self.headers.get("Content-Type") != "application/json"
            ):
                # A bounded body drain avoids closing a socket with unread POST
                # bytes (which can reset the connection before the 403 reaches
                # a Windows client). It never parses or executes the request.
                try:
                    rejected_length = int(self.headers.get("Content-Length", "0"))
                    if 0 < rejected_length <= 65536:
                        self.connection.settimeout(2)
                        self.rfile.read(rejected_length)
                except (ValueError, OSError):
                    pass
                self.send(
                    {"error": "Only same-origin dashboard requests are allowed"}, 403
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 65536:
                    raise ValueError("Request is too large or empty")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise TypeError("Expected a JSON object")
                if self.path == "/api/run":
                    self.send(app.start(body.get("prompt")), 202)
                elif self.path == "/api/research":
                    self.send(app.research.start(body.get("question"),
                        body.get("record_ids"), body.get("mode", "live"), body.get("allow_shadow", False)), 202)
                elif self.path == "/api/shadow/prepare":
                    self.send(app.shadow.prepare(body), 201)
                elif self.path == "/api/shadow/start":
                    if set(body) != {"id"}:
                        raise ValueError("Start accepts only a frozen trial ID")
                    self.send(app.shadow.start(body.get("id")), 202)
                elif self.path == "/api/release":
                    self.send(app.release(body.get("site"), body.get("kind")))
                elif self.path == "/api/replay":
                    self.send(app.replay())
                elif self.path == "/api/validate":
                    result = Toolbox(app.sites).execute(
                        "validate_heating",
                        {
                            "site": body.get("site"),
                            "case_id": "heating-response",
                            "power_scale": body.get("power_scale"),
                            "steps": 60,
                        },
                    )
                    self.send(result, 400 if "error" in result else 200)
                else:
                    self.send({"error": "Not found"}, 404)
            except (ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)

    return ThreadingHTTPServer((access.bind, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--bind", default=os.environ.get("FUSION_BIND", "127.0.0.1"))
    parser.add_argument("--import-log", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        access = AccessPolicy(
            bind=args.bind,
            public_origin=os.environ.get("FUSION_PUBLIC_ORIGIN") or None,
            proxy_token=os.environ.get("FUSION_PROXY_TOKEN") or None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    app = Dashboard()
    if args.import_log:
        if not args.run_id or not args.run_id.isdecimal():
            parser.error("--import-log requires a numeric --run-id")
        details = json.loads(
            app.cli(
                [
                    "ls",
                    "supergrid",
                    "--run-id",
                    args.run_id,
                    "--format",
                    "json",
                ]
            )
        )["runs"][0]
        app.update(
            run_id=args.run_id,
            status=details["status"],
            prompt="Investigate heating underperformance at demo-a. Compare "
            "partner evidence, validate a candidate, and recommend the next diagnostic.",
            **decode_logs(args.import_log.read_text(encoding="utf-8-sig")),
        )
    server = make_server(app, args.port, access=access)
    url = access.public_origin or f"http://127.0.0.1:{server.server_port}"
    print(f"Fusion Investigator: {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
