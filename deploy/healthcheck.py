"""Liveness only: never invokes inference, retrieval or an experiment."""
import json
from urllib.request import urlopen

with urlopen("http://127.0.0.1:8787/healthz", timeout=3) as response:
    if response.status != 200 or json.load(response) != {"status": "ok"}:
        raise SystemExit(1)
