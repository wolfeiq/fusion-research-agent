"""Live, bounded checks through the application's actual provider adapters.

Run with ``uv run --env-file .env.research --no-sync python
scripts/verify_research_providers.py``. Credentials are never saved in the report.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_agent.research.providers import NebiusModel, Settings, Tavily


def main():
    settings = Settings.from_env()
    settings.validate()
    started = time.monotonic()
    folder = Path(__file__).resolve().parents[1] / ".dashboard" / "provider-verification"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"started_at": datetime.now(UTC).isoformat(), "settings": settings.public()}

    def model_check():
        model = NebiusModel(settings)
        message, usage = model.complete(
            [{"role": "user", "content": "Call record_probe with status equal to ok."}],
            [{"type": "function", "function": {
                "name": "record_probe", "description": "Record a provider capability check.",
                "parameters": {"type": "object", "properties": {
                    "status": {"type": "string", "enum": ["ok"]}},
                    "required": ["status"], "additionalProperties": False},
            }}],
            {"type": "function", "function": {"name": "record_probe"}},
        )
        calls = message.get("tool_calls", [])
        passed = len(calls) == 1 and calls[0]["function"]["name"] == "record_probe"
        passed = passed and json.loads(calls[0]["function"]["arguments"]) == {"status": "ok"}
        return {"passed": passed, "message": message, "usage": usage}

    def retrieval_check():
        retrieval = Tavily(settings)
        search = retrieval.search("TORAX tokamak transport simulator original paper official documentation")
        urls = [r.get("url", "") for r in search.get("results", [])]
        primary = [u for u in urls if urlsplit(u).hostname in {
            "arxiv.org", "torax.readthedocs.io", "github.com"}]
        if not primary:
            return {"passed": False, "reason": "No primary-source URL found", "search": search}
        extract = retrieval.extract(primary[0])
        passed = any(r.get("raw_content", "").strip() for r in extract.get("results", []))
        return {"passed": bool(passed), "search": search, "extract": extract}

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {name: pool.submit(fn) for name, fn in (
            ("nebius_tool_call", model_check), ("tavily_search_extract", retrieval_check))}
        for name, future in jobs.items():
            try:
                report[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - never expose third-party error bodies
                # Third-party errors may include headers; save only the class.
                report[name] = {"passed": False, "error_type": type(exc).__name__}
    report["elapsed_seconds"] = time.monotonic() - started
    report["passed"] = all(report[k]["passed"] for k in jobs)
    destination = folder / "latest.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": {
        k: {n: v for n, v in report[k].items() if n in {"passed", "error_type", "reason"}}
        for k in jobs}, "elapsed_seconds": report["elapsed_seconds"], "report": str(destination)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
