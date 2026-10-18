"""One live formatting probe with an explicitly synthetic source, no web search."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_agent.research.engine import TOOLS, ResearchEngine
from fusion_agent.research.providers import NebiusModel, Settings


def main():
    settings = Settings.from_env()
    settings.validate()
    started = time.monotonic()
    result = {"purpose": "Synthetic serialization probe; not scientific evidence", "passed": False}
    tool = next(item for item in TOOLS if item["function"]["name"] == "finish_report")
    source = {"id": "E001", "kind": "experiment", "content": '{"cost":0.12}',
              "title": "Synthetic transport fixture", "uri": "fixture.json"}
    try:
        message, usage = NebiusModel(settings).complete([{"role": "user", "content":
            'This is a synthetic serialization test, not an experiment. Call finish_report '
            'using its flat fields, with answer="", one observation claim saying '
            '"The synthetic fixture cost is 0.12.", confidence="high", evidence '
            '[{"source_id":"E001","quote":"0.12","locator":"/cost"}], '
            'next_runs=[], limitations=["Synthetic formatting test only."], '
            'and stop_reason="Formatting probe completed.". Do not wrap fields in report '
            'or encode nested arrays as strings. Source E001 content is {"cost":0.12}.'}], [tool])
        result["usage"] = usage
        calls = message.get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "finish_report":
            raise ValueError("Missing finish_report call")
        args = json.loads(calls[0]["function"]["arguments"])
        result["argument_fields"] = sorted(args)
        engine = ResearchEngine([source], None, None, lambda _event: None)
        engine.planned = engine.assessed = True
        engine.read_ids.add("E001")
        report = engine.execute("finish_report", args)["report"]
        result["validation"] = report["verification"]
        result["passed"] = set(args) == {"answer", "claims", "next_runs", "limitations", "stop_reason"} and report["verification"]["traceable_claims"] == 1
    except Exception as exc:  # noqa: BLE001 - no provider bodies or credentials
        result["error_type"] = type(exc).__name__
    result["elapsed_seconds"] = time.monotonic() - started
    path = Path(__file__).resolve().parents[1] / ".dashboard/provider-verification/report-transport-probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
