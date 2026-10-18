"""Run the same read-only research service without the dashboard."""

import argparse
import json
import time
from pathlib import Path

from .service import DEFAULT_QUESTION, ResearchService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List permitted saved records"
    )
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        help="Selected catalogue ID (repeatable)",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--offline", action="store_true", help="Inventory only; no model or retrieval"
    )
    args = parser.parse_args()
    app_root = Path(__file__).resolve().parents[2]
    service = ResearchService(app_root.parent, app_root / ".dashboard" / "research-cli")
    if args.list:
        print(json.dumps(service.catalog.list_records(), indent=2))
        return
    if not args.record:
        parser.error("Choose --record IDs from --list")
    try:
        service.start(args.question, args.record, "offline" if args.offline else "live")
    except ValueError as exc:
        parser.error(str(exc))
    while service.active:
        time.sleep(0.1)
    run = service.export(service.run["id"])
    if run["error"]:
        raise SystemExit(run["error"])
    print(run["report"]["answer"])
    print("\nSaved research bundle:", service.state_dir / run["id"] / "run.json")
    if run["status"] == "insufficient_evidence":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
