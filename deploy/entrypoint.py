"""Load only named runtime secrets, then start the existing dashboard."""
import os
import runpy
from pathlib import Path


def main():
    for name in ("NEBIUS_API_KEY", "TAVILY_API_KEY", "FUSION_PROXY_TOKEN"):
        filename = os.environ.get(f"{name}_FILE")
        if filename:
            value = Path(filename).read_text(encoding="utf-8").strip()
            if not value or len(value) > 8192 or "\n" in value or "\r" in value:
                raise ValueError(f"Invalid secret file for {name}")
            os.environ[name] = value
    runpy.run_path("/app/flower-app/dashboard.py", run_name="__main__")


if __name__ == "__main__":
    main()
