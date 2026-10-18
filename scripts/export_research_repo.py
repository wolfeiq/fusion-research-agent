"""Export reviewed source and exact recorded evidence to a separate repository.

No Git, network, provider or experiment commands run here. Explicit invocation is
required. Existing files can be updated only if an earlier export owns their
unchanged bytes. No files are deleted, including files no longer in the allowlist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path, PurePosixPath

MANIFEST = ".export-manifest.json"
SCHEMA = "fusion-research-source-export-v1"
PILOTS = (
    "993df979e388405297f0195bfca504d6",
    "43d55b69d02440b3a019cf9b64f6845a",
)
BUNDLE_FILES = (
    "comparison.json", "manifest.json", "config.json", "run.json",
    "source.zip", "output/result.json",
)
ROOT_SCRIPTS = (
    "run_shadow_trial.py", "run_coupled_federation.py", "calibrate_deposition.py",
    "analyze_coupled_federation.py", "analyze_deposition_pilot.py",
    "stabilize_td3_heating.py", "export_research_repo.py",
)
ROOT_TESTS = (
    "__init__.py", "conftest.py", "test_shadow_trial.py", "test_beam_sac.py",
    "test_td3_initialization.py", "test_td3_action_coordinates.py",
    "test_critic_strategies.py", "test_repeated_federation.py",
    "test_coupled_control.py", "test_coupled_backend.py",
    "test_deposition_env.py", "test_torax_config.py", "test_devices.py",
    "test_export_research_repo.py",
)
PROTOCOLS = (
    "coldstart_coupled_federation.md", "coldstart_learning_diagnosis.md",
    "td3_heating_stability.md", "federated_rl_research.md",
)
APP_FILES = (
    "dashboard.py", "pyproject.toml", "uv.lock", ".gitignore", "README.md",
    "RESEARCH.md", "MERGE.md", "LICENSE.md", "THERMAL_LICENSE.md",
    "sites.example.json", "sites.remote.example.json",
)
APP_SCRIPTS = (
    "research_cookbook.py", "research-cookbook.requirements.txt",
    "verify_research_providers.py", "verify_research_judge.py",
    "verify_report_transport.py", "generate_thermal_fixtures.py",
    "prepare_hub.py", "HUB_README.md",
)

README = """# Fusion Research Agent

A research agent that reads real TORAX/TD3/SAC experiment evidence, searches and
extracts literature with Tavily, and uses seven LangGraph model contexts through
Nebius Token Factory to produce a sourced critique and a next-experiment proposal.
An optional bounded shadow trial tests a single change with fresh controllers.

This is an executable research prototype. Live-provider and end-to-end validation
is ongoing; the repository does not imply that every proposed change improves RL.
The bundled measurements are recorded local CPU experiments, not newly run when
you open the app. No Nebius VM, NVIDIA GPU deployment or production reactor control
is claimed. The current shadow worker uses CPU and in-process aggregation.

## Run the app and inspect real evidence

Install Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root:

```sh
cd flower-app
uv sync --locked --extra research --extra dev
uv run --no-sync python dashboard.py
```

Open http://127.0.0.1:8787/#research. Select the two published shadow comparisons
and use offline inspection without credentials. These are real recorded
measurements; offline inspection does not use an LLM or search the web.
Read [the evidence provenance](examples/evidence/README.md).

## Run the seven-agent research workflow

Copy `flower-app/research.env.example` to `flower-app/.env.research` and configure
`NEBIUS_API_KEY` and `TAVILY_API_KEY` locally. Never commit that file. Choose a
tool-capable model available to your Token Factory account. Stop the local
dashboard before restarting it with the environment file:

```sh
cd flower-app
uv run --env-file .env.research --no-sync python scripts/verify_research_providers.py
uv run --env-file .env.research --no-sync python dashboard.py
```

Ask: **Does the saved evidence justify claiming federation helps a cold-start
reactor, and which small experiment should we run next?** Select the real example
evidence and live mode. The trace records plans, search/extraction calls, findings,
citations and judgment. The judge can request a second research round or return
insufficient evidence. Exact quotations are checked against extracted pages;
experiment claims point into hash-verified JSON. Traceability alone is not proof
of semantic truth: inspect the judge's report and click the cited sources.

## Enable actual TORAX shadow trials

The app and RL worker deliberately use different virtual environments because
their Flower versions differ. The research UI can run without the RL environment.
To enable fresh numerical trials, run these commands from the repository root:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv --constraint constraints-rl-cpu.txt -e ".[rl,fed,dev]"
```

The constraints record the package versions of the CPU reference runtime; they
are not a complete transitive lock or a guarantee of identical cross-platform
numerics. The app has its own `uv.lock`. Root dependency resolution and platform
support must be verified before describing a new environment as reproduced.

In the research UI, prepare and start a shadow pilot, or explicitly enable one
shadow trial for a live agent run. A pilot freezes source, calibration and config,
then compares baseline/candidate across local, reset-only, uniform and clipped
arms. It uses real TORAX, TD3 and SAC. The default budget is 56 episodes and a
600-second wall limit; CPU resources remain shared with your other processes.
It does not load or modify existing experiments or apply a recommended policy.

## Architecture and scope

```mermaid
flowchart LR
  UI[Browser UI] --> APP[Python dashboard + LangGraph]
  APP --> TF[Nebius Token Factory: all research inference]
  APP --> TV[Tavily search + page extraction]
  DATA[Immutable recorded TORAX evidence] --> APP
  APP --> PLAN[Planner]
  PLAN --> TEAM[TD3 / SAC / federation / literature specialists]
  TEAM --> WRITE[Writer + citation verifier]
  WRITE --> JUDGE[Separate model judge]
  JUDGE -->|Evidence gaps: one revision| PLAN
  JUDGE --> REPORT[Cited findings + next-run proposal]
  TEAM -->|Optional bounded test| WORKER[Separate CPU TORAX worker]
  WORKER -->|Measured comparison + hashes| APP
```

The facility-investigation sections retain the earlier Flower AgentApp and
reduced-model demonstrations. Those synthetic facilities are distinct from the
research agent's real TORAX experiment evidence. The Flower hosted-run button is
an optional separate workflow requiring the user's own Flower login; it is not
the Token Factory research inference path. This repository does not require a
Flower account for the research workspace.

See [RESEARCH.md](flower-app/RESEARCH.md) for tools, fairness checks and limits.
Each numerical pilot is developmental evidence. One seed cannot establish
federation efficacy, and endpoint cost is not shots-to-competence.

## Tests

From `flower-app`:

```sh
uv run --no-sync python -m pytest tests -q
```

Frontend tests use Node's built-in test runner; there is no npm install/build:

```sh
node --test tests/test_shadow_frontend.cjs
```

With the root RL environment installed, from the root:

```sh
uv run --no-sync python -m pytest tests/test_shadow_trial.py -q
```

Unit fixtures may use a synthetic plant; they are labelled and never substituted
for the live TORAX command. A successful unit suite is not a live-provider result.
The four 3D device views require WebGL and externally hosted pinned Three.js
scripts; ordinary controls remain available if those views cannot load.
"""

IGNORE = """.venv*/
__pycache__/
*.py[cod]
*.egg-info/
.env
.env.*
.dashboard/
.fusion-state/
.uv-cache/
.jax_cache/
.pytest_cache/
.ruff_cache/
.test-tmp*/
pytest-cache-files-*/
.kaggle*/
dist/
*.fab
*.log
results/*
!results/deposition_calibration_verified/
results/deposition_calibration_verified/*
!results/deposition_calibration_verified/reference.json
"""

CONSTRAINTS = """# Observed CPU reference runtime; direct constraints, not a transitive lock.
# Install Torch from its CPU index first. +cpu satisfies the public version pin.
torax==1.4.3
torch==2.14.1
jax==0.11.1
jaxlib==0.11.1
numpy==2.5.3
scipy==1.18.1
flwr==1.22.0
gymnasium==1.3.0
matplotlib==3.11.2
pytest==9.1.1
ruff==0.16.7
"""

NOTICES = """# Provenance and third-party notices

This repository selectively exports the Fusion Research Agent application and
the minimal local TORAX/TD3/SAC shadow runner from the hfmarl-fusion working tree.
It retains the original Apache-2.0 notices. Consult flower-app/MERGE.md for the
Fusion Investigator/THERMAL merge history and original component attribution.
The export manifest records the copied source byte hashes; copied files may
include work that had not yet been committed in the original repository.

The 3D HTML exports load Three.js 0.128.0, OrbitControls and STLLoader from pinned
CDNs. Three.js is MIT licensed: https://github.com/mrdoob/three.js/blob/r128/LICENSE.
These visual assets are illustrations, not calibrated reactor geometry.

TORAX, JAX, PyTorch, Flower, LangGraph and other installed dependencies remain
subject to their respective licenses. Package source is not vendored here.
The app's use of the OpenAI Python client is solely an API-compatible client for
the configured Nebius Token Factory endpoint in the research workflow.

Evidence bundles contain only the explicitly selected terminal-run records and
their exact frozen source archives. They are measurements from a synthetic
SPARC-like circular TORAX model, not data from an operating reactor. Web sources
are retrieved at runtime through Tavily and are not republished in this export.
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(name):
    if (not isinstance(name, str) or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or PurePosixPath(name).is_absolute()):
        raise ValueError("Export paths must be canonical relative paths")
    return name


def reject_redirects(root, name):
    current = root
    for part in Path(safe_relative(name)).parts:
        current /= part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError(f"Export path contains a link or junction: {name}")
    if not current.resolve().is_relative_to(root):
        raise ValueError(f"Export path escapes its root: {name}")
    return current


def build_payload(source):
    source = Path(source).resolve(strict=True)
    payload, source_hashes = {}, {}

    def include(relative, destination=None):
        path = reject_redirects(source, relative)
        if not path.is_file():
            raise ValueError(f"Required export source is missing: {relative}")
        if path.stat().st_size > 16_000_000:
            raise ValueError(f"Export source exceeds size bound: {relative}")
        data = path.read_bytes()
        target = safe_relative(destination or relative)
        if target in payload:
            raise ValueError(f"Duplicate export target: {target}")
        payload[target] = data
        source_hashes[relative] = sha(data)

    include("pyproject.toml")
    for folder, suffixes in (
        ("hfmarl", {".py"}), ("flower-app/fusion_agent", {".py"}),
        ("flower-app/frontend", {".html", ".css", ".js"}),
        ("flower-app/tests", {".py", ".cjs"}),
    ):
        for path in sorted((source / folder).rglob("*")):
            relative = path.relative_to(source)
            if (path.is_file() and path.suffix in suffixes
                    and not any(part.startswith(".") or part == "__pycache__" for part in relative.parts)):
                include(relative.as_posix())
    for name in ROOT_SCRIPTS:
        include("scripts/" + name)
    for name in ROOT_TESTS:
        include("tests/" + name)
    for name in APP_FILES:
        include("flower-app/" + name)
    for name in APP_SCRIPTS:
        include("flower-app/scripts/" + name)
    for name in PROTOCOLS:
        include("docs/" + name)
    include("flower-app/fusion_agent/thermal/_devices.json")
    include("results/deposition_calibration_verified/reference.json")
    include("flower-app/THERMAL_LICENSE.md", "LICENSE.md")
    for name in ("diiid_like", "iter_like", "sparc_like", "tcv_like"):
        include(f"assets/3d/html/{name}.html")

    evidence_rows = []
    for identifier in PILOTS:
        private = f"flower-app/.dashboard/shadow/{identifier}"
        public = f"examples/evidence/shadow/{identifier}"
        run = json.loads(reject_redirects(source, private + "/run.json").read_bytes())
        if run.get("id") != identifier or run.get("status") != "completed":
            raise ValueError(f"Only the specified completed pilot may be exported: {identifier}")
        for name in BUNDLE_FILES:
            include(private + "/" + name, public + "/" + name)
        evidence_rows.append({
            "id": identifier, "recorded_status": run["status"], "config": run["config"],
            "result_sha256": sha(payload[public + "/output/result.json"]),
            "source_zip_sha256": sha(payload[public + "/source.zip"]),
        })

    evidence_readme = """# Recorded starter evidence

These two terminal local CPU pilots are genuine recorded TORAX runs, not mock
data and not a new live run. Open the research view to inspect both comparisons.
The catalogue verifies the source archive, frozen configuration, result hashes,
terminal run record and deterministic comparison arithmetic before use.

Each six-file bundle deliberately contains summary-level evidence, not every
raw trajectory. Its verification does not perform a new trajectory audit.
The exact original source archive is retained for provenance; it is hashed, never
executed by the evidence reader. New shadow runs use the current repository code.
Do not edit a recorded result to match a preferred interpretation or newer code.

The learning-rate pilot tested a lower TD3 actor rate and recorded worse endpoint
metrics. The clipping pilot tests a separate bounded aggregation hypothesis;
read its comparison for its result. Neither one-seed pilot establishes efficacy
or cold-start speedup. Positive, null and negative outcomes remain visible.

Recorded identifiers, configurations and immutable byte hashes:

```json
""" + json.dumps(evidence_rows, indent=2, sort_keys=True) + "\n```\n"
    generated = {
        "README.md": README, ".gitignore": IGNORE, "THIRD_PARTY_NOTICES.md": NOTICES,
        "constraints-rl-cpu.txt": CONSTRAINTS, ".python-version": "3.12\n",
        "examples/evidence/README.md": evidence_readme,
        "flower-app/research.env.example": "NEBIUS_API_KEY=\nTAVILY_API_KEY=\nNEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1\nNEBIUS_MODEL=MiniMaxAI/MiniMax-M3\n",
    }
    for name, text in generated.items():
        if name in payload:
            raise ValueError(f"Generated target collides with copied source: {name}")
        payload[name] = text.encode("utf-8")
    for name, digest in source_hashes.items():
        if sha(reject_redirects(source, name).read_bytes()) != digest:
            raise ValueError(f"Source changed during export capture: {name}")
    return payload, source_hashes


def write_export(source, destination, payload, source_hashes, *, dry_run=False):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if (destination.name != "fusion-research-agent" or destination.parent.resolve() != source.parent
            or destination.resolve() == source or destination.is_symlink()
            or (hasattr(destination, "is_junction") and destination.is_junction())):
        raise ValueError("Destination must be the separate sibling directory fusion-research-agent")
    if destination.exists() and not destination.is_dir():
        raise ValueError("Export destination is not a directory")
    old = {}
    manifest_path = reject_redirects(destination, MANIFEST)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_bytes())
        if previous.get("schema") != SCHEMA or not isinstance(previous.get("files"), dict):
            raise ValueError("Existing export manifest is not recognized")
        old = previous["files"]
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in old.values()):
            raise ValueError("Existing export manifest contains invalid hashes")
        for name in old:
            safe_relative(name)
    targets = {}
    for name in payload:
        path = reject_redirects(destination, name)
        if path.exists():
            if name not in old or not path.is_file():
                raise ValueError(f"Refusing to overwrite a file not owned by this export: {name}")
            if sha(path.read_bytes()) != old[name]:
                raise ValueError(f"Refusing to overwrite a modified exported file: {name}")
        targets[name] = path
    hashes = {name: sha(data) for name, data in payload.items()}
    manifest = {
        "schema": SCHEMA, "files": {**old, **hashes}, "source_file_sha256": source_hashes,
        "scope": "Selective source export plus two exact terminal evidence bundles; no credentials or runtime state",
        "deletion_policy": "Never delete; files absent from a later export are preserved",
    }
    if not dry_run:
        destination.mkdir(exist_ok=True)
        for name, path in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() == payload[name]:
                continue
            temporary = path.with_name(path.name + ".export-" + uuid.uuid4().hex + ".tmp")
            with temporary.open("xb") as handle:
                handle.write(payload[name])
            os.replace(temporary, path)
        temporary = manifest_path.with_name(MANIFEST + "." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
    return {"dry_run": dry_run, "destination": str(destination), "files": len(payload),
            "bytes": sum(map(len, payload.values())), "manifest": MANIFEST}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload, hashes = build_payload(args.source)
    print(json.dumps(write_export(args.source, args.destination, payload, hashes, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
