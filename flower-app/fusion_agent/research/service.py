"""Separate, persistent dashboard job state for read-only research."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .engine import ResearchEngine, offline_report
from .evidence import EvidenceCatalog
from .providers import NebiusModel, Settings, Tavily

DEFAULT_QUESTION = (
    "Does the saved evidence justify claiming federated TD3/SAC accelerates "
    "cold-start reactor control? Diagnose the strongest alternative explanations "
    "and propose a small, fair next experiment with cited research."
)

_STATE_REPLACE_DELAYS = (.01, .02, .04, .08, .16)


def _replace_state_file(temporary, target):
    """Bounded retries for Windows reader locks, without removing old state."""
    for attempt in range(len(_STATE_REPLACE_DELAYS) + 1):
        try:
            os.replace(temporary, target)
            return
        except PermissionError as exc:
            if (getattr(exc, "winerror", None) not in (5, 32, 33)
                    or attempt == len(_STATE_REPLACE_DELAYS)):
                raise
            time.sleep(_STATE_REPLACE_DELAYS[attempt])


def _failure_diagnostic(exc, secrets=()):
    """Keep bounded code locations, never exception text, source or locals."""
    def label(value, *, filename=False):
        if not isinstance(value, str):
            return "<redacted>"
        if filename:
            value = value.replace("\\", "/").rsplit("/", 1)[-1]
        if any(secret and secret in value for secret in secrets):
            return "<redacted>"
        if len(value) > 96 or not re.fullmatch(r"[A-Za-z0-9_.-]+|<(?:module|listcomp|dictcomp|setcomp|genexpr|lambda)>", value):
            return "<redacted>"
        return value

    frames = []
    for frame in traceback.extract_tb(exc.__traceback__, limit=-12):
        frames.append({"file": label(frame.filename, filename=True),
                       "function": label(frame.name), "line": frame.lineno})
    diagnostic = {"exception_type": label(type(exc).__name__), "frames": frames,
                  "scope": "Exception class and bounded code locations only; no exception message, source text, full paths, payload or locals."}
    for key in ("errno", "winerror"):
        value = getattr(exc, key, None)
        if type(value) is int and -(2**31) <= value < 2**31:
            diagnostic[key] = value
    return diagnostic


def _public_failure_message(exc, secrets=()):
    """Redact complete credentials before applying the public message bound."""
    from .providers import ProviderError

    message = (
        str(exc)
        if isinstance(exc, (ValueError, RuntimeError, ProviderError))
        else "Research failed while processing a response; inspect the saved trace and retry"
    )
    # Redacting after slicing can leave a credential prefix at the boundary.
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        message = message.replace(secret, "[redacted]")
    return message[:1500]


class ResearchService:
    def __init__(
        self,
        repo_root,
        state_dir,
        *,
        settings_factory=Settings.from_env,
        model_factory=NebiusModel,
        retrieval_factory=Tavily,
        shadow_service=None,
    ):
        self.catalog = EvidenceCatalog(Path(repo_root))
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.settings_factory = settings_factory
        self.model_factory, self.retrieval_factory = model_factory, retrieval_factory
        self.shadow_service = shadow_service
        self.lock = threading.RLock()
        self.active = False
        self.run = {
            "id": None,
            "status": "idle",
            "mode": "offline",
            "question": DEFAULT_QUESTION,
            "events": [],
            "sources": [],
            "report": None,
            "error": None,
        }
        latest = self.state_dir / "latest.json"
        if latest.is_file():
            try:
                run_id = json.loads(latest.read_text(encoding="utf-8"))["id"]
                self.run = self.export(run_id)
                if self.run["status"] in ("running", "starting"):
                    self.run["status"] = "interrupted"
                    self.run["error"] = (
                        "Server restarted; saved evidence is retained. Start a new run to continue."
                    )
                    self._save()
            except (ValueError, KeyError, OSError):
                self.run["error"] = "Saved research state could not be read"

    def _save(self):
        if not self.run.get("id"):
            return
        folder = self.state_dir / self.run["id"]
        folder.mkdir(exist_ok=True)
        temporary = folder / "run.tmp"
        temporary.write_text(
            json.dumps(self.run, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        _replace_state_file(temporary, folder / "run.json")
        latest_tmp = self.state_dir / "latest.tmp"
        latest_tmp.write_text(json.dumps({"id": self.run["id"]}), encoding="utf-8")
        _replace_state_file(latest_tmp, self.state_dir / "latest.json")

    def snapshot(self):
        with self.lock:
            run = copy.deepcopy(self.run)
            run["sources"] = [
                {k: v for k, v in s.items() if k != "content"} for s in run["sources"]
            ]
            try:
                catalog, catalog_error = self.catalog.list_records(), None
            except ValueError:
                catalog, catalog_error = (
                    [],
                    "Evidence catalogue unavailable or over its bounded record limit",
                )
            return {
                "active": self.active,
                "configured": self.settings_factory().public(),
                "run": run,
                "catalog": catalog,
                "catalog_error": catalog_error,
            }

    def start(self, question, record_ids, mode="live", allow_shadow=False):
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 4000:
            raise ValueError("Enter a research question of1–4000 characters")
        if mode not in ("live", "offline"):
            raise ValueError("Choose live or offline mode")
        if type(allow_shadow) is not bool or (allow_shadow and (mode != "live" or self.shadow_service is None)):
            raise ValueError("Shadow execution requires an enabled local shadow service and live research mode")
        if (
            not isinstance(record_ids, list)
            or not 1 <= len(record_ids) <= 8
            or any(not isinstance(s, str) for s in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError("Select1–8 distinct evidence records")
        settings = self.settings_factory()
        if mode == "live":
            settings.validate()
        with self.lock:
            if self.active:
                raise ValueError("A research run is already active")
            if self.run.get("persistence_failure", {}).get("terminal_state_persistence_confirmed") is False:
                # A prior final report may exist only in this session. Preserve
                # it, including its failure audit, before replacing self.run.
                # _save already bounds transient Windows replace retries.
                try:
                    self._save()
                except Exception:  # noqa: BLE001 - do not expose storage errors or replace retained evidence
                    raise ValueError(
                        "The previous research run could not be saved, so a new run was not started. "
                        "Download its run export before restarting this server, then check "
                        "available storage and write permissions and retry."
                    ) from None
            sources = []
            for index, record_id in enumerate(record_ids, 1):
                source = self.catalog.snapshot(record_id)
                source["record_id"] = source["id"]
                source["id"] = f"E{index:03d}"
                sources.append(source)
            if not any(s["kind"] == "experiment" for s in sources):
                raise ValueError("Select at least one experiment summary")
            if sum(len(s["content"]) for s in sources) > 100_000:
                raise ValueError("Selected evidence is too large; choose fewer records")
            self.run = {
                "id": uuid.uuid4().hex,
                "status": "running",
                "mode": mode,
                "allow_shadow": allow_shadow,
                "question": question.strip(),
                "started_at": datetime.now(UTC).isoformat(),
                "sources": sources,
                "events": [],
                "report": None,
                "error": None,
                "provider": {
                    "model": settings.model,
                    "endpoint": settings.public()["base_url"],
                    "max_output_tokens": settings.max_output_tokens,
                    "enable_thinking": settings.enable_thinking,
                }
                if mode == "live"
                else {"model": None, "endpoint": None},
                "access": "Read-only snapshots plus one isolated bounded shadow pilot; no current experiment mutation" if allow_shadow else "Read-only snapshots; no experiment launch or mutation tools",
            }
            self.active = True
            try:
                self._save()
                threading.Thread(
                    target=self._worker, args=(settings,), daemon=True
                ).start()
            except Exception:  # noqa: BLE001 - restore state if worker startup fails
                self.active = False
                self.run["status"] = "failed"
                self.run["error"] = (
                    "Could not start the research job or save its evidence"
                )
                raise ValueError(self.run["error"]) from None
        return self.snapshot()

    def _event(self, event, *, run_id=None):
        with self.lock:
            # A cancelled parallel researcher may finish after its graph failed.
            # Its callback must never append evidence to a subsequent job.
            if run_id is not None and (self.run.get("id") != run_id or not self.active):
                return
            event = copy.deepcopy(event)
            source = event.pop("source", None)
            if source:
                sources = {s["id"]: s for s in self.run["sources"]}
                sources[source["id"]] = source
                self.run["sources"] = list(sources.values())
                event["source_id"] = source["id"]
            event["time"] = datetime.now(UTC).isoformat()
            self.run["events"].append(event)
            self._save()

    def _worker(self, settings):
        run_id = self.run["id"]

        def emit(event):
            self._event(event, run_id=run_id)

        try:
            emit(
                {
                    "kind": "snapshot",
                    "message": "Selected saved records captured. Current experiments are untouched.",
                }
            )
            if self.run["mode"] == "offline":
                sources = self.run["sources"]
                dossiers = [
                    s
                    for s in sources
                    if s.get("metadata", {}).get("adapter") == "federation_dossier"
                ]
                if dossiers:
                    inspector = ResearchEngine(sources, None, None, emit)
                    inspector.execute(
                        "plan_research",
                        {
                            "questions": [
                                "Inspect the latest saved round of every recorded job in each selected dossier"
                            ],
                            "reason": "Deterministic coordinator inspection; no model decisions or new policy recommendations.",
                        },
                    )
                    for source in dossiers:
                        inspector.execute(
                            "inspect_federation",
                            {
                                "source_id": source["id"],
                                "reason": "Read the captured coordinator dossier.",
                            },
                        )
                        data = json.loads(source["content"])
                        for job in data["jobs"]:
                            if job["rounds"]:
                                inspector.execute(
                                    "compare_clipping",
                                    {
                                        "source_id": source["id"],
                                        "job_id": job["id"],
                                        "round": job["rounds"][-1]["round"],
                                        "threshold_scale": 1,
                                        "reason": "Check recorded clipping activation at the latest saved round; report unavailable telemetry explicitly.",
                                    },
                                )
                    sources = list(inspector.sources.values())
                report = offline_report(sources)
                emit(
                    {
                        "kind": "offline",
                        "message": "Deterministic local evidence inspection; no LLM or web requests.",
                    }
                )
            else:
                execution_options = ({"shadow_executor": self.shadow_service.run_for_advisor, "max_seconds": 1080, "max_turns": 36}
                                     if self.run.get("allow_shadow") else {})
                engine = ResearchEngine(
                    self.run["sources"],
                    self.model_factory(settings),
                    self.retrieval_factory(settings),
                    emit,
                    **execution_options,
                )
                report = engine.run(self.run["question"])
            with self.lock:
                self.run["report"] = report
                if self.run["mode"] == "offline":
                    self.run["status"] = "inspection_complete"
                else:
                    self.run["status"] = (
                        "completed"
                        if report.get("evidence_sufficient")
                        else "insufficient_evidence"
                    )
        except Exception as exc:  # noqa: BLE001 - isolate background job failures
            # Only our safe validation/provider messages may reach the browser.
            message = _public_failure_message(
                exc, (settings.nebius_key, settings.tavily_key)
            )
            with self.lock:
                diagnostic = _failure_diagnostic(
                    exc, (settings.nebius_key, settings.tavily_key)
                )
                self.run.update(status="failed", error=message,
                                failure_diagnostic=diagnostic)
                self.run["events"].append({
                    "kind": "failure_diagnostic",
                    "message": "Research failed; bounded code-location diagnostics were retained without provider content.",
                    "diagnostic": copy.deepcopy(diagnostic),
                    "time": datetime.now(UTC).isoformat(),
                })
        finally:
            with self.lock:
                self.active = False
                self.run["finished_at"] = datetime.now(UTC).isoformat()
                try:
                    self._save()
                except Exception as exc:  # noqa: BLE001 - preserve the session export when terminal persistence fails
                    diagnostic = _failure_diagnostic(
                        exc, (settings.nebius_key, settings.tavily_key)
                    )
                    # _save keeps atomic old snapshots and already bounds Windows
                    # replace retries. Do not recursively save this failure or
                    # claim the final report is durable when publication failed.
                    self.run["persistence_failure"] = {
                        "terminal_state_persistence_confirmed": False,
                        "previous_run_status": self.run["status"],
                        "previous_error": self.run.get("error"),
                        "diagnostic": diagnostic,
                        "evidence_scope": "Current server-session export and the last successfully saved snapshot; final disk contents are unconfirmed.",
                    }
                    message = (
                        "The final research state could not be saved. Download the run export "
                        "before restarting this server, then check available storage and write "
                        "permissions. Captured evidence remains available in this session; "
                        "the final on-disk state is unconfirmed."
                    )
                    self.run.update(status="failed", error=message)
                    self.run["events"].append({
                        "kind": "persistence_failure", "message": message,
                        "diagnostic": copy.deepcopy(diagnostic),
                        "time": datetime.now(UTC).isoformat(),
                    })

    def export(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Unknown research run")
        with self.lock:
            if self.run.get("id") == run_id:
                return copy.deepcopy(self.run)
            path = self.state_dir / run_id / "run.json"
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ValueError("Unknown research run") from None

    def source(self, run_id, source_id):
        for source in self.export(run_id)["sources"]:
            if source["id"] == source_id:
                return source
        raise ValueError("Unknown research source")
