"""Bounded local shadow experiments, isolated from the live experiment tree.

Only a typed single-variable trial is executable. The worker imports a private
source snapshot, trains fresh learners, and writes into its own UUID directory.
This is process/output isolation, not an operating-system security sandbox.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path

SCHEMA = "fusion-shadow-config-v1"
MAX_SECONDS = 600
REFERENCE = "results/deposition_calibration_verified/reference.json"
CHANGES = {
    "td3_actor_lr": {"label": "TD3 actor learning rate", "type": "number", "min": 1e-6, "max": 1e-3, "baseline": .0001, "default": .00005},
    "sac_actor_lr": {"label": "SAC actor learning rate", "type": "number", "min": 1e-6, "max": 1e-3, "baseline": .0003, "default": .0001},
    "sac_alpha_lr": {"label": "SAC entropy learning rate", "type": "number", "min": 1e-6, "max": 1e-3, "baseline": .0003, "default": .0001},
    "td3_mix": {"label": "TD3 critic blend fraction", "type": "number", "min": 0., "max": 1., "baseline": .25, "default": .1},
    "sac_mix": {"label": "SAC critic blend fraction", "type": "number", "min": 0., "max": 1., "baseline": .25, "default": .1},
    "clipping_initial_norm_fraction": {"label": "Critic delta clipping fraction", "type": "number", "min": .001, "max": 1., "baseline": .1, "default": .01},
    "heating_critic_coordinates": {"label": "TD3 critic action coordinates", "type": "select", "options": ["normalized", "physical"], "baseline": "normalized", "default": "physical"},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def validate_config(request):
    if not isinstance(request, dict) or set(request) - {"schema", "change", "value", "seeds", "rounds"}:
        raise ValueError("Shadow trials accept only change, value, seeds and rounds")
    if request.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("Unknown shadow configuration schema")
    name = request.get("change")
    if not isinstance(name, str) or name not in CHANGES:
        raise ValueError("Choose a registered shadow change")
    spec, value = CHANGES[name], request.get("value")
    if spec["type"] == "select":
        if value not in spec["options"]:
            raise ValueError("Unknown critic coordinates")
    elif type(value) not in (int, float) or not math.isfinite(value) or not spec["min"] <= value <= spec["max"]:
        raise ValueError("Candidate value is outside the bounded development range")
    if value == spec["baseline"]:
        raise ValueError("Candidate must change the baseline value")
    seeds, rounds = request.get("seeds", [17]), request.get("rounds", 1)
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or any(type(seed) is not int or not 0 <= seed <= 1_000_000 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Use 1–3 distinct nonnegative development seeds")
    if type(rounds) is not int or rounds not in (1, 2):
        raise ValueError("Shadow pilots support 1 or 2 rounds")
    return {"schema": SCHEMA, "change": name, "value": value, "seeds": seeds, "rounds": rounds}


def process_identity(pid):
    """PID plus process creation identity; never signal a reused PID."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            stamps = [wintypes.FILETIME() for _ in range(4)]
            if not api.GetProcessTimes(handle, *(ctypes.byref(s) for s in stamps)):
                return None
            return str((stamps[0].dwHighDateTime << 32) | stamps[0].dwLowDateTime)
        finally:
            api.CloseHandle(handle)
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None


class ShadowService:
    def __init__(self, repo_root, state_dir, *, python=None, popen=subprocess.Popen):
        self.root = Path(repo_root).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.python = Path(python or self.root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        self._popen = popen
        self.lock = threading.RLock()
        self.run = {"id": None, "status": "idle", "events": [], "comparison": None, "error": None}
        self.active = False
        self.process = None
        self.lease = self.state_dir / "active.json"
        latest = self.state_dir / "latest.json"
        if self.lease.exists():
            latest = self.lease
        if latest.is_file():
            try:
                self.run = self.export(json.loads(latest.read_text())["id"])
                if self.run["status"] in ("running", "launching", "recovery_required") or (
                    self.lease.exists() and self.run["status"] not in ("completed", "failed", "timed_out", "interrupted")
                ):
                    self.active = True
                    # A server restart cannot launch a duplicate. Recover only the
                    # saved worker's identity or its durable completed result.
                    threading.Thread(target=self._recover, daemon=True).start()
                elif self.lease.exists():
                    self._release_lease()
            except (ValueError, OSError, KeyError):
                self.run = {"id": None, "status": "idle", "events": [], "comparison": None, "error": "Saved shadow index could not be read"}

    def _claim_lease(self):
        try:
            with self.lease.open("xb") as handle:
                handle.write(json_bytes({"id": self.run["id"]}))
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise ValueError("Another shadow worker owns the execution lease; no duplicate was launched") from None

    def _release_lease(self):
        if self.lease.exists() and json.loads(self.lease.read_text()).get("id") == self.run.get("id"):
            self.lease.unlink()

    def _folder(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch("[a-f0-9]{32}", identifier):
            raise ValueError("Unknown shadow trial")
        folder = self.state_dir / identifier
        if folder.resolve() != folder or self.state_dir.resolve() != self.state_dir:
            raise ValueError("Shadow paths may not redirect outside their trial")
        return folder

    def _save(self):
        atomic_json(self._folder(self.run["id"]) / "run.json", self.run)
        atomic_json(self.state_dir / "latest.json", {"id": self.run["id"]})

    def _event(self, message):
        self.run["events"].append({"time": time.time(), "message": message})

    def snapshot(self):
        with self.lock:
            run = copy.deepcopy(self.run)
            if run.get("id"):
                try:
                    path = self._folder(run["id"]) / "output/progress.json"
                    if path.stat().st_size < 4000:
                        run["progress"] = json.loads(path.read_text())
                except (OSError, ValueError):
                    pass
            return {"active": self.active, "run": run,
                    "runtime": {"available": self.python.is_file(), "reason": "Uses the repository's RL Python environment in a separate process with private outputs; CPU hardware remains shared" if self.python.is_file() else "Repository RL virtual environment is missing"},
                    "catalog": {"changes": [{"id": key, **value} for key, value in CHANGES.items()],
                                "limits": {"maximum_wall_seconds": MAX_SECONDS, "maximum_seeds": 3, "maximum_rounds": 2, "cpu_threads": 1,
                                           "thread_limit_scope": "Torch and BLAS limits; JAX may use additional runtime/compilation threads"}}}

    def _source_files(self):
        paths = sorted((self.root / "hfmarl").rglob("*.py"))
        paths += [self.root / relative for relative in (
            "scripts/run_coupled_federation.py", "scripts/run_shadow_trial.py",
            "scripts/analyze_coupled_federation.py", "scripts/analyze_deposition_pilot.py",
            "scripts/stabilize_td3_heating.py", "pyproject.toml", REFERENCE)]
        if not paths or any(not p.is_file() or not p.resolve().is_relative_to(self.root) or p.is_symlink() for p in paths):
            raise ValueError("Shadow source or calibration is missing or redirected")
        return paths

    def prepare(self, request):
        config = validate_config(request)
        with self.lock:
            if self.active or self.lease.exists():
                raise ValueError("A shadow trial is already running")
            identifier = uuid.uuid4().hex
            folder = self._folder(identifier)
            folder.mkdir()
            source = folder / "source"
            source.mkdir()
            self.run = {"id": identifier, "status": "preparing", "config": config, "events": [], "comparison": None, "error": None}
            self._save()
            try:
                hashes = {}
                with zipfile.ZipFile(folder / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for original in self._source_files():
                        relative = original.relative_to(self.root).as_posix()
                        data = original.read_bytes()
                        if len(data) > 4_000_000:
                            raise ValueError("Shadow source exceeds its size bound")
                        hashes[relative] = digest(data)
                        destination = "calibration/reference.json" if relative == REFERENCE else relative
                        target = source / destination
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(data)
                        archive.writestr(destination, data)
                if any(digest((self.root / relative).read_bytes()) != sha for relative, sha in hashes.items()):
                    raise ValueError("Source changed during freezing; prepare a fresh trial")
                atomic_json(folder / "config.json", config)
                n, rounds = len(config["seeds"]), config["rounds"]
                budget = {"maximum_wall_seconds": MAX_SECONDS, "seeds": n, "rounds": rounds,
                    "training_episodes": n * (16 + 16 * rounds), "evaluation_episodes": n * (4 + 20 * rounds),
                    "total_episodes": n * (20 + 36 * rounds), "maximum_action_windows": 30 * n * (20 + 36 * rounds),
                    "scope": "Both configurations; donor pretraining, target learning, initial, gate and endpoint evaluations. Replay-only optimizer work is recorded separately. Existing calibration is additional sunk cost."}
                manifest = {"schema": "fusion-shadow-manifest-v1", "id": identifier, "config": config,
                            "expected_budget": budget,
                            "config_sha256": digest(json_bytes(config)), "source_hashes": hashes,
                            "source_zip_sha256": digest((folder / "source.zip").read_bytes()),
                            "calibration_sha256": hashes[REFERENCE],
                            "baseline": {key: spec["baseline"] for key, spec in CHANGES.items()},
                            "maximum_wall_seconds": MAX_SECONDS,
                            "isolation": "Fresh learners in a private source snapshot. Existing runs and checkpoints are never loaded or modified. CPU resources remain shared.",
                            "transport": "In-process numerical aggregation; Flower network transport is not exercised"}
                manifest["identity"] = digest(json_bytes(manifest))
                atomic_json(folder / "manifest.json", manifest)
                self.run.update(status="frozen", manifest=manifest, budget=budget)
                self._event("Source, calibration, baseline and candidate configuration frozen. Physics has not started.")
                self._save()
            except Exception:
                self.run.update(status="failed", error="Could not freeze the shadow source; partial files retained")
                self._save()
                raise
        return self.snapshot()

    def _verify(self, folder):
        manifest = json.loads((folder / "manifest.json").read_text())
        identity = manifest.pop("identity")
        if digest(json_bytes(manifest)) != identity or manifest != {k: v for k, v in self.run["manifest"].items() if k != "identity"}:
            raise ValueError("Frozen shadow manifest changed")
        if digest((folder / "config.json").read_bytes()) != manifest["config_sha256"]:
            raise ValueError("Frozen shadow configuration changed")
        if digest((folder / "source.zip").read_bytes()) != manifest["source_zip_sha256"]:
            raise ValueError("Frozen source archive changed")
        for relative, sha in manifest["source_hashes"].items():
            target = folder / "source" / ("calibration/reference.json" if relative == REFERENCE else relative)
            if not target.resolve().is_relative_to(folder / "source") or digest(target.read_bytes()) != sha:
                raise ValueError("Frozen worker source changed")

    def start(self, identifier):
        with self.lock:
            if self.active:
                raise ValueError("A shadow trial is already running")
            if self.run.get("id") != identifier or self.run["status"] != "frozen":
                raise ValueError("Select the current frozen trial; completed attempts cannot be replayed")
            folder = self._folder(identifier)
            self._verify(folder)
            if not self.python.is_file():
                raise ValueError("Repository RL Python environment is missing")
            self._claim_lease()
            self.run.update(status="launching", started_unix=time.time())
            self._save()  # Durable intent precedes spawn; uncertain recovery fails closed.
            output = folder / "output"
            output.mkdir()  # Never overwrite an earlier worker's output.
            environment = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PYTHONPATH", "PYTHONHOME"))}
            environment.update(PYTHONPATH=str(folder / "source"), PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                JAX_PLATFORMS="cpu", JAX_ENABLE_COMPILATION_CACHE="true", JAX_COMPILATION_CACHE_DIR=str(folder / "jax-cache"),
                CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
            command = [str(self.python), "-B", str(folder / "source/scripts/run_shadow_trial.py"),
                       "--config", str(folder / "config.json"), "--reference", str(folder / "source/calibration/reference.json"),
                       "--output", str(output), "--max-seconds", str(MAX_SECONDS)]
            self.active = True
            self.run.update(status="running", started_unix=time.time(), command_kind="frozen-shadow-worker")
            try:
                with (folder / "worker.log").open("wb") as log:
                    self.process = self._popen(command, cwd=folder / "source", env=environment, stdout=log,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.run.update(pid=self.process.pid, process_identity=process_identity(self.process.pid))
                self._event("Started isolated CPU TORAX worker; existing experiment execution is untouched.")
                self._save()
                threading.Thread(target=self._monitor, daemon=True).start()
            except Exception:  # noqa: BLE001 - preserve failed startup without leaking environment
                if self.process is not None and self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
                self.active = False
                self.run.update(status="failed", error="Shadow worker could not start; attempt retained")
                self._save()
                self._release_lease()
                raise ValueError(self.run["error"]) from None
        return self.snapshot()

    def _monitor(self):
        try:
            returncode = self.process.wait(timeout=MAX_SECONDS + 10)
            self._finish(returncode, "timed_out" if returncode == 124 else "failed")
        except subprocess.TimeoutExpired:
            self.process.kill()  # Only the owned worker; never a process-name kill.
            self.process.wait()
            self._finish(None, "timed_out")

    def _recover(self):
        if not self.run.get("process_identity"):
            with self.lock:
                self.run.update(status="recovery_required", error="Previous shadow launch has uncertain ownership. Its execution lease is retained to prevent duplicate work.")
                self._save()
            return
        while True:
            identity = process_identity(self.run.get("pid"))
            if not identity or identity != self.run.get("process_identity"):
                self._finish(None, "interrupted")
                return
            # The worker also has its own deadline. We do not signal a process
            # reconstructed from a PID after a server restart.
            time.sleep(1)

    def _finish(self, returncode, fallback="failed"):
        with self.lock:
            folder = self._folder(self.run["id"])
            if (folder / "output/timeout.json").is_file():
                fallback = "timed_out"
            try:
                self._verify(folder)
                result_path = folder / "output/result.json"
                if result_path.stat().st_size > 4_000_000:
                    raise ValueError("Shadow result exceeds the bounded summary size")
                raw = result_path.read_bytes()
                result = json.loads(raw)
                if result.get("config") != self.run["config"]:
                    raise ValueError("Shadow result does not match its frozen configuration")
                from .shadow_analysis import (
                    analyze_shadow_result,
                    attach_execution_status,
                )

                comparison = analyze_shadow_result(result)
                success = result.get("status") in ("complete", "completed") and comparison.get("status") == "complete" and (returncode == 0 or fallback == "interrupted")
                terminal_status = "completed" if success else fallback
                comparison = attach_execution_status(comparison, terminal_status)
                comparison["provenance"] = {"trial_id": self.run["id"], "manifest_identity": self.run["manifest"]["identity"],
                                            "result_sha256": digest(raw), "source_zip_sha256": self.run["manifest"]["source_zip_sha256"]}
                atomic_json(folder / "comparison.json", comparison)
                self.run["comparison"] = comparison
                self.run.update(status=terminal_status, returncode=returncode,
                                error=None if success else "Shadow trial incomplete; failures and partial results retained")
                self._event("Measured comparison and next-run recommendation saved; no policy was activated.")
            except (OSError, ValueError, KeyError, TypeError):
                self.run.update(status=fallback, error="Shadow worker did not produce a valid completed result; its log and partial files are retained")
            self.run["finished_unix"] = time.time()
            self.active = False
            self._save()
            self._release_lease()

    def export(self, identifier):
        folder = self._folder(identifier)
        with self.lock:
            if self.run.get("id") == identifier:
                return copy.deepcopy(self.run)
            try:
                path = folder / "run.json"
                if path.stat().st_size > 4_000_000:
                    raise ValueError("Shadow export is too large")
                return json.loads(path.read_text())
            except (OSError, ValueError):
                raise ValueError("Unknown shadow trial") from None

    def run_for_advisor(self, change, value, *, seed=17):
        """One caller's bounded, single-seed experiment, returning measured evidence."""
        config = validate_config({"change": change, "value": value, "seeds": [seed], "rounds": 1})
        with self.lock:
            frozen = self.prepare(config)
            identifier = frozen["run"]["id"]
            self.start(identifier)
        while True:
            with self.lock:
                run = self.export(identifier)
            if run["status"] != "running":
                evidence = {"schema": "fusion-advisor-shadow-evidence-v1", "trial_id": identifier,
                        "status": run["status"], "config": run["config"], "budget": run["budget"],
                        "manifest_identity": run["manifest"]["identity"], "comparison": run.get("comparison"),
                        "error": run.get("error"), "policy_activated": False}
                if run.get("comparison"):
                    from .shadow_telemetry import clipping_diagnostic_view

                    try:
                        path = self._folder(identifier) / "output/result.json"
                        if path.stat().st_size > 4_000_000:
                            raise ValueError("Shadow result exceeds the bounded summary size")
                        raw = path.read_bytes()
                        raw_hash = digest(raw)
                        if raw_hash != run["comparison"]["provenance"]["result_sha256"]:
                            raise ValueError("Shadow result changed after comparison")
                        evidence["round_aggregation_diagnostics"] = clipping_diagnostic_view(json.loads(raw), raw_hash)
                    except (OSError, ValueError, KeyError, TypeError):
                        evidence["round_aggregation_diagnostics"] = {
                            "schema": "fusion-shadow-clipping-diagnostic-v1",
                            "status": "unavailable_unverified_result",
                            "scope": "Clipping diagnostics withheld: saved worker bytes could not be verified against the completed comparison.",
                        }
                return evidence
            time.sleep(.5)
