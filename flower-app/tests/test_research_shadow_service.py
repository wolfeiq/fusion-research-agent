"""Shadow execution boundary: immutable inputs and an owned isolated process."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pytest

from fusion_agent.research.shadow import REFERENCE, ShadowService, validate_config

REQUEST = {"change": "td3_actor_lr", "value": .00005, "seeds": [17], "rounds": 1}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    for relative in ["hfmarl/__init__.py", "hfmarl/agents/local.py", "scripts/run_shadow_trial.py",
                     "scripts/run_coupled_federation.py", "scripts/analyze_coupled_federation.py",
                     "scripts/analyze_deposition_pilot.py", "scripts/stabilize_td3_heating.py", "pyproject.toml", REFERENCE]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if relative.endswith(".json") else "# frozen fixture\n")
    (root / "results/current-running").mkdir()
    (root / "results/current-running/checkpoint.bin").write_bytes(b"live checkpoint")
    return root


def service(repo, tmp_path, **kwargs):
    interpreter = tmp_path / "python.exe"
    interpreter.write_bytes(b"test stub")
    return ShadowService(repo, tmp_path / "shadows", python=interpreter, **kwargs)


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def finished(svc):
    stop = time.monotonic() + 3
    while svc.active and time.monotonic() < stop:
        time.sleep(.005)
    assert not svc.active
    return svc.snapshot()["run"]


@pytest.mark.parametrize("override", [
    {"change": "__import__"}, {"value": float("nan")}, {"value": float("inf")},
    {"value": True}, {"value": 0}, {"value": .0001}, {"value": .1},
    {"seeds": [17, 17]}, {"seeds": [True]}, {"seeds": []}, {"seeds": [1, 2, 3, 4]},
    {"seeds": [-1]}, {"rounds": 0}, {"rounds": True}, {"rounds": 3},
    {"command": "anything"}, {"output": "../current"}, {"schema": "wrong"},
])
def test_reject_unbounded_or_noop_requests(override):
    with pytest.raises(ValueError):
        validate_config({**REQUEST, **override})


def test_freeze_preserves_original_sources_and_checkpoints(repo, tmp_path):
    before = files(repo)
    svc = service(repo, tmp_path)
    run = svc.prepare(REQUEST)["run"]
    assert run["status"] == "frozen"
    assert files(repo) == before
    frozen = svc.state_dir / run["id"]
    assert (frozen / "source/calibration/reference.json").read_bytes() == (repo / REFERENCE).read_bytes()
    assert (frozen / "source.zip").is_file()
    # Subsequent live edits do not leak into an already frozen shadow.
    (repo / "hfmarl/agents/local.py").write_text("# later unrelated edit")
    svc._verify(frozen)
    assert (frozen / "source/hfmarl/agents/local.py").read_text() == "# frozen fixture\n"


@pytest.mark.parametrize("relative", ["config.json", "manifest.json", "source.zip", "source/hfmarl/agents/local.py"])
def test_refuses_tampered_frozen_inputs(repo, tmp_path, relative):
    svc = service(repo, tmp_path)
    run = svc.prepare(REQUEST)["run"]
    (svc.state_dir / run["id"] / relative).write_bytes(b"{}")
    with pytest.raises((ValueError, KeyError)):
        svc.start(run["id"])
    assert not svc.active


def test_rejects_arbitrary_trial_paths(repo, tmp_path):
    svc = service(repo, tmp_path)
    for identifier in (None, "../live", "z" * 32, "a" * 31):
        with pytest.raises(ValueError):
            svc.export(identifier)


@pytest.mark.parametrize("tampered", [False, True])
def test_advisor_return_projects_only_hash_verified_worker_telemetry(repo, tmp_path, monkeypatch, tampered):
    svc = service(repo, tmp_path)
    run = svc.prepare(REQUEST)["run"]
    folder = svc.state_dir / run["id"]
    (folder / "output").mkdir()
    raw = json.dumps({"baseline": {"seeds": []}, "candidate": {"seeds": []}}).encode()
    path = folder / "output/result.json"
    path.write_bytes(raw + (b" " if tampered else b""))
    run.update(status="completed", comparison={"provenance": {"result_sha256": hashlib.sha256(raw).hexdigest()}})
    before = files(folder)
    monkeypatch.setattr(svc, "prepare", lambda request: {"run": run})
    monkeypatch.setattr(svc, "start", lambda identifier: None)
    monkeypatch.setattr(svc, "export", lambda identifier: copy.deepcopy(run))
    evidence = svc.run_for_advisor("td3_actor_lr", .00005)
    view = evidence["round_aggregation_diagnostics"]
    assert view["status"] == ("unavailable_unverified_result" if tampered else "partial_or_unavailable")
    if not tampered:
        assert view["raw_result_sha256"] == hashlib.sha256(raw).hexdigest()
    assert evidence["comparison"] == run["comparison"]
    assert evidence["policy_activated"] is False
    assert files(folder) == before


def test_frozen_worker_has_own_cwd_outputs_secrets_removed_and_result_provenance(repo, tmp_path, monkeypatch):
    before = files(repo)
    calls = []
    monkeypatch.setenv("NEBIUS_API_KEY", "must-not-reach-physics")
    monkeypatch.setenv("TAVILY_API_KEY", "also-private")
    from fusion_agent.research import shadow_analysis
    monkeypatch.setattr(shadow_analysis, "analyze_shadow_result", lambda result: {"schema": "fusion-shadow-comparison-v1", "status": "complete", "recommendation": {"code": "repeat_paired_seeds"}})

    class Worker:
        pid = 0

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            output = Path(command[command.index("--output") + 1])
            config = json.loads(Path(command[command.index("--config") + 1]).read_text())
            (output / "result.json").write_text(json.dumps({"status": "complete", "config": config}))

        def wait(self, timeout=None):
            return 0

    svc = service(repo, tmp_path, popen=Worker)
    identifier = svc.prepare(REQUEST)["run"]["id"]
    svc.start(identifier)
    run = finished(svc)
    assert run["status"] == "completed"
    _command, options = calls[0]
    assert options["cwd"] == svc.state_dir / identifier / "source"
    assert not options.get("shell")
    assert "NEBIUS_API_KEY" not in options["env"] and "TAVILY_API_KEY" not in options["env"]
    assert options["env"]["OMP_NUM_THREADS"] == "1"
    assert options["env"]["PYTHONPATH"] == str(options["cwd"])
    assert run["comparison"]["provenance"]["manifest_identity"] == run["manifest"]["identity"]
    assert files(repo) == before
    with pytest.raises(ValueError, match="cannot be replayed"):
        svc.start(identifier)


def test_start_failure_and_timeout_preserve_attempt(repo, tmp_path):
    def failure(*args, **kwargs):
        raise OSError("fixture failure")

    svc = service(repo, tmp_path, popen=failure)
    identifier = svc.prepare(REQUEST)["run"]["id"]
    with pytest.raises(ValueError, match="could not start"):
        svc.start(identifier)
    assert not svc.active and svc.run["status"] == "failed"
    assert (svc.state_dir / identifier / "source.zip").is_file()

    class Worker:
        pid = 0
        killed = False

        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("owned-worker", timeout)
            return 1

        def kill(self):
            self.killed = True

    svc = service(repo, tmp_path, popen=Worker)
    identifier = svc.prepare(REQUEST)["run"]["id"]
    svc.start(identifier)
    assert finished(svc)["status"] == "timed_out"
    assert svc.process.killed


def test_active_trial_blocks_prepare_and_duplicate_start(repo, tmp_path):
    svc = service(repo, tmp_path)
    identifier = svc.prepare(REQUEST)["run"]["id"]
    before = copy.deepcopy(svc.run)
    svc.active = True
    for call in (lambda: svc.prepare(REQUEST), lambda: svc.start(identifier)):
        with pytest.raises(ValueError, match="already running"):
            call()
    assert svc.run == before


def test_cross_instance_lease_is_exclusive_and_uncertain_launch_fails_closed(repo, tmp_path):
    first = service(repo, tmp_path)
    identifier = first.prepare(REQUEST)["run"]["id"]
    first._claim_lease()  # Crash immediately after durable intent, before Popen.
    with pytest.raises(ValueError, match="execution lease"):
        first._claim_lease()
    second = service(repo, tmp_path)
    deadline = time.monotonic() + 2
    while second.run["status"] != "recovery_required" and time.monotonic() < deadline:
        time.sleep(.005)
    assert second.active and second.run["status"] == "recovery_required"
    assert second.run["id"] == identifier
    with pytest.raises(ValueError, match="already running"):
        second.prepare(REQUEST)
    assert second.lease.is_file()


def test_worker_watchdog_exit_keeps_timeout_classification(repo, tmp_path):
    class Worker:
        pid = 0

        def __init__(self, command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            (output / "timeout.json").write_text('{"type":"TimeoutError"}')

        def wait(self, timeout=None):
            return 124

    svc = service(repo, tmp_path, popen=Worker)
    identifier = svc.prepare(REQUEST)["run"]["id"]
    svc.start(identifier)
    assert finished(svc)["status"] == "timed_out"
    assert not svc.lease.exists()
