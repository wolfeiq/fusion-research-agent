"""Safe failure observability and bounded Windows state-publication retries."""

import json
import time
from pathlib import Path

import pytest

from fusion_agent.research import service as service_module
from fusion_agent.research.providers import Settings
from fusion_agent.research.service import (
    ResearchService,
    _failure_diagnostic,
    _replace_state_file,
)


def permission_error(winerror):
    error = PermissionError(13, "Sensitive payload and path", "C:/private/secret")
    if winerror is not None:
        error.winerror = winerror
    return error


def test_diagnostic_excludes_message_paths_source_and_locals():
    secret = "credentials-must-never-appear"
    namespace = {}
    exec(compile("def fail():\n    secret_local = 'credentials-must-never-appear'\n    raise KeyError(secret_local)\n",  # noqa: S102 - fixed synthetic traceback filename
                 "C:/private/credentials-must-never-appear/provider.py", "exec"), namespace)
    try:
        namespace["fail"]()
    except KeyError as exc:
        diagnostic = _failure_diagnostic(exc, (secret,))
    serialized = json.dumps(diagnostic)
    assert diagnostic["exception_type"] == "KeyError"
    assert diagnostic["frames"][-1] == {"file": "provider.py", "function": "fail", "line": 3}
    assert secret not in serialized
    assert "C:/private" not in serialized
    assert "secret_local" not in serialized
    assert all(set(frame) == {"file", "function", "line"} for frame in diagnostic["frames"])


def test_diagnostic_redacts_sensitive_basename_and_keeps_numeric_os_codes():
    secret = "credential"
    namespace = {"error": permission_error(32)}
    try:
        exec(compile("raise error", "C:/private/credential.py", "exec"), namespace)  # noqa: S102 - fixed synthetic traceback filename
    except PermissionError as exc:
        diagnostic = _failure_diagnostic(exc, (secret,))
    assert diagnostic["frames"][-1]["file"] == "<redacted>"
    assert diagnostic["errno"] == 13
    assert diagnostic["winerror"] == 32
    assert secret not in json.dumps(diagnostic)


def test_diagnostic_caps_traceback_and_does_not_evaluate_exception_str():
    class PrivateError(Exception):
        def __str__(self):  # noqa: PLE0307 - deliberately fail if diagnostics format the exception
            pytest.fail("Diagnostic must not format the exception")

    def recurse(depth):
        if depth:
            return recurse(depth - 1)
        raise PrivateError()

    try:
        recurse(30)
    except PrivateError as exc:
        diagnostic = _failure_diagnostic(exc)
    assert len(diagnostic["frames"]) == 12
    assert diagnostic["exception_type"] == "PrivateError"


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_lock_retries_only_replace_then_publishes(tmp_path, monkeypatch, code):
    temporary, target = tmp_path / "run.tmp", tmp_path / "run.json"
    temporary.write_text("new complete state", encoding="utf-8")
    target.write_text("old complete state", encoding="utf-8")
    actual_replace = service_module.os.replace
    calls, sleeps = [], []

    def replace(src, dst):
        calls.append((src, dst))
        assert target.read_text(encoding="utf-8") == "old complete state"
        assert temporary.read_text(encoding="utf-8") == "new complete state"
        if len(calls) < 3:
            raise permission_error(code)
        return actual_replace(src, dst)

    monkeypatch.setattr(service_module.os, "replace", replace)
    monkeypatch.setattr(service_module.time, "sleep", sleeps.append)
    _replace_state_file(temporary, target)
    assert len(calls) == 3
    assert sleeps == [.01, .02]
    assert target.read_text(encoding="utf-8") == "new complete state"
    assert not temporary.exists()


def test_permanent_windows_lock_preserves_both_files_and_propagates_same_error(tmp_path, monkeypatch):
    temporary, target = tmp_path / "run.tmp", tmp_path / "run.json"
    temporary.write_text("new state", encoding="utf-8")
    target.write_text("old state", encoding="utf-8")
    error, calls, sleeps = permission_error(5), [], []

    def replace(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(service_module.os, "replace", replace)
    monkeypatch.setattr(service_module.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        _replace_state_file(temporary, target)
    assert caught.value is error
    assert len(calls) == 6 and sum(sleeps) == pytest.approx(.31)
    assert temporary.read_text(encoding="utf-8") == "new state"
    assert target.read_text(encoding="utf-8") == "old state"


@pytest.mark.parametrize("error", [permission_error(None), permission_error(87), OSError("not a reader lock")])
def test_unrecognized_failure_is_never_retried(monkeypatch, error):
    calls, sleeps = [], []

    def replace(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(service_module.os, "replace", replace)
    monkeypatch.setattr(service_module.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as caught:
        _replace_state_file(Path("run.tmp"), Path("run.json"))
    assert caught.value is error and len(calls) == 1 and sleeps == []


def test_unknown_worker_failure_diagnostic_is_persisted_without_secret(tmp_path, monkeypatch):
    record_id = "results/td3_heating_stability/fixed_partner_comparison.json"
    repo = tmp_path / "repo"
    record = repo / record_id
    record.parent.mkdir(parents=True)
    record.write_text('{"status":"complete","cost":0.12}', encoding="utf-8")
    secret = "test-key-never-published"
    settings = Settings(nebius_key=secret, tavily_key="test-tavily-never-published")

    def broken_inspection(_sources):
        raise KeyError(secret)

    monkeypatch.setattr(service_module, "offline_report", broken_inspection)
    service = ResearchService(repo, tmp_path / "state", settings_factory=lambda: settings)
    service.start("Inspect saved evidence", [record_id], "offline")
    deadline = time.monotonic() + 3
    while service.active and time.monotonic() < deadline:
        time.sleep(.005)
    assert not service.active
    run = service.export(service.run["id"])
    persisted = json.loads((service.state_dir / run["id"] / "run.json").read_text(encoding="utf-8"))
    diagnostic = run["failure_diagnostic"]
    assert run["status"] == "failed"
    assert diagnostic["exception_type"] == "KeyError"
    assert diagnostic["frames"][-1]["function"] == "broken_inspection"
    assert run["events"][-1]["kind"] == "failure_diagnostic"
    assert persisted["failure_diagnostic"] == diagnostic
    assert persisted["events"][-1]["diagnostic"] == diagnostic
    assert secret not in json.dumps([run, persisted, service.snapshot()])


def test_save_retries_run_and_latest_pointer_publications(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    service = ResearchService(repo, tmp_path / "state", settings_factory=Settings)
    service.run.update(id="a" * 32, status="running")
    actual = service_module.os.replace
    attempts = {}
    monkeypatch.setattr(service_module.time, "sleep", lambda _delay: None)

    def replace(src, dst):
        name = Path(dst).name
        attempts[name] = attempts.get(name, 0) + 1
        if attempts[name] == 1:
            raise permission_error(32)
        return actual(src, dst)

    monkeypatch.setattr(service_module.os, "replace", replace)
    service._save()
    assert attempts == {"run.json": 2, "latest.json": 2}
    assert json.loads((service.state_dir / "latest.json").read_text())["id"] == "a" * 32
    assert json.loads((service.state_dir / ("a" * 32) / "run.json").read_text())["status"] == "running"
