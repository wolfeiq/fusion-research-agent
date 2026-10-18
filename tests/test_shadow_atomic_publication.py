"""Progress publication survives short Windows locks without hiding real failures."""

import errno
import json

import pytest

from scripts import run_shadow_trial as worker


def windows_permission(code):
    error = PermissionError(errno.EACCES, "Simulated Windows publication conflict")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_locks_keep_old_json_until_complete_atomic_replacement(tmp_path, monkeypatch, code):
    path = tmp_path / "progress.json"
    old = '{"completed_episodes": 7}'
    path.write_text(old, encoding="utf-8")
    payload = {"completed_episodes": 8, "phase": "baseline/seed-17/donor_warm/train-1"}
    replace = worker.os.replace
    attempts, sleeps = [], []

    def interrupted(temporary, destination):
        attempts.append((temporary, destination))
        assert path.read_text(encoding="utf-8") == old
        assert json.loads(temporary.read_text(encoding="utf-8")) == payload
        if len(attempts) <= 2:
            raise windows_permission(code)
        replace(temporary, destination)

    monkeypatch.setattr(worker.os, "replace", interrupted)
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    worker._write(path, payload)
    assert len(attempts) == 3
    assert sleeps == [.01, .02]
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not path.with_name("progress.json.tmp").exists()


def test_persistent_windows_error_exhausts_short_budget_and_preserves_both_files(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"completed_episodes": 7}', encoding="utf-8")
    error = windows_permission(5)
    attempts, sleeps = [], []

    def blocked(*args):
        attempts.append(args)
        raise error

    monkeypatch.setattr(worker.os, "replace", blocked)
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        worker._write(path, {"completed_episodes": 8})
    assert caught.value is error
    assert len(attempts) == 6
    assert sleeps == list(worker._REPLACE_RETRY_DELAYS)
    assert sum(sleeps) == pytest.approx(.31)
    assert json.loads(path.read_text(encoding="utf-8")) == {"completed_episodes": 7}
    assert json.loads(path.with_name("progress.json.tmp").read_text(encoding="utf-8")) == {"completed_episodes": 8}


@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "Ordinary non-Windows permission error"),
    windows_permission(1), FileNotFoundError("Missing parent"),
    IsADirectoryError("Destination is a directory"), OSError(errno.ENOSPC, "Full disk"),
])
def test_unrelated_errors_are_not_retried_or_reclassified(tmp_path, monkeypatch, error):
    calls, sleeps = [], []

    def fail(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(worker.os, "replace", fail)
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    with pytest.raises(type(error)) as caught:
        worker._write(tmp_path / "progress.json", {"completed_episodes": 7})
    assert caught.value is error
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("deadline,oversleep,expected_attempts,expected_sleeps", [
    (10., 0., 1, []), (10.025, 0., 2, [.01]), (10.025, .1, 1, [.01]),
])
def test_retry_never_renews_or_crosses_existing_worker_deadline(
        tmp_path, monkeypatch, deadline, oversleep, expected_attempts, expected_sleeps):
    error = windows_permission(32)
    calls, sleeps, clock = [], [], [10.]

    def blocked(*args):
        calls.append(args)
        raise error

    def sleep(delay):
        sleeps.append(delay)
        clock[0] += delay + oversleep

    monkeypatch.setattr(worker.os, "replace", blocked)
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(worker.time, "sleep", sleep)
    with pytest.raises(PermissionError) as caught:
        worker._write(tmp_path / "progress.json", {"completed_episodes": 7}, retry_deadline=deadline)
    assert caught.value is error
    assert len(calls) == expected_attempts
    assert sleeps == expected_sleeps


def test_serialization_failure_is_not_retried_or_published(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"completed_episodes": 7}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(worker.os, "replace", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="JSON compliant"):
        worker._write(path, {"cost": float("nan")})
    assert calls == []
    assert json.loads(path.read_text(encoding="utf-8")) == {"completed_episodes": 7}
