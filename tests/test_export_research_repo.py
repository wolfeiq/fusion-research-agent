"""Filesystem safety checks for source export; all fixture runs are synthetic."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import export_research_repo as exporter


@pytest.fixture
def roots(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    return source, tmp_path / "fusion-research-agent"


def test_explicit_export_preserves_source_and_writes_owned_manifest(roots):
    source, destination = roots
    original = source / "untouched.txt"
    original.write_text("existing experiment")
    payload = {"README.md": b"readme", "flower-app/file.py": b"pass\n"}
    report = exporter.write_export(source, destination, payload, {"original.py": "a" * 64})
    assert report["files"] == 2 and not report["dry_run"]
    manifest = json.loads((destination / exporter.MANIFEST).read_text())
    assert manifest["files"] == {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    assert (destination / "flower-app/file.py").read_bytes() == b"pass\n"
    assert original.read_text() == "existing experiment"
    assert list(source.iterdir()) == [original]


def test_dry_run_does_not_create_destination(roots):
    source, destination = roots
    report = exporter.write_export(source, destination, {"README.md": b"readme"}, {}, dry_run=True)
    assert report["dry_run"] and not destination.exists()


def test_subsequent_export_can_update_only_unchanged_owned_files(roots):
    source, destination = roots
    exporter.write_export(source, destination, {"README.md": b"old", "retained.py": b"first"}, {})
    user_file = destination / "user-notes.txt"
    user_file.write_text("keep me")
    exporter.write_export(source, destination, {"README.md": b"new"}, {})
    assert (destination / "README.md").read_bytes() == b"new"
    assert (destination / "retained.py").read_bytes() == b"first"
    assert user_file.read_text() == "keep me"


def test_unknown_existing_file_blocks_whole_export_before_writes(roots):
    source, destination = roots
    destination.mkdir()
    (destination / "README.md").write_bytes(b"same bytes still unowned")
    with pytest.raises(ValueError, match="not owned"):
        exporter.write_export(source, destination, {
            "first.py": b"would be new", "README.md": b"same bytes still unowned",
        }, {})
    assert not (destination / "first.py").exists()
    assert not (destination / exporter.MANIFEST).exists()


def test_modified_owned_file_blocks_reexport(roots):
    source, destination = roots
    exporter.write_export(source, destination, {"README.md": b"old"}, {})
    (destination / "README.md").write_bytes(b"user edit")
    with pytest.raises(ValueError, match="modified exported"):
        exporter.write_export(source, destination, {"README.md": b"new"}, {})
    assert (destination / "README.md").read_bytes() == b"user edit"


@pytest.mark.parametrize("name", ["../escape.py", "/outside.py", "C:/outside.py", "a\\b.py", "a//b.py"])
def test_noncanonical_payload_paths_rejected_before_write(roots, name):
    source, destination = roots
    with pytest.raises(ValueError, match="relative paths"):
        exporter.write_export(source, destination, {name: b"data"}, {})
    assert not destination.exists()


def test_rejects_wrong_or_nested_destination(roots):
    source, _ = roots
    for destination in (source, source / "fusion-research-agent", source.parent / "wrong-name"):
        with pytest.raises(ValueError, match="separate sibling"):
            exporter.write_export(source, destination, {"README.md": b"test"}, {})


def test_rejects_redirected_destination_component(roots, monkeypatch):
    source, destination = roots
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == destination / "flower-app" or original(path))
    with pytest.raises(ValueError, match="link or junction"):
        exporter.write_export(source, destination, {"flower-app/main.py": b"test"}, {})


def test_unknown_manifest_is_not_overwritten(roots):
    source, destination = roots
    destination.mkdir()
    (destination / exporter.MANIFEST).write_text('{"schema":"user-data"}')
    with pytest.raises(ValueError, match="not recognized"):
        exporter.write_export(source, destination, {"README.md": b"test"}, {})


def test_payload_is_exact_allowlist_and_only_selected_terminal_evidence(roots, monkeypatch):
    source, _ = roots
    for constant in ("ROOT_SCRIPTS", "ROOT_TESTS", "PROTOCOLS", "APP_FILES", "APP_SCRIPTS"):
        monkeypatch.setattr(exporter, constant, ())

    def put(name, data=b"safe public source"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    for name in (
        "pyproject.toml", "flower-app/THERMAL_LICENSE.md",
        "flower-app/fusion_agent/thermal/_devices.json",
        "results/deposition_calibration_verified/reference.json",
        "hfmarl/example.py", "flower-app/fusion_agent/example.py",
        "flower-app/frontend/research.js", "flower-app/tests/test_example.py",
    ):
        put(name)
    for name in ("diiid_like", "iter_like", "sparc_like", "tcv_like"):
        put(f"assets/3d/html/{name}.html")
    for identifier in exporter.PILOTS:
        prefix = f"flower-app/.dashboard/shadow/{identifier}/"
        for name in exporter.BUNDLE_FILES:
            put(prefix + name, b"{}")
        put(prefix + "run.json", json.dumps({"id": identifier, "status": "completed", "config": {}}).encode())
        put(prefix + "worker.log", b"PRIVATE_SENTINEL")
    for name in (
        "flower-app/.env.research", "flower-app/.dashboard/credentials.json",
        "flower-app/fusion_agent/thermal/_fixtures.json", "hfmarl/__pycache__/private.py",
        "flower-app/.venv/private.py", "results/arbitrary.json", ".git/config",
    ):
        put(name, b"PRIVATE_SENTINEL")
    payload, _ = exporter.build_payload(source)
    assert "hfmarl/example.py" in payload
    assert "flower-app/research.env.example" in payload
    assert "PRIVATE_SENTINEL" not in b"".join(payload.values()).decode()
    assert not any(".dashboard" in name or ".env.research" in name for name in payload)
    assert sum(name.endswith("/source.zip") for name in payload) == 2
    assert payload["results/deposition_calibration_verified/reference.json"] == b"safe public source"


def test_payload_requires_declared_pilot_to_be_completed(roots, monkeypatch):
    source, _ = roots
    for constant in ("ROOT_SCRIPTS", "ROOT_TESTS", "PROTOCOLS", "APP_FILES", "APP_SCRIPTS"):
        monkeypatch.setattr(exporter, constant, ())
    for name in (
        "pyproject.toml", "flower-app/THERMAL_LICENSE.md",
        "flower-app/fusion_agent/thermal/_devices.json",
        "results/deposition_calibration_verified/reference.json",
        *(f"assets/3d/html/{device}.html" for device in ("diiid_like", "iter_like", "sparc_like", "tcv_like")),
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public")
    path = source / f"flower-app/.dashboard/shadow/{exporter.PILOTS[0]}/run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": exporter.PILOTS[0], "status": "running"}))
    with pytest.raises(ValueError, match="completed pilot"):
        exporter.build_payload(source)
