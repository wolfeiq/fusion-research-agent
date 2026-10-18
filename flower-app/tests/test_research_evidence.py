import hashlib
import json
from pathlib import Path

import pytest

from fusion_agent.research.evidence import (
    EvidenceCatalog,
    EvidenceError,
    resolve_pointer,
)

SUMMARY = "results/td3_heating_stability/seed29/independent_analysis.json"


def put(root, path=SUMMARY, data=None, text=None):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        text if text is not None else json.dumps(data or {"status": "incomplete"}),
        encoding="utf-8",
    )
    return target


def test_allowlist_does_not_discover_raw_models_secrets_or_random_docs(tmp_path):
    put(tmp_path)
    for path in [".env", "docs/private.md", "results/random/summary.json",
                 "results/td3_heating_stability/seed29/model.pkl",
                 "results/td3_heating_stability/seed29/raw-case0.json"]:
        put(tmp_path, path, text="DO NOT SEND")
    records = EvidenceCatalog(tmp_path).list_records()
    assert [r["id"] for r in records] == [SUMMARY]
    assert records[0]["status"] == "incomplete"


@pytest.mark.parametrize("path", [
    "../docs/coldstart_coupled_federation.md", "/docs/coldstart_coupled_federation.md",
    "C:/secret.json", "results/../secret.json", "docs\\coldstart_coupled_federation.md",
    "results/td3_heating_stability/seed29/model.pkl", "docs/private.md",
])
def test_reject_arbitrary_paths(tmp_path, path):
    with pytest.raises(EvidenceError, match="allowlist"):
        EvidenceCatalog(tmp_path).snapshot(path)


def test_snapshot_hashes_exact_returned_content_and_original_bytes(tmp_path):
    path = put(tmp_path, data={
        "status": "complete_verified", "mean_cost": 0.12,
        "independent_audits": [str(tmp_path / SUMMARY)],
        "trace": ["RAW"], "api_key": "SECRET",
    })
    raw = path.read_bytes()
    snapshot = EvidenceCatalog(tmp_path).snapshot(SUMMARY)
    assert snapshot["sha256"] == hashlib.sha256(snapshot["content"].encode()).hexdigest()
    assert snapshot["metadata"]["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert snapshot["metadata"]["status"] == "complete_verified"
    assert "not freshly audited" in snapshot["metadata"]["verification_scope"]
    assert "RAW" not in snapshot["content"] and "SECRET" not in snapshot["content"]
    assert str(tmp_path) not in snapshot["content"]
    assert resolve_pointer(snapshot["content"], "/independent_audits/0") == SUMMARY
    assert set(snapshot["metadata"]["omitted_json_pointers"]) == {"/trace", "/api_key"}


@pytest.mark.parametrize("status", ["incomplete", "failed", "running", "completed"])
def test_preserves_reported_status(tmp_path, status):
    put(tmp_path, data={"overall_status": status})
    assert EvidenceCatalog(tmp_path).snapshot(SUMMARY)["metadata"]["status"] == status


def test_missing_status_is_unknown_not_complete(tmp_path):
    put(tmp_path, data={"comparisons": [{"pass": True}]})
    assert EvidenceCatalog(tmp_path).list_records()[0]["status"] == "unknown"


@pytest.mark.parametrize("content", ['{', '[]', '{"status":"complete","status":"failed"}', '{"cost":NaN}', '{"cost":1e999}'])
def test_malformed_is_invalid_and_cannot_be_snapshotted(tmp_path, content):
    put(tmp_path, text=content)
    catalog = EvidenceCatalog(tmp_path)
    assert catalog.list_records()[0]["status"] == "invalid"
    with pytest.raises(EvidenceError):
        catalog.snapshot(SUMMARY)


def test_missing_file(tmp_path):
    with pytest.raises(EvidenceError, match="missing"):
        EvidenceCatalog(tmp_path).snapshot(SUMMARY)


def test_changed_bytes_rejected_even_with_same_stat(tmp_path, monkeypatch):
    target = put(tmp_path)
    original = Path.read_bytes
    count = 0

    def changing(path):
        nonlocal count
        result = original(path)
        if path == target:
            count += 1
            if count == 2:
                return b'{"status":"complete"}'
        return result

    monkeypatch.setattr(Path, "read_bytes", changing)
    with pytest.raises(EvidenceError, match="changed"):
        EvidenceCatalog(tmp_path).snapshot(SUMMARY)


def test_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    put(outside, "seed29/independent_analysis.json", data={"secret": "NO"})
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    try:
        (root / "results/td3_heating_stability").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Host does not grant symlink creation")
    with pytest.raises(EvidenceError, match="Symlinks"):
        EvidenceCatalog(root).snapshot(SUMMARY)


def test_symlink_policy_without_host_symlink_privilege(tmp_path, monkeypatch):
    put(tmp_path)
    original = Path.is_symlink
    forbidden = tmp_path / "results/td3_heating_stability"
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path == forbidden or original(path),
    )
    with pytest.raises(EvidenceError, match="Symlinks"):
        EvidenceCatalog(tmp_path).snapshot(SUMMARY)


def test_local_paths_are_redacted_in_protocol(tmp_path):
    name = "docs/td3_heating_stability.md"
    put(tmp_path, name, text="Evidence at C:\\Users\\researcher\\secret.json and /home/me/result.")
    snapshot = EvidenceCatalog(tmp_path).snapshot(name)
    assert snapshot["kind"] == "protocol"
    assert snapshot["metadata"]["status"] == "protocol"
    assert "researcher" not in snapshot["content"]
    assert "/home/me" not in snapshot["content"]


def test_large_analyses_project_explicitly_without_changing_retained_pointers(tmp_path):
    put(tmp_path, data={"status": "incomplete", "jobs": {"secretly_large": "x" * 100000},
                        "completed_job_episodes": 906, "cold_target_findings": {"pass": False}})
    snapshot = EvidenceCatalog(tmp_path).snapshot(SUMMARY)
    assert snapshot["metadata"]["omitted_json_pointers"] == ["/jobs"]
    assert resolve_pointer(snapshot["content"], "/completed_job_episodes") == 906


def test_content_limit_rejects_instead_of_truncating(tmp_path):
    put(tmp_path, data={"status": "incomplete", "note": "x" * 40000})
    with pytest.raises(EvidenceError, match="content limit"):
        EvidenceCatalog(tmp_path).snapshot(SUMMARY)


def test_source_byte_limit_before_read(tmp_path, monkeypatch):
    import fusion_agent.research.evidence as module
    put(tmp_path)
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 2)
    with pytest.raises(EvidenceError, match="byte limit"):
        EvidenceCatalog(tmp_path).snapshot(SUMMARY)


def test_record_limit_is_explicit(tmp_path, monkeypatch):
    import fusion_agent.research.evidence as module
    put(tmp_path)
    put(tmp_path, "results/td3_heating_stability/seed17/independent_analysis.json")
    monkeypatch.setattr(module, "MAX_RECORDS", 1)
    with pytest.raises(EvidenceError, match="record limit"):
        EvidenceCatalog(tmp_path).list_records()


def test_pointer_exact_array_and_escaped_keys():
    content = '{"a/b":{"~key":[1,2]},"":3}'
    assert resolve_pointer(content, "/a~1b/~0key/1") == 2
    assert resolve_pointer(content, "/") == 3
    assert resolve_pointer(content, "")["a/b"]["~key"] == [1, 2]


@pytest.mark.parametrize("pointer", ["a", "/x/-", "/x/01", "/x/-1", "/x/4", "/x/~2", "/missing"])
def test_bad_pointers(pointer):
    with pytest.raises(EvidenceError):
        resolve_pointer({"x": [1, 2]}, pointer)
