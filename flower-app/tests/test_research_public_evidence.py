"""Published starter bundles use the same verification as private shadow output.

Fixtures contain synthetic summaries and are not experimental evidence.
"""

import json
import shutil

import pytest

from fusion_agent.research.evidence import EvidenceCatalog, EvidenceError

pytest_plugins = ["test_research_shadow_analysis"]


@pytest.fixture
def published_shadow(saved_shadow):
    root, folder, relative = saved_shadow
    published = relative.replace("flower-app/.dashboard/shadow/", "examples/evidence/shadow/")
    destination = root / published.rsplit("/", 1)[0]
    shutil.copytree(folder, destination)
    return root, destination, published


def test_both_shadow_prefixes_verify_identical_content_without_writes(saved_shadow, published_shadow):
    root, private_folder, private_id = saved_shadow
    _, public_folder, public_id = published_shadow
    before = {
        path: path.read_bytes()
        for folder in (private_folder, public_folder)
        for path in folder.rglob("*") if path.is_file()
    }
    catalog = EvidenceCatalog(root)
    private, public = catalog.snapshot(private_id), catalog.snapshot(public_id)
    assert private["content"] == public["content"]
    assert private["sha256"] == public["sha256"]
    assert private["metadata"] == public["metadata"]
    assert public["id"] == public["uri"] == public_id
    assert public["metadata"]["status_basis"] == "hash_bound_worker_summary_and_recomputed_comparison"
    assert {path: path.read_bytes() for path in before} == before
    records = catalog.list_records()
    assert {record["id"] for record in records} == {private_id, public_id}
    assert all(record["adapter"] == "shadow_comparison" for record in records)
    assert all(record["status"] == "complete" for record in records)


@pytest.mark.parametrize("path", [
    "examples/evidence/shadow/../comparison.json",
    "examples/evidence/shadow/" + "a" * 31 + "/comparison.json",
    "examples/evidence/shadow/" + "a" * 33 + "/comparison.json",
    "examples/evidence/shadow/" + "A" * 32 + "/comparison.json",
    "examples/evidence/shadow/" + "g" * 32 + "/comparison.json",
    "examples/evidence/shadow/" + "a" * 32 + "/../comparison.json",
    "examples/evidence/shadow/" + "a" * 32 + "/comparison.json/extra",
    "examples/evidence/shadow/" + "a" * 32 + "/comparison.json?raw=1",
    "examples/evidence/shadow/" + "a" * 32 + "/comparison.json:stream",
    "examples/evidence/shadow/" + "a" * 32 + "/%2e%2e/comparison.json",
    "examples/evidence/shadow//" + "a" * 32 + "/comparison.json",
    "examples/evidence/other/" + "a" * 32 + "/comparison.json",
    "/examples/evidence/shadow/" + "a" * 32 + "/comparison.json",
    "examples\\evidence\\shadow\\" + "a" * 32 + "\\comparison.json",
])
def test_published_paths_cannot_escape_exact_allowlist(tmp_path, path):
    catalog = EvidenceCatalog(tmp_path)
    with pytest.raises(EvidenceError, match="allowlist"):
        catalog.snapshot(path)
    with pytest.raises(EvidenceError, match="allowlist"):
        catalog._safe_path(path, shadow_auxiliary=True)


@pytest.mark.parametrize("name", [
    "manifest.json", "run.json", "config.json", "source.zip", "output/result.json",
])
def test_published_auxiliary_files_are_only_internal_verification_inputs(published_shadow, name):
    root, folder, relative = published_shadow
    catalog = EvidenceCatalog(root)
    auxiliary = relative.rsplit("/", 1)[0] + "/" + name
    with pytest.raises(EvidenceError, match="allowlist"):
        catalog.snapshot(auxiliary)
    assert catalog._safe_path(auxiliary, shadow_auxiliary=True) == folder / name


@pytest.mark.parametrize("name", [
    "output/episodes/0.json", "checkpoint.pt", "worker.log", "source/main.py",
    "../source.zip", "output/../config.json", "output/result.json/extra",
    "output/result.json:stream", ".env", "manifest.json.bak",
])
def test_published_auxiliary_allowance_cannot_read_other_files(published_shadow, name):
    root, _, relative = published_shadow
    auxiliary = relative.rsplit("/", 1)[0] + "/" + name
    with pytest.raises(EvidenceError, match="allowlist"):
        EvidenceCatalog(root)._safe_path(auxiliary, shadow_auxiliary=True)


@pytest.mark.parametrize("name", ["source.zip", "config.json", "output/result.json"])
def test_public_bundle_rejects_mutated_hash_bound_bytes(published_shadow, name):
    root, folder, relative = published_shadow
    target = folder / name
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(EvidenceError, match="hash|configuration|provenance"):
        EvidenceCatalog(root).snapshot(relative)
    row = next(row for row in EvidenceCatalog(root).list_records() if row["id"] == relative)
    assert row["status"] == "invalid"


def test_public_bundle_rejects_forged_provenance(published_shadow):
    root, folder, relative = published_shadow
    comparison_path = folder / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["provenance"]["result_sha256"] = "0" * 64
    comparison_path.write_text(json.dumps(comparison))
    with pytest.raises(EvidenceError, match="provenance"):
        EvidenceCatalog(root).snapshot(relative)


def test_public_bundle_recomputes_claims_even_if_run_and_comparison_agree(published_shadow):
    root, folder, relative = published_shadow
    comparison_path, run_path = folder / "comparison.json", folder / "run.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["recommendation"]["efficacy_claim"] = True
    run = json.loads(run_path.read_text())
    run["comparison"] = comparison
    comparison_path.write_text(json.dumps(comparison))
    run_path.write_text(json.dumps(run))
    with pytest.raises(EvidenceError, match="deterministic analysis"):
        EvidenceCatalog(root).snapshot(relative)


def test_recent_local_trials_do_not_hide_public_starter_bundle(saved_shadow, published_shadow, monkeypatch):
    import fusion_agent.research.evidence as module

    root, _, private_id = saved_shadow
    _, _, public_id = published_shadow
    monkeypatch.setattr(module, "MAX_SHADOW_RECORDS", 1)
    assert {row["id"] for row in EvidenceCatalog(root).list_records()} == {private_id, public_id}


def test_public_prefix_keeps_symlink_policy(published_shadow, monkeypatch):
    from pathlib import Path

    root, _, relative = published_shadow
    original = Path.is_symlink
    forbidden = root / "examples/evidence/shadow"
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == forbidden or original(path))
    with pytest.raises(EvidenceError, match="Symlinks"):
        EvidenceCatalog(root).snapshot(relative)
