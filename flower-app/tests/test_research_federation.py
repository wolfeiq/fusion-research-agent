"""Federation dossier checks use miniature metadata-only fixture archives."""

import hashlib
import io
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fusion_agent.research.evidence import EvidenceError, resolve_pointer
from fusion_agent.research.federation import FederationCatalog

ID = "federation:local-smoke"
BASE = "results/coldstart_federation_live_smoke"
ROLES = ("heating_td3", "deposition_sac")
COHORT = ["donor_cool", "donor_warm", "cold_target"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def npz_bytes(metadata, *, object_metadata=False, junk_tensor=False):
    member = io.BytesIO()
    np.save(member, np.array(metadata if object_metadata else json.dumps(metadata),
                             dtype=object if object_metadata else None), allow_pickle=True)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("metadata.npy", member.getvalue())
        if junk_tensor:
            bundle.writestr("heating_td3/q0.body.weight.npy", b"NOT A VALID ARRAY")
    return archive.getvalue()


def fixture_run(root, *, pending=False, strategy="uniform"):
    base = root / BASE
    config = {"seeds": [17], "strategies": [strategy], "rounds": 1,
              "facility_factors": [[name, 1.0] for name in COHORT]}
    manifest = {"study": "metadata-only-test", "config": config,
                "learner_configs": {"td3": {"adapt_steps": 40}, "sac": {"adapt_steps": 40}},
                "expected_budget": {"total_episodes": 100},
                "protocol": {"scope": "Development only"}}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    manifest["identity"] = identity
    write_json(base / "manifest.json", manifest)
    status = "running" if pending else "complete"
    write_json(base / "study_state.json", {"status": status, "identity": identity})
    write_json(base / "study_summary.json", {"status": status, "identity": identity})
    jobdir = base / "jobs" / f"{strategy}-seed-17"
    job = {"revision": identity, "strategy": strategy, "seed": 17,
           "rounds": 1, "cohort": COHORT, "run_id": f"{identity}:{strategy}:seed-17",
           "initial_base_hashes": {r: "a" * 64 for r in ROLES},
           "schemas": {r: f"schema-{r}" for r in ROLES}}
    write_json(jobdir / "job.json", job)
    write_json(jobdir / "summary.json", {"status": status, "completed_rounds": 0 if pending else 1})
    channels = {r: {"cohort": COHORT, "schema": job["schemas"][r],
                    "revisions": {s: identity for s in COHORT}, "version": 1,
                    "hash": "b" * 64, "keys": ["q0.body.weight", "q1.body.weight"],
                    "consumed": [f"{job['run_id']}:1:{s}:{r}" for s in COHORT]}
                for r in ROLES}
    diagnostics = {r: {"base_hash": "a" * 64, "proposal_hash": "b" * 64} for r in ROLES}
    if strategy == "clipped":
        for row in diagnostics.values():
            row.pop("proposal_hash")
            row.update(proposal_hashes={"clipped": "b" * 64, "uniform": "b" * 64},
                       source_norms={s: {"q0": 0.01, "q1": 0.02} for s in COHORT},
                       weights={s: 1 / 3 for s in COHORT}, thresholds={"q0": 1.0, "q1": 1.0},
                       clipping_factors={s: {"q0": 1.0, "q1": 1.0} for s in COHORT})
    ack = {s: {"round": 1, "raw_costs": {"raw_federated": [0.12], "local": [0.15]},
               "deployed_cost": 0.12, "raw_events": {"raw_federated": [[]]},
               "gate": {"accept": True}, "repair_optimization": {
                   "raw_federated": {"heating_td3": {"actor_adaptation_updates": 20}}},
               "training_episodes": 6,
               "trajectory": ["SECRET RAW TRANSITION MUST NEVER BE INCLUDED"]}
           for s in COHORT}
    meta = {"format": 1, "version": 1, "pending": pending,
            "run_id": job["run_id"], "strategy": strategy, "channels": channels,
            "history": [{"round": 1, "strategy": strategy, "channels": diagnostics,
                         "acknowledgements": None if pending else ack}]}
    save_metadata(jobdir, meta)
    return base, jobdir, meta


def save_metadata(jobdir, metadata, **kwargs):
    raw = npz_bytes(metadata, **kwargs)
    (jobdir / "coordinator.npz").write_bytes(raw)
    label = f"round-{metadata['version']}-{'proposal' if metadata['pending'] else 'acknowledged'}"
    filename = f"coordinator-{label}.npz"
    (jobdir / filename).write_bytes(raw)
    write_json(jobdir / "coordinator-checkpoints.json", {
        label: {"filename": filename, "sha256": hashlib.sha256(raw).hexdigest()},
    })


def test_completed_dossier_retains_round_evidence_and_exact_source_hashes(tmp_path):
    base, _, _ = fixture_run(tmp_path)
    catalog = FederationCatalog(tmp_path)
    snapshot = catalog.snapshot(ID)
    data = json.loads(snapshot["content"])
    assert data["schema"] == "fusion-federation-dossier-v1"
    assert data["status"] == "complete"
    job = data["jobs"][0]
    assert job["completed_rounds"] == 1 and job["pending"] is False
    assert job["archive_hash_verified"] is True
    assert resolve_pointer(snapshot["content"], "/jobs/0/rounds/0/acknowledgements/cold_target/raw_costs/raw_federated/0") == 0.12
    assert snapshot["sha256"] == hashlib.sha256(snapshot["content"].encode()).hexdigest()
    manifest_provenance = snapshot["metadata"]["provenance"]["files"][f"{BASE}/manifest.json"]
    assert manifest_provenance["sha256"] == hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest()
    assert "SECRET RAW" not in snapshot["content"]
    assert snapshot["metadata"]["provenance"]["tensor_recomputation"] is False
    assert catalog.list_records()[0]["id"] == ID
    assert snapshot["metadata"]["adapter"] == "federation_dossier"
    assert catalog.list_records()[0]["metadata"]["adapter"] == "federation_dossier"


def test_pending_proposal_is_not_a_completed_round(tmp_path):
    fixture_run(tmp_path, pending=True)
    data = json.loads(FederationCatalog(tmp_path).snapshot(ID)["content"])
    assert data["status"] == "running"
    assert data["jobs"][0]["proposed_rounds"] == 1
    assert data["jobs"][0]["completed_rounds"] == 0
    assert data["jobs"][0]["pending"] is True
    assert data["jobs"][0]["rounds"][0]["acknowledgements"] is None


def test_tensor_arrays_are_never_loaded(tmp_path, monkeypatch):
    _, jobdir, meta = fixture_run(tmp_path)
    save_metadata(jobdir, meta, junk_tensor=True)
    original = Path.read_bytes
    opened = []

    def tracked(path):
        opened.append(path.name)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    snapshot = FederationCatalog(tmp_path).snapshot(ID)
    assert snapshot["metadata"]["provenance"]["tensor_recomputation"] is False
    assert not any(name.endswith(".pkl") or name.startswith("round-1.json") for name in opened)


def test_clipped_diagnostics_remain_exact_and_uniform_missing_fields_explicit(tmp_path):
    fixture_run(tmp_path, strategy="clipped")
    data = json.loads(FederationCatalog(tmp_path).snapshot(ID)["content"])
    channel = data["jobs"][0]["rounds"][0]["channels"]["heating_td3"]
    assert channel["source_norms"]["cold_target"]["q0"] == 0.01
    assert channel["clipping_factors"]["cold_target"]["q0"] == 1.0
    assert "source_delta_cosines" in channel["unavailable"]
    assert "source_norms" not in channel["unavailable"]


def test_uniform_source_norms_are_unavailable_not_zero(tmp_path):
    fixture_run(tmp_path)
    data = json.loads(FederationCatalog(tmp_path).snapshot(ID)["content"])
    channel = data["jobs"][0]["rounds"][0]["channels"]["heating_td3"]
    assert "source_norms" not in channel
    assert "source_norms" in channel["unavailable"]


@pytest.mark.parametrize("record_id", ["../private", "federation:other", "C:/secret", "federation:local-smoke/../../secret"])
def test_only_registered_run_ids_are_allowed(tmp_path, record_id):
    with pytest.raises(EvidenceError, match="Unknown registered"):
        FederationCatalog(tmp_path).snapshot(record_id)


def test_bad_manifest_identity_is_rejected(tmp_path):
    base, _, _ = fixture_run(tmp_path)
    value = json.loads((base / "manifest.json").read_text())
    value["config"]["rounds"] = 2
    write_json(base / "manifest.json", value)
    with pytest.raises(EvidenceError, match="identity"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_archive_hash_tampering_is_rejected(tmp_path):
    _, jobdir, _ = fixture_run(tmp_path)
    (jobdir / "coordinator-round-1-acknowledged.npz").write_bytes(b"changed")
    with pytest.raises(EvidenceError, match="hash mismatch"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_archive_ledger_cannot_redirect_to_an_arbitrary_file(tmp_path):
    _, jobdir, _ = fixture_run(tmp_path)
    ledger = json.loads((jobdir / "coordinator-checkpoints.json").read_text())
    ledger["round-1-acknowledged"]["filename"] = "../../../../../secret.pkl"
    write_json(jobdir / "coordinator-checkpoints.json", ledger)
    with pytest.raises(EvidenceError, match="exact archive"):
        FederationCatalog(tmp_path).snapshot(ID)


@pytest.mark.parametrize("mutation,match", [
    ("hash_chain", "hash chain"), ("role_schema", "schemas are not separated"),
    ("duplicate_update", "consumed update IDs"), ("missing_ack", "full-cohort"),
    ("wrong_ack_round", "wrong round"), ("bad_shape", "Malformed"),
])
def test_inconsistent_coordinator_contract_rejected(tmp_path, mutation, match):
    _, jobdir, meta = fixture_run(tmp_path)
    if mutation == "hash_chain":
        meta["history"][0]["channels"]["heating_td3"]["base_hash"] = "c" * 64
    elif mutation == "role_schema":
        meta["channels"]["heating_td3"]["schema"] = meta["channels"]["deposition_sac"]["schema"]
    elif mutation == "duplicate_update":
        meta["channels"]["heating_td3"]["consumed"].append(meta["channels"]["heating_td3"]["consumed"][0])
    elif mutation == "missing_ack":
        meta["history"][0]["acknowledgements"].pop("cold_target")
    elif mutation == "wrong_ack_round":
        meta["history"][0]["acknowledgements"]["cold_target"]["round"] = 2
    else:
        meta["channels"]["heating_td3"] = None
    save_metadata(jobdir, meta)
    with pytest.raises(EvidenceError, match=match):
        FederationCatalog(tmp_path).snapshot(ID)


def test_pickle_metadata_rejected_without_deserialization(tmp_path):
    _, jobdir, meta = fixture_run(tmp_path)
    save_metadata(jobdir, meta, object_metadata=True)
    with pytest.raises(EvidenceError, match="Unicode scalar"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_metadata_uncompressed_limit(tmp_path, monkeypatch):
    import fusion_agent.research.federation as module
    fixture_run(tmp_path)
    monkeypatch.setattr(module, "MAX_METADATA_BYTES", 10)
    with pytest.raises(EvidenceError, match="uncompressed limit"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_zip_total_uncompressed_limit(tmp_path, monkeypatch):
    import fusion_agent.research.federation as module
    fixture_run(tmp_path)
    monkeypatch.setattr(module, "MAX_ZIP_BYTES", 10)
    with pytest.raises(EvidenceError, match="ZIP exceeds bounds"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_content_bound_fails_instead_of_silently_truncating(tmp_path, monkeypatch):
    import fusion_agent.research.federation as module
    fixture_run(tmp_path)
    monkeypatch.setattr(module, "MAX_CONTENT_CHARS", 10)
    with pytest.raises(EvidenceError, match="content limit"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_symlink_ancestor_policy_without_windows_privilege(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    forbidden = tmp_path / "results"
    old = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == forbidden or old(p))
    with pytest.raises(EvidenceError, match="Symlink"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_changed_source_is_rejected(tmp_path, monkeypatch):
    fixture_run(tmp_path)
    old = Path.read_bytes
    count = 0

    def changing(path):
        nonlocal count
        raw = old(path)
        if path.name == "manifest.json":
            count += 1
            if count == 2:
                return raw + b" "
        return raw

    monkeypatch.setattr(Path, "read_bytes", changing)
    with pytest.raises(EvidenceError, match="changed"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_coordinator_changed_after_its_initial_stable_read_is_rejected(tmp_path, monkeypatch):
    _, _, meta = fixture_run(tmp_path)
    old = Path.read_bytes
    count = 0
    changed = deepcopy(meta)
    changed["history"][0]["acknowledgements"]["cold_target"]["deployed_cost"] = 9.0
    changed_bytes = npz_bytes(changed)

    def changing(path):
        nonlocal count
        raw = old(path)
        if path.name == "coordinator.npz":
            count += 1
            if count >= 3:
                return changed_bytes
        return raw

    monkeypatch.setattr(Path, "read_bytes", changing)
    with pytest.raises(EvidenceError, match="study changed"):
        FederationCatalog(tmp_path).snapshot(ID)


@pytest.mark.parametrize("missing", ["initial_base_hashes", "schemas"])
def test_missing_nested_job_fields_fail_as_evidence_error(tmp_path, missing):
    _, jobdir, _ = fixture_run(tmp_path)
    job = json.loads((jobdir / "job.json").read_text())
    job.pop(missing)
    write_json(jobdir / "job.json", job)
    with pytest.raises(EvidenceError, match="Malformed"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_failed_job_without_coordinator_is_retained_as_failure(tmp_path):
    base, jobdir, _ = fixture_run(tmp_path)
    identity = json.loads((base / "manifest.json").read_text())["identity"]
    write_json(base / "study_state.json", {"status": "failed", "identity": identity})
    write_json(jobdir / "summary.json", {"status": "failed"})
    (jobdir / "coordinator.npz").unlink()
    data = json.loads(FederationCatalog(tmp_path).snapshot(ID)["content"])
    assert data["status"] == "failed"
    assert data["jobs"][0]["availability"] == "no_coordinator_evidence"
    assert data["jobs"][0]["pending"] is None


def test_complete_job_without_coordinator_is_not_relabelled_success(tmp_path):
    _, jobdir, _ = fixture_run(tmp_path)
    (jobdir / "coordinator.npz").unlink()
    catalog = FederationCatalog(tmp_path)
    assert catalog.list_records()[0]["status"] == "invalid"
    with pytest.raises(EvidenceError, match="no coordinator"):
        catalog.snapshot(ID)


def test_pending_coordinator_cannot_support_complete_status(tmp_path):
    _, jobdir, _ = fixture_run(tmp_path, pending=True)
    write_json(jobdir / "summary.json", {"status": "complete"})
    with pytest.raises(EvidenceError, match="contradicts coordinator"):
        FederationCatalog(tmp_path).snapshot(ID)


def test_latest_coordinator_and_archived_metadata_must_agree(tmp_path):
    _, jobdir, meta = fixture_run(tmp_path)
    changed = deepcopy(meta)
    changed["history"][0]["acknowledgements"]["cold_target"]["deployed_cost"] = 10.0
    (jobdir / "coordinator.npz").write_bytes(npz_bytes(changed))
    with pytest.raises(EvidenceError, match="archived phase disagree"):
        FederationCatalog(tmp_path).snapshot(ID)
