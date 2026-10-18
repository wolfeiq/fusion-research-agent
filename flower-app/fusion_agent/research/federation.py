"""Read-only round dossiers from registered Flower experiment artifacts.

Only JSON and the coordinator's bounded scalar metadata member are read. No
learner, simulator, pickle, replay buffer or raw episode trajectory is loaded.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .evidence import EvidenceError, _json_load

REGISTERED_RUNS = {
    "federation:local-smoke": {
        "title": "Flower round audit — local TORAX smoke",
        "path": "results/coldstart_federation_live_smoke",
    },
    "federation:kaggle-359552340": {
        "title": "Flower round audit — Kaggle 359552340",
        "path": "results/coldstart_kaggle_audit/359552340/output/"
        "coldstart_federation_job/study",
    },
}
ROLES = ("heating_td3", "deposition_sac")
MAX_FILE_BYTES = 8_000_000
MAX_ZIP_BYTES = 32_000_000
MAX_METADATA_BYTES = 2_000_000
MAX_CONTENT_CHARS = 70_000
CHANNEL_FIELDS = (
    "base_hash", "proposal_hash", "schema", "source_ids", "update_ids",
    "revisions", "source_norms", "weights", "arithmetic", "thresholds",
    "clipping_factors", "clipped_sources", "uniform_mean_norms",
    "clipped_mean_norms", "norm_match_factors", "norm_match_available",
    "norm_match_amplified", "realized_delta_norms", "proposal_hashes",
)
ACK_FIELDS = (
    "round", "gate", "gate_reference", "deployed_cost", "raw_costs",
    "raw_events", "gate_episodes", "evaluation_episodes", "training_episodes",
    "contribution_optimization", "repair_optimization", "evidence",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _stat(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def _metadata(raw):
    """Inspect archive bounds before NumPy sees the metadata NPY header."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if (len(members) > 128 or len(names) != len(set(names))
                    or sum(member.file_size for member in members) > MAX_ZIP_BYTES
                    or "metadata.npy" not in names):
                raise EvidenceError("Coordinator ZIP exceeds bounds or has ambiguous members")
            member = archive.getinfo("metadata.npy")
            if member.file_size > MAX_METADATA_BYTES:
                raise EvidenceError("Coordinator metadata exceeds the uncompressed limit")
            payload = archive.read(member)
        stream = io.BytesIO(payload)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise EvidenceError("Unsupported coordinator metadata NPY version")
        if (shape != () or dtype.kind != "U" or dtype.hasobject
                or dtype.itemsize > MAX_METADATA_BYTES
                or len(payload) - stream.tell() != dtype.itemsize):
            raise EvidenceError("Coordinator metadata must be a bounded Unicode scalar")
        value = np.load(io.BytesIO(payload), allow_pickle=False).item()
        metadata = _json_load(value)
        if not isinstance(metadata, dict):
            raise EvidenceError("Coordinator metadata must be an object")
        return metadata
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        if isinstance(exc, EvidenceError):
            raise
        raise EvidenceError("Invalid coordinator metadata archive") from exc


def _numeric_tree(value, *, depth=0):
    """Bound summary objects; never accept an array-sized payload or NaN."""
    if depth > 10:
        raise EvidenceError("Federation metadata nesting exceeds the limit")
    if isinstance(value, dict):
        if len(value) > 100:
            raise EvidenceError("Federation metadata object exceeds the limit")
        return {key: _numeric_tree(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 100:
            raise EvidenceError("Federation metadata array exceeds the limit")
        return [_numeric_tree(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if len(value) > 4000:
            raise EvidenceError("Federation metadata text exceeds the limit")
        # Identifiers, error descriptions and protocol strings are never paths
        # to open. Redact user filesystem roots before any external reasoning.
        return re.sub(r"[A-Za-z]:[\\/][^\s\"<>]*|/(?:Users|home)/[^\s\"<>]*",
                      "[redacted local path]", value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    raise EvidenceError("Unsupported federation metadata value")


class FederationCatalog:
    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve(strict=True)
        if not self.repo_root.is_dir():
            raise EvidenceError("Repository root is not a directory")

    def _path(self, relative):
        parts = relative.split("/")
        if any(part in {"", ".", ".."} for part in parts) or "\\" in relative:
            raise EvidenceError("Invalid registered artifact path")
        current = self.repo_root
        for part in parts:
            current = current / part
            if current.is_symlink() or (
                hasattr(current, "is_junction") and current.is_junction()
            ):
                raise EvidenceError("Symlink or junction artifacts are not accepted")
        if not current.resolve().is_relative_to(self.repo_root):
            raise EvidenceError("Artifact leaves the repository")
        return current

    def _read(self, relative, files, *, optional=False):
        path = self._path(relative)
        if optional and not path.exists():
            return None
        try:
            before = _stat(path)
            if before[0] > MAX_FILE_BYTES:
                raise EvidenceError("Federation artifact exceeds the byte limit")
            raw = path.read_bytes()
            self._path(relative)
            middle = _stat(path)
            again = path.read_bytes()
            self._path(relative)
            if before != middle or middle != _stat(path) or raw != again:
                raise EvidenceError("Federation artifact changed while being read")
            if len(raw) > MAX_FILE_BYTES:
                raise EvidenceError("Federation artifact exceeds the byte limit")
        except OSError as exc:
            raise EvidenceError("Registered federation artifact is missing or unreadable") from exc
        files[relative] = {"sha256": _sha(raw), "size_bytes": len(raw)}
        return raw

    def _json(self, relative, files, *, optional=False):
        raw = self._read(relative, files, optional=optional)
        if raw is None:
            return None
        try:
            value = _json_load(raw.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise EvidenceError("Federation JSON is not UTF-8") from exc
        if not isinstance(value, dict):
            raise EvidenceError("Federation JSON must be an object")
        return value

    def _job(self, base, manifest, strategy, seed, files):
        name = f"{strategy}-seed-{seed}"
        directory = f"{base}/jobs/{name}"
        summary = self._json(f"{directory}/summary.json", files, optional=True)
        job = self._json(f"{directory}/job.json", files, optional=True)
        record = {"id": name, "strategy": strategy, "seed": seed,
                  "recorded_status": (summary or {}).get("status", "not_recorded"),
                  "completed_rounds": 0, "pending": None, "roles": {}, "rounds": []}
        if job is None:
            if summary is not None:
                raise EvidenceError("Job summary has no frozen job contract")
            record["availability"] = "no_job_artifacts"
            return record
        cohort = [item[0] for item in manifest["config"]["facility_factors"]]
        if (job.get("revision") != manifest["identity"] or job.get("strategy") != strategy
                or job.get("seed") != seed or job.get("cohort") != cohort
                or job.get("rounds") != manifest["config"]["rounds"]
                or job.get("run_id") != f"{manifest['identity']}:{strategy}:seed-{seed}"):
            raise EvidenceError("Job contract differs from the frozen manifest")
        raw = self._read(f"{directory}/coordinator.npz", files, optional=True)
        if raw is None:
            if record["recorded_status"] == "complete":
                raise EvidenceError("Completed job has no coordinator evidence")
            record["availability"] = "no_coordinator_evidence"
            return record
        meta = _metadata(raw)
        version = meta.get("version")
        if (type(version) is not int or not 0 <= version <= 32
                or (version == 0 and meta.get("pending"))):
            raise EvidenceError("Invalid coordinator round count")
        if (meta.get("format") != 1 or meta.get("run_id") != job["run_id"]
                or meta.get("strategy") != strategy or type(meta.get("pending")) is not bool
                or set(meta.get("channels", {})) != set(ROLES)
                or len(meta.get("history", [])) != version
                or version > job["rounds"]):
            raise EvidenceError("Coordinator does not match the frozen job")
        if version:
            phase = "proposal" if meta["pending"] else "acknowledged"
            label = f"round-{version}-{phase}"
            ledger = self._json(f"{directory}/coordinator-checkpoints.json", files)
            entry = ledger.get(label, {})
            filename = f"coordinator-{label}.npz"
            if entry.get("filename") != filename:
                raise EvidenceError("Latest coordinator phase has no exact archive record")
            archived = self._read(f"{directory}/{filename}", files)
            if _sha(archived) != entry.get("sha256"):
                raise EvidenceError("Coordinator archive hash mismatch")
            if _metadata(archived) != meta:
                raise EvidenceError("Current coordinator and archived phase disagree; retry later")
            record["archive_hash_verified"] = True
            record["coordinator_artifact"] = f"{directory}/{filename}"
        else:
            record["archive_hash_verified"] = False
            record["coordinator_artifact"] = f"{directory}/coordinator.npz"
        if meta["channels"][ROLES[0]].get("schema") == meta["channels"][ROLES[1]].get("schema"):
            raise EvidenceError("Heating and deposition schemas are not separated")
        previous = dict(job["initial_base_hashes"])
        for role in ROLES:
            channel = meta["channels"][role]
            expected_ids = {f"{job['run_id']}:{n}:{source}:{role}"
                            for n in range(1, version + 1) for source in cohort}
            if (channel.get("cohort") != cohort or channel.get("schema") != job["schemas"][role]
                    or channel.get("version") != version
                    or channel.get("revisions") != {source: manifest["identity"] for source in cohort}
                    or len(channel.get("consumed", [])) != len(expected_ids)
                    or set(channel.get("consumed", [])) != expected_ids):
                raise EvidenceError("Role channel or consumed update IDs differ from contract")
            record["roles"][role] = {
                key: channel[key] for key in ("cohort", "schema", "revisions", "version", "hash")
            }
            record["roles"][role]["consumed_update_count"] = len(expected_ids)
        for number, row in enumerate(meta["history"], 1):
            if (row.get("round") != number or row.get("strategy") != strategy
                    or set(row.get("channels", {})) != set(ROLES)):
                raise EvidenceError("Misordered or inconsistent coordinator round")
            acknowledgements = row.get("acknowledgements")
            is_pending = meta["pending"] and number == version
            if is_pending:
                if acknowledgements is not None:
                    raise EvidenceError("Pending coordinator has unexpected acknowledgements")
            elif not isinstance(acknowledgements, dict) or set(acknowledgements) != set(cohort):
                raise EvidenceError("Completed round lacks full-cohort acknowledgements")
            channels = {}
            for role, diagnostics in row["channels"].items():
                if diagnostics.get("base_hash") != previous[role]:
                    raise EvidenceError("Coordinator base/proposal hash chain is broken")
                next_hash = (diagnostics.get("proposal_hashes", {}).get("clipped")
                             if strategy == "clipped" else diagnostics.get("proposal_hash"))
                if not isinstance(next_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", next_hash):
                    raise EvidenceError("Coordinator proposal hash is missing or invalid")
                previous[role] = next_hash
                channels[role] = _numeric_tree({
                    key: diagnostics[key] for key in CHANNEL_FIELDS if key in diagnostics
                })
                channels[role]["unavailable"] = [
                    key for key in ("source_norms", "clipping_factors", "weights")
                    if key not in diagnostics
                ] + ["source_delta_cosines", "historical_td_errors", "historical_actor_gradients"]
            acks = None
            if acknowledgements is not None:
                acks = {}
                for source, report in acknowledgements.items():
                    if report.get("round") != number:
                        raise EvidenceError("Client acknowledgement names the wrong round")
                    acks[source] = _numeric_tree({key: report[key] for key in ACK_FIELDS if key in report})
            record["rounds"].append({"round": number, "strategy": strategy,
                                     "channels": channels, "acknowledgements": acks})
        if any(previous[role] != meta["channels"][role]["hash"] for role in ROLES):
            raise EvidenceError("Final channel hash differs from the proposal chain")
        completed = version - int(meta["pending"])
        if record["recorded_status"] == "complete" and (
            meta["pending"] or completed != job["rounds"]
        ):
            raise EvidenceError("Completed job status contradicts coordinator phase")
        record.update(completed_rounds=completed, pending=meta["pending"],
                      proposed_rounds=version, availability="coordinator_metadata")
        return record

    def snapshot(self, record_id):
        try:
            return self._snapshot(record_id)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as exc:
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError("Malformed registered federation evidence") from exc

    def _snapshot(self, record_id):
        if record_id not in REGISTERED_RUNS:
            raise EvidenceError("Unknown registered federation run")
        definition = REGISTERED_RUNS[record_id]
        base, files = definition["path"], {}
        manifest = self._json(f"{base}/manifest.json", files)
        identity = _sha(_canonical({key: value for key, value in manifest.items()
                                   if key != "identity"}).encode())
        if identity != manifest.get("identity"):
            raise EvidenceError("Federation manifest identity mismatch")
        config = manifest.get("config", {})
        seeds, strategies = config.get("seeds"), config.get("strategies")
        if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 8
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or len(set(seeds)) != len(seeds)
                or not isinstance(strategies, list) or not 1 <= len(strategies) <= 2
                or len(set(strategies)) != len(strategies)
                or not set(strategies) <= {"uniform", "clipped"}
                or type(config.get("rounds")) is not int or not 1 <= config["rounds"] <= 32
                or not isinstance(config.get("facility_factors"), list)):
            raise EvidenceError("Frozen federation matrix exceeds the allowed bounds")
        facilities = config["facility_factors"]
        if (not 1 <= len(facilities) <= 8
                or any(not isinstance(item, list) or len(item) != 2
                       or not isinstance(item[0], str)
                       or not re.fullmatch(r"[a-zA-Z0-9_-]{1,50}", item[0])
                       for item in facilities)
                or len({item[0] for item in facilities}) != len(facilities)):
            raise EvidenceError("Invalid registered facility cohort")
        summary = self._json(f"{base}/study_summary.json", files, optional=True) or {}
        state = self._json(f"{base}/study_state.json", files, optional=True) or {}
        for reported in (summary, state):
            if reported and reported.get("identity") != identity:
                raise EvidenceError("Study state belongs to another frozen manifest")
        status = state.get("status", summary.get("status", "unknown"))
        jobs = [self._job(base, manifest, strategy, seed, files)
                for strategy in strategies for seed in seeds]
        if status == "complete" and any(
            job["completed_rounds"] != config["rounds"] or job["pending"] is not False
            for job in jobs
        ):
            raise EvidenceError("Complete study status contradicts its job evidence")
        limitations = [
            "Read-only saved-artifact dossier; no current Kaggle API status was checked.",
            "Archive byte hashes, manifest identity, role separation and round hash/acknowledgement chains are checked. Critic tensors and raw physics trajectories are not freshly recomputed.",
            "Actors, replay, optimizer states and raw trajectories are never loaded or sent to the research model.",
            "Raw costs, deployed costs and empirical gate decisions remain separate. Lower endpoint cost does not by itself establish fewer shots to competence.",
            "Reported metadata is development simulation evidence, not confirmation or real-reactor efficacy.",
            "Uniform coordinator metadata does not record source norms or delta directions. Missing diagnostics are unavailable, not zero.",
            "Clipping metadata alone cannot establish improved control; inactive clipping may equal uniform aggregation.",
            "Checkpoint gradient probes and synthetic conditioning probes are not included: no historical optimizer-gradient or causal transfer claim is inferred.",
        ]
        if "gate_reference" not in config:
            limitations.append("This archived manifest predates gate_reference; its historical repaired-control gate must not be relabelled as the newer incumbent gate.")
        payload = {
            "schema": "fusion-federation-dossier-v1", "run_id": record_id,
            "status": status,
            "manifest": _numeric_tree({key: manifest[key] for key in (
                "identity", "study", "config", "learner_configs", "expected_budget", "protocol"
            ) if key in manifest}),
            "jobs": jobs, "limitations": limitations,
        }
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(content) > MAX_CONTENT_CHARS:
            raise EvidenceError("Federation dossier exceeds the content limit")
        # A file can be stable during its own read but change while another
        # artifact is inspected. Require the full input set to match again.
        for relative, provenance in files.items():
            reread = self._read(relative, {})
            if _sha(reread) != provenance["sha256"]:
                raise EvidenceError("Federation study changed during dossier construction; retry later")
        return {
            "id": record_id, "title": definition["title"], "kind": "experiment",
            "uri": base, "content": content, "sha256": _sha(content.encode()),
            "metadata": {
                "adapter": "federation_dossier",
                "status": status, "format": "json", "content_chars": len(content),
                "status_basis": "saved_study_and_coordinator_metadata_no_remote_status",
                "provenance": {"files": files, "manifest_identity_verified": True,
                               "raw_trajectory_audit": False, "tensor_recomputation": False},
                "transformation": "Bounded manifest and coordinator scalar-metadata projection; no model or trajectory contents",
            },
        }

    def list_records(self):
        records = []
        for record_id, definition in REGISTERED_RUNS.items():
            if not (self.repo_root / definition["path"] / "manifest.json").exists():
                continue
            try:
                snapshot = self.snapshot(record_id)
                records.append({"id": record_id, "title": definition["title"],
                                "path": definition["path"], "kind": "experiment",
                                "status": snapshot["metadata"]["status"],
                                "size": len(snapshot["content"].encode()),
                                "adapter": "federation_dossier",
                                "metadata": {"adapter": "federation_dossier"}})
            except EvidenceError as exc:
                records.append({"id": record_id, "title": definition["title"],
                                "path": definition["path"], "kind": "experiment",
                                "status": "invalid", "size": None, "error": str(exc),
                                "adapter": "federation_dossier",
                                "metadata": {"adapter": "federation_dossier"}})
        return records
