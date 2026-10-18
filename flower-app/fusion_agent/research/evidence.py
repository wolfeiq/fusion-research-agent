"""Bounded, read-only snapshots of explicitly published experiment evidence.

This adapter never runs experiments, imports a learner, or deserializes a model.
An audit's reported status is evidence, not a fresh audit by this adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 2_000_000
MAX_CONTENT_CHARS = 30_000
MAX_RECORDS = 48
MAX_SHADOW_RECORDS = 8
MAX_SHADOW_CONTENT_CHARS = 70_000
_SHADOW_ROOTS = ("flower-app/.dashboard/shadow", "examples/evidence/shadow")
_SHADOW_PREFIX = r"(?:flower-app/\.dashboard/shadow|examples/evidence/shadow)"
_SHADOW_RECORD = re.compile(_SHADOW_PREFIX + r"/[a-f0-9]{32}/comparison\.json")
_SHADOW_AUXILIARY = re.compile(
    _SHADOW_PREFIX + r"/[a-f0-9]{32}/"
    r"(?:manifest\.json|run\.json|config\.json|source\.zip|output/result\.json)"
)

_EXACT = (
    "results/td3_heating_stability/fixed_partner_comparison.json",
    "results/td3_heating_stability/coupled_comparison.json",
    "results/td3_heating_stability/repair_comparison.json",
    "results/coldstart_learning_diagnostic/round2_forks/summary.json",
    "results/coldstart_learning_diagnostic/round2_gate/summary.json",
    "results/coldstart_federation_live_smoke/independent_analysis.json",
    "results/coldstart_fixed_feasibility/summary.json",
    "results/coldstart_learning_diagnostic/checkpoint_probe.json",
    "results/td3_heating_stability/conditioning_probe/results.json",
    "docs/coldstart_coupled_federation.md",
    "docs/coldstart_learning_diagnosis.md",
    "docs/td3_heating_stability.md",
    "docs/federated_rl_research.md",
)
_PATTERNS = (
    "results/td3_heating_stability/*/independent_analysis.json",
    "results/coldstart_kaggle_audit/*/audit_summary.json",
    "results/coldstart_kaggle_audit/*/independent_analysis.json",
)
_EXCLUDED_KEYS = {
    "api_key", "apikey", "access_token", "refresh_token", "authorization",
    "password", "secret", "secrets", "credentials", "environment", "env",
    "trace", "traces", "trajectory", "trajectories", "replay", "replay_buffer",
    "state_dict", "model_weights", "optimizer_state", "checkpoint_bytes",
}
_LOCAL_PATH = re.compile(
    r"[A-Za-z]:[\\/][^\s\"<>]*|\\\\[^\s\"<>]+"
    r"|(?<![\w:])/(?:Users|home|tmp|mnt|var|private)/[^\s\"<>]*"
)


class EvidenceError(ValueError):
    """Evidence is unavailable, malformed, changing, or outside policy."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("JSON has duplicate object keys")
        result[key] = value
    return result


def _json_load(content: str) -> Any:
    def invalid_constant(_: str) -> None:
        raise EvidenceError("JSON contains a non-finite number")

    def finite_float(token: str) -> float:
        result = float(token)
        if not math.isfinite(result):
            raise EvidenceError("JSON contains a non-finite number")
        return result

    try:
        return json.loads(
            content, object_pairs_hook=_unique_object,
            parse_constant=invalid_constant, parse_float=finite_float,
        )
    except (ValueError, RecursionError) as exc:
        raise EvidenceError("Malformed JSON evidence") from exc


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer, without expression evaluation or wildcards.

    ``document`` may be a parsed JSON value or a JSON content string. Array
    indices must be canonical nonnegative integers; '-' is not a read index.
    """
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise EvidenceError("JSON pointer must be empty or start with '/'")
    value = _json_load(document) if isinstance(document, str) else document
    if not pointer:
        return value
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise EvidenceError("Invalid JSON pointer escape")
        key = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
            try:
                value = value[int(key)]
            except (IndexError, ValueError) as exc:
                raise EvidenceError("JSON pointer does not resolve") from exc
        else:
            raise EvidenceError("JSON pointer does not resolve")
    return value


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


class EvidenceCatalog:
    """Read only a fixed catalogue of summaries and protocol documents.

    Record IDs are repo-relative POSIX paths, but a caller cannot choose an
    arbitrary path: every access must match an exact entry or bounded pattern.
    Snapshots are self-contained and their hash covers the exact returned UTF-8
    content. ``metadata.original_sha256`` separately authenticates source bytes.
    """

    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root).resolve(strict=True)
        if not self.repo_root.is_dir():
            raise EvidenceError("Repository root is not a directory")

    @staticmethod
    def _allowed(record_id: str) -> bool:
        if not isinstance(record_id, str) or "\\" in record_id:
            return False
        parts = record_id.split("/")
        if any(part in {"", ".", ".."} for part in parts) or ":" in record_id:
            return False
        if record_id in _EXACT:
            return True
        if _SHADOW_RECORD.fullmatch(record_id):
            return True
        if len(parts) != 4:
            return False
        if parts[:2] == ["results", "td3_heating_stability"]:
            return (
                parts[3] == "independent_analysis.json"
                and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", parts[2]) is not None
            )
        return (
            parts[:2] == ["results", "coldstart_kaggle_audit"]
            and re.fullmatch(r"[0-9]{1,20}", parts[2]) is not None
            and parts[3] in {"audit_summary.json", "independent_analysis.json"}
        )

    def _safe_path(self, record_id: str, *, shadow_auxiliary: bool = False) -> Path:
        if not self._allowed(record_id) and not (
            shadow_auxiliary and _SHADOW_AUXILIARY.fullmatch(record_id)
        ):
            raise EvidenceError("Record is not in the evidence allowlist")
        path = self.repo_root / record_id
        current = self.repo_root
        for component in Path(record_id).parts:
            current = current / component
            if current.is_symlink() or (
                hasattr(current, "is_junction") and current.is_junction()
            ):
                raise EvidenceError("Symlinks and junctions are not evidence records")
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.repo_root) or not resolved.is_file():
                raise EvidenceError("Evidence path leaves the repository")
        except OSError as exc:
            raise EvidenceError("Evidence record is missing or unreadable") from exc
        return path

    def _read_stable(
        self, record_id: str, *, shadow_auxiliary: bool = False,
        max_bytes: int | None = None,
    ) -> bytes:
        max_bytes = MAX_FILE_BYTES if max_bytes is None else max_bytes
        try:
            path = self._safe_path(record_id, shadow_auxiliary=shadow_auxiliary)
            before = _fingerprint(path)
            if before[0] > max_bytes:
                raise EvidenceError("Evidence source exceeds the byte limit")
            first = path.read_bytes()
            self._safe_path(record_id, shadow_auxiliary=shadow_auxiliary)
            middle = _fingerprint(path)
            second = path.read_bytes()
            self._safe_path(record_id, shadow_auxiliary=shadow_auxiliary)
            after = _fingerprint(path)
            if before != middle or middle != after or first != second:
                raise EvidenceError("Evidence changed while being read; retry later")
            if len(first) > max_bytes:
                raise EvidenceError("Evidence source exceeds the byte limit")
            return first
        except OSError as exc:
            raise EvidenceError("Evidence record is missing or unreadable") from exc

    def _sanitize_text(self, value: str) -> str:
        root = str(self.repo_root).replace("\\", "/").rstrip("/")
        normalized = value.replace("\\", "/")
        if normalized.lower().startswith(root.lower() + "/"):
            return normalized[len(root) + 1:]
        return _LOCAL_PATH.sub("[redacted local path]", value)

    def _project(
        self, value: Any, omissions: set[str], *, compact: bool = False,
        pointer: str = "", depth: int = 0,
    ) -> Any:
        if depth > 64:
            raise EvidenceError("Evidence nesting exceeds the limit")
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                child_pointer = pointer + "/" + escaped
                if key.lower() in _EXCLUDED_KEYS or (
                    compact and key == "per_case"
                ) or (not pointer and key in {"jobs", "paired_by_seed_facility_round"}):
                    omissions.add(child_pointer)
                    continue
                result[key] = self._project(
                    child, omissions, compact=compact,
                    pointer=child_pointer, depth=depth + 1,
                )
            return result
        if isinstance(value, list):
            return [self._project(
                child, omissions, compact=compact, pointer=f"{pointer}/{i}",
                depth=depth + 1,
            ) for i, child in enumerate(value)]
        return self._sanitize_text(value) if isinstance(value, str) else value

    def snapshot(self, record_id: str) -> dict[str, Any]:
        if isinstance(record_id, str) and record_id.startswith("federation:"):
            from .federation import FederationCatalog

            return FederationCatalog(self.repo_root).snapshot(record_id)
        if isinstance(record_id, str) and _SHADOW_RECORD.fullmatch(record_id):
            return self._shadow_snapshot(record_id)
        raw = self._read_stable(record_id)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise EvidenceError("Evidence is not UTF-8 text") from exc
        omissions: set[str] = set()
        is_json = record_id.endswith(".json")
        status = "unknown" if is_json else "protocol"
        if is_json:
            parsed = _json_load(text)
            if not isinstance(parsed, dict):
                raise EvidenceError("Evidence summary must be a JSON object")
            reported = parsed.get("status", parsed.get("overall_status", "unknown"))
            if isinstance(reported, str) and len(reported) <= 100:
                status = reported
            projected = self._project(parsed, omissions)
            content = json.dumps(projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if len(content) > MAX_CONTENT_CHARS:
                projected = self._project(parsed, omissions, compact=True)
                content = json.dumps(projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        else:
            content = self._sanitize_text(text)
        if len(content) > MAX_CONTENT_CHARS:
            raise EvidenceError("Evidence exceeds the content limit after projection")
        kind = "experiment" if is_json else "protocol"
        return {
            "id": record_id, "title": self._title(record_id), "kind": kind,
            "uri": record_id, "content": content,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": {
                "status": status, "original_sha256": hashlib.sha256(raw).hexdigest(),
                "status_basis": "reported_not_recomputed",
                "freshness": "not_independently_verified",
                "original_size_bytes": len(raw), "content_chars": len(content),
                "format": "json" if is_json else "markdown",
                "transformation": (
                    "JSON serialization and local-path redaction; omitted pointers listed"
                    if is_json else "UTF-8 decoding and local-path redaction"
                ),
                "omitted_json_pointers": sorted(omissions),
                "verification_scope": (
                    "Stable source bytes and snapshot hash only. Status is the saved "
                    "report's assertion; raw trajectories and remote job completion "
                    "were not freshly audited. Protocols are plans, not outcomes."
                ),
            },
        }

    def _shadow_snapshot(self, record_id: str) -> dict[str, Any]:
        """Verify a saved shadow analysis against its immutable local inputs.

        Only the comparison is exposed as evidence. Auxiliary reads are exact
        adjacent files, not model-selectable paths. Source ZIP bytes are hashed
        without extraction or executable/model deserialization.
        """
        from .shadow_analysis import analyze_shadow_result, attach_execution_status

        prefix = record_id.rsplit("/", 1)[0]
        identifier = prefix.rsplit("/", 1)[1]
        captured = {}

        def read(name):
            path = f"{prefix}/{name}"
            limit = 16_000_000 if name == "source.zip" else 4_000_000
            data = self._read_stable(
                path, shadow_auxiliary=name != "comparison.json", max_bytes=limit,
            )
            captured[path] = (data, limit)
            if name == "source.zip":
                return data
            try:
                parsed = _json_load(data.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise EvidenceError("Shadow evidence is not UTF-8 JSON") from exc
            if not isinstance(parsed, dict):
                raise EvidenceError("Shadow evidence must be a JSON object")
            return parsed

        def parent_digest(value):
            return hashlib.sha256(json.dumps(
                value, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest()

        try:
            comparison, manifest, run = (read(name) for name in (
                "comparison.json", "manifest.json", "run.json"
            ))
            result, config, archive = read("output/result.json"), read("config.json"), read("source.zip")
            if (manifest.get("schema") != "fusion-shadow-manifest-v1"
                    or manifest.get("id") != identifier or run.get("id") != identifier):
                raise EvidenceError("Shadow manifest/run identity mismatch")
            unsigned = {key: value for key, value in manifest.items() if key != "identity"}
            if parent_digest(unsigned) != manifest.get("identity"):
                raise EvidenceError("Shadow manifest identity hash mismatch")
            config_bytes = captured[f"{prefix}/config.json"][0]
            if (hashlib.sha256(config_bytes).hexdigest() != manifest.get("config_sha256")
                    or config != manifest.get("config") or config != result.get("config")
                    or config != run.get("config")):
                raise EvidenceError("Shadow frozen configuration mismatch")
            archive_hash = hashlib.sha256(archive).hexdigest()
            if archive_hash != manifest.get("source_zip_sha256"):
                raise EvidenceError("Shadow source archive hash mismatch")
            raw_result_hash = hashlib.sha256(captured[f"{prefix}/output/result.json"][0]).hexdigest()
            expected_provenance = {
                "trial_id": identifier, "manifest_identity": manifest["identity"],
                "result_sha256": raw_result_hash, "source_zip_sha256": archive_hash,
            }
            if comparison.get("provenance") != expected_provenance:
                raise EvidenceError("Shadow comparison provenance mismatch")
            if run.get("status") not in {"completed", "failed", "timed_out", "interrupted"}:
                raise EvidenceError("Shadow run is not terminal")
            recomputed = attach_execution_status(analyze_shadow_result(result), run.get("status"))
            recomputed["provenance"] = expected_provenance
            # JSON roundtrip canonicalizes integer seed keys exactly as saved.
            if comparison != json.loads(json.dumps(recomputed, allow_nan=False)):
                raise EvidenceError("Shadow comparison does not match deterministic analysis")
            if run.get("status") == "completed" and comparison.get("status") != "complete":
                raise EvidenceError("Shadow run completion disagrees with comparison")
            if run.get("comparison") != comparison or run.get("manifest") != manifest:
                raise EvidenceError("Shadow run references different saved evidence")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EvidenceError):
                raise
            raise EvidenceError("Malformed or inconsistent shadow evidence") from exc
        for path, (before, limit) in captured.items():
            after = self._read_stable(
                path, shadow_auxiliary=path != record_id, max_bytes=limit,
            )
            if before != after:
                raise EvidenceError("Shadow evidence changed during verification")
        omissions: set[str] = set()
        projected = self._project(comparison, omissions)
        from .shadow_telemetry import clipping_diagnostic_view

        projected["round_aggregation_diagnostics"] = clipping_diagnostic_view(result, raw_result_hash)
        content = json.dumps(projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(content) > MAX_SHADOW_CONTENT_CHARS:
            raise EvidenceError("Shadow comparison exceeds its content limit")
        raw = captured[record_id][0]
        return {
            "id": record_id, "title": f"Shadow trial {identifier[:8]}: paired comparison",
            "kind": "experiment", "uri": record_id, "content": content,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": {
                "adapter": "shadow_comparison", "status": comparison["status"],
                "execution_status": run["status"], "format": "json",
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "original_size_bytes": len(raw), "content_chars": len(content),
                "status_basis": "hash_bound_worker_summary_and_recomputed_comparison",
                "freshness": "saved_isolated_trial_no_remote_completion_claim",
                "omitted_json_pointers": sorted(omissions),
                "transformation": "Deterministic comparison verified unchanged; additional bounded clipping diagnostics projected from its hash-verified worker result; JSON serialization and local-path redaction",
                "provenance": expected_provenance,
                "verification_scope": "Verified source archive bytes, manifest, config and worker summary hashes; recomputed summary arithmetic. No raw trajectory audit, code execution or live policy adoption.",
            },
        }

    @staticmethod
    def _title(record_id: str) -> str:
        path = Path(record_id)
        stem = path.stem.replace("_", " ")
        return stem if path.parent.name == "docs" else f"{path.parent.name}: {stem}"

    def list_records(self) -> list[dict[str, Any]]:
        ids = {name for name in _EXACT if (self.repo_root / name).exists()}
        for shadow_root in _SHADOW_ROOTS:
            shadow_paths = [path for path in self.repo_root.glob(
                shadow_root + "/*/comparison.json"
            ) if _SHADOW_RECORD.fullmatch(path.relative_to(self.repo_root).as_posix())]
            # Bound each collection separately so new local trials do not hide
            # published starter evidence. Older records remain directly readable;
            # no scientific rows within a selected comparison are truncated.
            shadow_paths.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
            ids.update(path.relative_to(self.repo_root).as_posix()
                       for path in shadow_paths[:MAX_SHADOW_RECORDS])
        if len(ids) > MAX_RECORDS:
            raise EvidenceError("Evidence catalogue exceeds the record limit")
        for pattern in _PATTERNS:
            for path in self.repo_root.glob(pattern):
                record_id = path.relative_to(self.repo_root).as_posix()
                if self._allowed(record_id):
                    ids.add(record_id)
                if len(ids) > MAX_RECORDS:
                    raise EvidenceError("Evidence catalogue exceeds the record limit")
        records = []
        for record_id in sorted(ids):
            try:
                snapshot = self.snapshot(record_id)
            except EvidenceError as exc:
                records.append({
                    "id": record_id, "title": self._title(record_id),
                    "path": record_id,
                    "kind": "experiment" if record_id.endswith(".json") else "protocol",
                    "status": "invalid", "size": None, "error": str(exc),
                })
                continue
            records.append({
                "id": record_id, "title": snapshot["title"], "path": record_id,
                "kind": snapshot["kind"], "status": snapshot["metadata"]["status"],
                "size": snapshot["metadata"]["original_size_bytes"],
                **({"adapter": "shadow_comparison", "metadata": {"adapter": "shadow_comparison"}}
                   if snapshot["metadata"].get("adapter") == "shadow_comparison" else {}),
            })
        from .federation import FederationCatalog

        records.extend(FederationCatalog(self.repo_root).list_records())
        if len(records) > MAX_RECORDS:
            raise EvidenceError("Evidence catalogue exceeds the record limit")
        return records
