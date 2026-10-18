"""Read-only scalar diagnostics from an already hash-verified worker result."""
from __future__ import annotations

import math
import re


def _number(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError("Expected a finite nonnegative recorded scalar")
    return value


def clipping_diagnostic_view(result, raw_sha256):
    """Expose recorded clipping for the clipped arm, never alter comparisons.

    The uniform arm may also record hypothetical clipped proposals; it is
    deliberately excluded so calculated clipping is not confused with applied
    clipping. Bounds/factors are checked, but saved tensors are not loaded.
    """
    records, missing = [], []
    for configuration in ("baseline", "candidate"):
        for seed_index, seed_row in enumerate(result.get(configuration, {}).get("seeds", [])):
            for round_index, point in enumerate(seed_row.get("rounds", [])):
                prefix = f"/{configuration}/seeds/{seed_index}/rounds/{round_index}/aggregation/clipped"
                aggregation = point.get("aggregation")
                channels = aggregation.get("clipped", {}) if isinstance(aggregation, dict) else {}
                if not isinstance(channels, dict):
                    channels = {}
                for role in ("heating_td3", "deposition_sac"):
                    pointer = prefix + "/" + role
                    if role not in channels:
                        missing.append(pointer)
                        continue
                    row = {"configuration": configuration, "seed": seed_row["seed"],
                           "round": point["round"], "arm": "clipped", "role": role,
                           "raw_json_pointer": pointer}
                    try:
                        channel = channels[role]
                        source_ids = channel["source_ids"]
                        if (not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 16
                                or any(not isinstance(item, str) or not 1 <= len(item) <= 96 for item in source_ids)
                                or len(set(source_ids)) != len(source_ids)):
                            raise ValueError("Invalid recorded contributor identities")
                        norms, factors = channel["source_norms"], channel["clipping_factors"]
                        if set(norms) != set(source_ids) or set(factors) != set(source_ids):
                            raise ValueError("Contributor norms and factors must match recorded identities")
                        counts = {}
                        for branch in ("q0", "q1"):
                            threshold = _number(channel["thresholds"][branch], positive=True)
                            active = []
                            for identifier in source_ids:
                                norm = _number(norms[identifier][branch])
                                factor = _number(factors[identifier][branch], positive=True)
                                expected = min(1.0, threshold / norm) if norm else 1.0
                                if factor > 1 or not math.isclose(factor, expected, rel_tol=1e-8, abs_tol=1e-12):
                                    raise ValueError("Recorded clipping factor disagrees with norm and threshold")
                                if factor < 1:
                                    active.append(identifier)
                            if sorted(channel["clipped_sources"][branch]) != sorted(active):
                                raise ValueError("Recorded clipped identities disagree with factors")
                            counts[branch] = len(active)
                        hashes = channel["proposal_hashes"]
                        if any(not isinstance(hashes.get(variant), str)
                               or not re.fullmatch(r"[0-9a-f]{64}", hashes[variant])
                               for variant in ("uniform", "clipped")):
                            raise ValueError("Missing recorded proposal hashes")
                        row.update({
                            "status": "recorded_scalars_consistent",
                            "source_ids": list(source_ids),
                            "source_count": len(source_ids),
                            "thresholds": {branch: channel["thresholds"][branch] for branch in ("q0", "q1")},
                            "source_norms": {identifier: {branch: norms[identifier][branch] for branch in ("q0", "q1")} for identifier in source_ids},
                            "clipping_factors": {identifier: {branch: factors[identifier][branch] for branch in ("q0", "q1")} for identifier in source_ids},
                            "clipped_sources": {branch: list(channel["clipped_sources"][branch]) for branch in ("q0", "q1")},
                            "clipped_source_counts": counts,
                            "proposal_hashes": {variant: hashes[variant] for variant in ("uniform", "clipped")},
                            "recorded_uniform_and_clipped_hashes_equal": hashes["uniform"] == hashes["clipped"],
                        })
                    except (ValueError, TypeError, KeyError, AttributeError):
                        row.update(status="invalid_recorded_telemetry", error="Required recorded clipping scalars or identities are missing or inconsistent.")
                    records.append(row)
    return {
        "schema": "fusion-shadow-clipping-diagnostic-v1",
        "status": "available" if records and not missing and all(row["status"] == "recorded_scalars_consistent" for row in records)
                  else "partial_or_unavailable",
        "raw_result_sha256": raw_sha256,
        "scope": "New read-only evidence projection from the hash-verified worker result. Counts derive from recorded factors; no tensor recomputation, new rollout or efficacy inference. Historical worker results and comparisons are unchanged.",
        "arm_scope": "Only the actually selected clipped arm is shown; hypothetical clipped proposals recorded under the uniform arm are excluded.",
        "records": records,
        "missing_json_pointers": missing,
    }
