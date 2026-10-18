"""Durable, role-separated synchronous critic federation.

The coordinator knows only critic tensors and approved scalar metrics. Local
actors/replay and the combined-controller acceptance decision stay at clients.
"""
import copy
import json
import os
from pathlib import Path

import numpy as np

from .critic_strategies import aggregate_variants
from .representation import UniformRepresentationServer, model_hash


ROLES = ("heating_td3", "deposition_sac")


class RepeatedCoordinator:
    def __init__(self, bases, cohort, schemas, revisions, *, strategy="uniform",
                 thresholds=None, run_id="development"):
        if set(bases) != set(ROLES) or set(schemas) != set(ROLES):
            raise ValueError("Separate heating TD3 and deposition SAC channels required")
        if strategy not in {"uniform", "clipped"}:
            raise ValueError("Unknown strategy")
        if not run_id or schemas[ROLES[0]] == schemas[ROLES[1]]:
            raise ValueError("Distinct role schemas and a run ID are required")
        self.strategy, self.run_id = strategy, run_id
        self.thresholds = copy.deepcopy(thresholds)
        self.channels = {
            role: UniformRepresentationServer(bases[role], cohort, schemas[role], revisions)
            for role in ROLES
        }
        if strategy == "clipped":
            if not thresholds or set(thresholds) != set(ROLES):
                raise ValueError("Freeze independent clipping thresholds for both roles")
            for limits in thresholds.values():
                if set(limits) != {"q0", "q1"} or any(
                    not np.isfinite(x) or x <= 0 for x in limits.values()
                ):
                    raise ValueError("Positive finite per-branch thresholds required")
        self.version = 0
        self.pending = False
        self.history = []

    def bases(self):
        return {r: {k: v.copy() for k, v in c.tensors.items()}
                for r, c in self.channels.items()}

    def aggregate(self, envelopes):
        """Validate both role rounds before any mutation; no partial application."""
        if self.pending:
            raise RuntimeError("Finish acknowledgement of the pending proposal first")
        if set(envelopes) != set(ROLES):
            raise ValueError("Both role envelopes are required")
        proposed = copy.deepcopy(self.channels)
        diagnostics = {}
        for role in ROLES:
            channel = proposed[role]
            updates = envelopes[role]
            expected = {
                f"{self.run_id}:{self.version + 1}:{source}:{role}"
                for source in channel.cohort
            }
            if {u.update_id for u in updates} != expected:
                raise ValueError("Update IDs must bind run, round, source and role")
            if any(u.update_id != f"{self.run_id}:{self.version + 1}:{u.source}:{role}"
                   for u in updates):
                raise ValueError("Each update ID must bind its own source and role")
            base = {k: v.copy() for k, v in channel.tensors.items()}
            proposal = channel.aggregate(updates)
            if self.strategy == "clipped":
                variants = aggregate_variants(
                    base, updates, channel.cohort, channel.schema,
                    channel.revisions, self.thresholds[role],
                )
                proposal = variants["proposals"]["clipped"]
                diagnostics[role] = variants["diagnostics"]
                channel.tensors = proposal
            else:
                diagnostics[role] = {"base_hash": model_hash(base),
                                     "proposal_hash": model_hash(proposal)}
        self.channels = proposed
        self.version += 1
        self.pending = True
        self.history.append({"round": self.version, "strategy": self.strategy,
                             "channels": diagnostics, "acknowledgements": None})
        return self.bases()

    def acknowledge(self, metrics):
        if not self.pending or set(metrics) != set(self.channels[ROLES[0]].cohort):
            raise ValueError("Full registered cohort acknowledgement required")
        # Fail before mutating if metrics contain NaN or non-JSON objects.
        json.dumps(metrics, allow_nan=False)
        self.history[-1]["acknowledgements"] = copy.deepcopy(metrics)
        self.pending = False

    def save(self, path):
        """One atomic NPZ archive, no pickle. Includes consumed IDs and phase."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": 1, "version": self.version, "pending": self.pending,
            "strategy": self.strategy, "run_id": self.run_id,
            "thresholds": self.thresholds, "history": self.history,
            "channels": {},
        }
        arrays = {}
        for role, channel in self.channels.items():
            metadata["channels"][role] = {
                "cohort": channel.cohort, "schema": channel.schema,
                "revisions": channel.revisions, "consumed": sorted(channel.consumed),
                "version": channel.version, "hash": model_hash(channel.tensors),
                "keys": sorted(channel.tensors),
            }
            arrays.update({f"{role}/{k}": v for k, v in channel.tensors.items()})
        arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta["format"] != 1 or set(meta["channels"]) != set(ROLES):
                raise ValueError("Unknown coordinator checkpoint")
            bases = {r: {k: archive[f"{r}/{k}"].copy()
                         for k in meta["channels"][r]["keys"]} for r in ROLES}
        first = meta["channels"][ROLES[0]]
        obj = cls(bases, first["cohort"],
                  {r: meta["channels"][r]["schema"] for r in ROLES},
                  first["revisions"], strategy=meta["strategy"],
                  thresholds=meta["thresholds"], run_id=meta["run_id"])
        for role, channel in obj.channels.items():
            state = meta["channels"][role]
            if (state["cohort"] != first["cohort"] or
                    state["revisions"] != first["revisions"] or
                    model_hash(bases[role]) != state["hash"] or
                    state["version"] != meta["version"]):
                raise ValueError("Inconsistent coordinator checkpoint")
            channel.consumed = set(state["consumed"])
            channel.version = state["version"]
        obj.version, obj.pending = meta["version"], meta["pending"]
        obj.history = meta["history"]
        if len(obj.history) != obj.version:
            raise ValueError("Missing round history")
        return obj


class TensorWire:
    """Only known trunk tensors cross the Flower transport boundary."""
    def __init__(self, bases):
        self.layout = [(r, k, v.shape, v.dtype) for r in ROLES
                       for k, v in sorted(bases[r].items())]

    def encode(self, values):
        if set(values) != set(ROLES) or any(
            set(values[r]) != {k for role, k, _, _ in self.layout if role == r}
            for r in ROLES
        ):
            raise ValueError("Unexpected tensors on wire")
        arrays = [values[r][k].copy() for r, k, _, _ in self.layout]
        self.decode(arrays)
        return arrays

    def decode(self, arrays):
        if len(arrays) != len(self.layout):
            raise ValueError("Wrong tensor count")
        result = {r: {} for r in ROLES}
        for value, (role, key, shape, dtype) in zip(arrays, self.layout):
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError("Invalid wire tensor")
            result[role][key] = value.copy()
        return result
