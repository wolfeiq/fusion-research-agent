"""Synchronous equal-client critic-trunk deltas; separate from legacy FedBuff."""
from dataclasses import dataclass
import hashlib
import numpy as np


def model_hash(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        a = np.ascontiguousarray(value)
        h.update(f"{name}:{a.shape}:{a.dtype}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class RepresentationDelta:
    update_id: str
    source: str
    base_hash: str
    schema: str
    revision: str
    tensors: dict


class UniformRepresentationServer:
    def __init__(self, tensors, cohort, schema, revisions):
        self.tensors = {k: v.copy() for k, v in tensors.items()}
        self.cohort = tuple(cohort)
        if len(set(cohort)) != len(cohort) or len(cohort) < 2:
            raise ValueError("Need at least two distinct contributors")
        if not tensors or not all(np.isfinite(v).all() for v in tensors.values()):
            raise ValueError("Invalid base")
        if set(revisions) != set(cohort):
            raise ValueError("Register every source revision")
        self.schema, self.revisions = schema, dict(revisions)
        self.consumed = set()
        self.version = 0

    def aggregate(self, updates):
        """Validate the entire round before mutating anything. No partial cohort."""
        if len(updates) != len(self.cohort):
            raise ValueError("Incomplete cohort")
        if {u.source for u in updates} != set(self.cohort):
            raise ValueError("Duplicate or unknown source")
        ids = [u.update_id for u in updates]
        if len(set(ids)) != len(ids) or set(ids) & self.consumed:
            raise ValueError("Duplicate delivery")
        base = model_hash(self.tensors)
        for u in updates:
            if u.base_hash != base or u.schema != self.schema:
                raise ValueError("Base or schema mismatch")
            if u.revision != self.revisions[u.source]:
                raise ValueError("Source configuration changed")
            if set(u.tensors) != set(self.tensors):
                raise ValueError("Both registered critic branches required")
            for k, v in u.tensors.items():
                if v.shape != self.tensors[k].shape or v.dtype != self.tensors[k].dtype:
                    raise ValueError("Tensor shape/dtype mismatch")
                if not np.isfinite(v).all():
                    raise ValueError("Nonfinite delta")
        new = {}
        with np.errstate(over="ignore", invalid="ignore"):
            for k, v in self.tensors.items():
                # Accumulate in float64 and divide before summing: the sum of
                # finite deltas can overflow even when their mean is finite.
                mean = np.zeros(v.shape, dtype=np.result_type(v.dtype, np.float64))
                for u in updates:
                    mean += u.tensors[k].astype(mean.dtype) / len(updates)
                if not np.isfinite(mean).all():
                    raise ValueError("Nonfinite aggregate")
                # Retain the protocol's delta rounding before base addition.
                # Ordinary float32 rounds still install float32 tensors.
                if np.issubdtype(v.dtype, np.inexact):
                    mean = mean.astype(v.dtype)
                new[k] = v + mean
                if not np.isfinite(new[k]).all():
                    raise ValueError("Nonfinite aggregate")
        self.tensors = new
        self.consumed.update(ids)
        self.version += 1
        return {k: v.copy() for k, v in new.items()}
