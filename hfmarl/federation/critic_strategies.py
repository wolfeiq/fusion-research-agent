"""Pure critic-delta strategy comparisons for matched-client development runs.

Every variant uses the same validated source envelopes and common base. These
functions do not mutate a live server or track deliveries across calls; the
resumable experiment must persist completed operations and immutable input IDs.
No target trajectories or target evaluation scores enter calibration.
"""
import math

import numpy as np

from hfmarl.federation.representation import UniformRepresentationServer, model_hash


BRANCHES = ("q0", "q1")


def _norm(values):
    """A float64 norm with scaling to avoid squaring large finite entries."""
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    scale = max((float(np.max(np.abs(value))) for value in arrays if value.size), default=0.)
    if not math.isfinite(scale):
        raise ValueError("Nonfinite branch values")
    if scale == 0:
        return 0.
    squared = sum(float(np.sum((value / scale) ** 2, dtype=np.float64)) for value in arrays)
    norm = scale * math.sqrt(squared)
    if not math.isfinite(norm):
        raise FloatingPointError("Branch norm exceeds finite float64 range")
    return norm


def _validate(base, updates, cohort, schema, revisions):
    if not base:
        raise ValueError("Both nonempty critic branches are required")
    names = {branch: [] for branch in BRANCHES}
    for name, value in base.items():
        branch = name.split(".", 1)[0]
        if branch not in names or "." not in name:
            raise ValueError("Only q0 and q1 critic-trunk tensors may be shared")
        if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
            raise ValueError("Critic base tensors must be floating NumPy arrays")
        if value.size == 0:
            raise ValueError("Critic tensors must not be empty")
        names[branch].append(name)
    if any(not branch_names for branch_names in names.values()):
        raise ValueError("Both critic branches are required")
    for branch_names in names.values():
        branch_names.sort()
    updates = list(updates)
    cohort = tuple(cohort)
    # Reuse the protocol's full-cohort, duplicate, base, schema, revision,
    # dtype, shape and finite checks on an isolated server. No input is changed.
    server = UniformRepresentationServer(base, cohort, schema, revisions)
    server.aggregate(updates)
    by_source = {update.source: update for update in updates}
    ordered = [by_source[source] for source in cohort]
    source_norms = {
        update.source: {branch: _norm(update.tensors[name] for name in branch_names)
                        for branch, branch_names in names.items()}
        for update in ordered
    }
    diagnostics = {
        "base_hash": model_hash(base), "schema": schema,
        "source_ids": list(cohort), "update_ids": [update.update_id for update in ordered],
        "revisions": dict(revisions), "source_norms": source_norms,
        "weights": {source: 1. / len(cohort) for source in cohort},
        "arithmetic": "branch norms and delta means float64; proposal cast to base tensor dtype",
    }
    return ordered, names, diagnostics


def calibrate_thresholds(base, updates, cohort, schema, revisions):
    """Return per-branch median source norms and JSON-safe provenance.

    Call this with the declared independent source-only calibration cohort.
    Statistical independence is an experiment-design obligation, not inferable
    from tensor values. Zero thresholds are retained and explicitly reported.
    """
    ordered, _, diagnostics = _validate(base, updates, cohort, schema, revisions)
    thresholds = {}
    for branch in BRANCHES:
        norms = sorted(diagnostics["source_norms"][update.source][branch] for update in ordered)
        middle = len(norms) // 2
        # Average halves, avoiding overflow when two individually finite norms
        # are large. No target-derived floor or threshold adjustment is applied.
        thresholds[branch] = (norms[middle] if len(norms) % 2
                              else norms[middle - 1] / 2. + norms[middle] / 2.)
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise FloatingPointError("Nonfinite calibration threshold")
    diagnostics.update({"method": "per_branch_median_source_delta_norm",
                        "target_data_used": False,
                        "zero_threshold_branches": [k for k, v in thresholds.items() if v == 0]})
    return {"thresholds": thresholds, "diagnostics": diagnostics}


def _thresholds(values):
    if set(values) != set(BRANCHES):
        raise ValueError("One frozen threshold for each of q0 and q1 is required")
    result = {}
    for branch, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise ValueError("Clipping thresholds must be finite nonnegative numbers")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Clipping thresholds must be finite and nonnegative")
        result[branch] = value
    return result


def _proposal(base, delta):
    result = {}
    for name, value in base.items():
        # Conversion back to the protocol's dtype must not hide overflow.
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = np.asarray(value, dtype=np.float64) + delta[name]
            result[name] = candidate.astype(value.dtype)
        if not np.isfinite(candidate).all() or not np.isfinite(result[name]).all():
            raise FloatingPointError("Nonfinite strategy proposal")
    return result


def aggregate_variants(base, updates, cohort, schema, revisions, thresholds):
    """Return uniform, clipped, and norm-matched-uniform proposals and diagnostics.

    Each proposal is a full common-base representation for the recipient's fixed
    mixing/repair procedure. Clipping acts jointly on all tensors in each branch.
    Norm matching preserves the uniform direction and may amplify it. When a zero
    uniform mean cannot match a nonzero clipped mean, that branch stays at the base
    and is explicitly marked unavailable, rather than claiming a matched control.
    """
    thresholds = _thresholds(thresholds)
    ordered, names, diagnostics = _validate(base, updates, cohort, schema, revisions)
    count = len(ordered)
    factors = {}
    for update in ordered:
        factors[update.source] = {}
        for branch in BRANCHES:
            norm = diagnostics["source_norms"][update.source][branch]
            factors[update.source][branch] = thresholds[branch] / norm if norm > thresholds[branch] else 1.
    uniform = {name: np.zeros_like(value, dtype=np.float64) for name, value in base.items()}
    clipped = {name: np.zeros_like(value, dtype=np.float64) for name, value in base.items()}
    for update in ordered:
        for branch, branch_names in names.items():
            for name in branch_names:
                delta = np.asarray(update.tensors[name], dtype=np.float64)
                uniform[name] += delta / count
                clipped[name] += (delta * factors[update.source][branch]) / count
    mean_norms = {"uniform": {}, "clipped": {}}
    matched, match_factors, available, amplified = {}, {}, {}, {}
    for branch, branch_names in names.items():
        u_norm = _norm(uniform[name] for name in branch_names)
        c_norm = _norm(clipped[name] for name in branch_names)
        mean_norms["uniform"][branch] = u_norm
        mean_norms["clipped"][branch] = c_norm
        available[branch] = u_norm > 0 or c_norm == 0
        factor = c_norm / u_norm if u_norm > 0 else (0. if c_norm == 0 else None)
        if factor is not None and not math.isfinite(factor):
            raise FloatingPointError("Nonfinite norm-matching multiplier")
        match_factors[branch] = factor
        amplified[branch] = bool(factor is not None and factor > 1.)
        for name in branch_names:
            matched[name] = np.zeros_like(uniform[name]) if factor is None else uniform[name] * factor
    proposals = {name: _proposal(base, delta) for name, delta in
                 (("uniform", uniform), ("clipped", clipped), ("norm_matched_uniform", matched))}
    # Record realized deltas after casting too; parameter rounding can erase an
    # extremely small update or slightly perturb an otherwise exact norm match.
    realized_norms = {variant: {branch: _norm(
        proposal[name].astype(np.float64) - base[name].astype(np.float64) for name in branch_names)
        for branch, branch_names in names.items()} for variant, proposal in proposals.items()}
    diagnostics.update({
        "thresholds": thresholds,
        "clipping_factors": factors,
        "clipped_sources": {branch: [source for source in diagnostics["source_ids"]
                                      if factors[source][branch] < 1.] for branch in BRANCHES},
        "uniform_mean_norms": mean_norms["uniform"],
        "clipped_mean_norms": mean_norms["clipped"],
        "norm_match_factors": match_factors,
        "norm_match_available": available,
        "norm_match_amplified": amplified,
        "realized_delta_norms": realized_norms,
        "proposal_hashes": {name: model_hash(proposal) for name, proposal in proposals.items()},
    })
    return {"proposals": proposals, "diagnostics": diagnostics}
