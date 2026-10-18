"""Deterministic federation diagnostics and non-executable next-run contracts.

Inputs are captured coordinator dossiers, never mutable learners. Source norms
can assess clipping activation, but cannot reconstruct update directions or
predict the performance of a different controller.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

ROLES = ("heating_td3", "deposition_sac")
BRANCHES = ("q0", "q1")
SCHEMA = "fusion-federation-dossier-v1"
CONTROLS = (
    "local",
    "reset_only",
    "initial_base_pull",
    "raw_federated",
    "deployed",
    "joint_sac",
    "calibration_fixed_command",
)
CHANGES = {
    "clipping_initial_norm_fraction": (
        ("config", "clipping_initial_norm_fraction"),
        "programmatic_configuration_only",
    ),
    "heating_critic_coordinates": (
        ("config", "heating_critic_coordinates"),
        "runner_cli_supported",
    ),
    "td3_mix": (("learner_configs", "td3", "mix"), "requires_runner_change"),
    "sac_mix": (("learner_configs", "sac", "mix"), "requires_runner_change"),
    "td3_actor_lr": (("learner_configs", "td3", "actor_lr"), "requires_runner_change"),
    "sac_actor_lr": (("learner_configs", "sac", "actor_lr"), "requires_runner_change"),
    "sac_alpha_lr": (("learner_configs", "sac", "alpha_lr"), "requires_runner_change"),
}


def dossier(source):
    if not isinstance(source, dict) or source.get("kind") != "experiment":
        raise ValueError("Select a captured federation dossier")
    content = source.get("content")
    if not isinstance(content, str) or hashlib.sha256(
        content.encode()
    ).hexdigest() != source.get("sha256"):
        raise ValueError("Federation snapshot integrity check failed")
    value = json.loads(content)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("This source is not a federation runtime dossier")
    if not isinstance(value.get("manifest"), dict) or not isinstance(
        value.get("jobs"), list
    ):
        raise ValueError("Federation dossier lacks a manifest or jobs")  # noqa: TRY004 - public validation error
    return value


def _number(value, name, *, minimum=0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def compare_clipping(source, job_id, round_number, threshold_scale):
    """Exact norm-cap arithmetic; explicitly not tensor reaggregation or RL."""
    value = dossier(source)
    if type(threshold_scale) not in (int, float) or threshold_scale not in (0.5, 1, 2):
        raise ValueError("Choose a threshold scale of 0.5, 1 or 2")
    if type(round_number) is not int or round_number < 1:
        raise ValueError("Choose a recorded positive round")
    jobs = [job for job in value["jobs"] if job.get("id") == job_id]
    if len(jobs) != 1:
        raise ValueError("Choose a job ID in the selected dossier")
    rounds = [row for row in jobs[0]["rounds"] if row.get("round") == round_number]
    if len(rounds) != 1:
        raise ValueError("This round is not recorded in the selected job")
    row = rounds[0]
    roles = {}
    for role in ROLES:
        channel = row.get("channels", {}).get(role, {})
        norms, thresholds = channel.get("source_norms"), channel.get("thresholds")
        weights = channel.get("weights")
        if (
            not isinstance(norms, dict)
            or not isinstance(thresholds, dict)
            or not isinstance(weights, dict)
        ):
            roles[role] = {
                "status": "unavailable",
                "reason": "This round did not export per-source norms, thresholds and weights. Aggregate tensors cannot recover donor deltas.",
            }
            continue
        if (
            len(norms) < 2
            or set(norms) != set(weights)
            or set(thresholds) != set(BRANCHES)
        ):
            raise ValueError("Incomplete recorded clipping diagnostics")
        weights = {
            name: _number(weight, "source weight", maximum=1)
            for name, weight in weights.items()
        }
        if not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
            raise ValueError("Recorded weights must sum to one")
        if any(
            not math.isclose(weight, 1 / len(weights), abs_tol=1e-9)
            for weight in weights.values()
        ):
            raise ValueError(
                "This comparison requires the registered equal-weight policy"
            )
        if any(
            not isinstance(pair, dict) or set(pair) != set(BRANCHES)
            for pair in norms.values()
        ):
            raise ValueError("Both recorded source branches are required")
        branches = {}
        for branch in BRANCHES:
            threshold = _number(thresholds[branch], "recorded threshold")
            candidate_threshold = _number(threshold * threshold_scale, "candidate threshold")
            sources = {}
            for name, pair in norms.items():
                norm = _number(pair[branch], "source norm")
                sources[name] = {
                    "norm": norm,
                    "weight": weights[name],
                    "recorded_factor": min(1.0, threshold / norm) if norm else 1.0,
                    "candidate_factor": min(1.0, candidate_threshold / norm)
                    if norm
                    else 1.0,
                }
            branches[branch] = {
                "recorded_threshold": threshold,
                "candidate_threshold": candidate_threshold,
                "sources": sources,
                "recorded_clipped_count": sum(
                    s["recorded_factor"] < 1 for s in sources.values()
                ),
                "candidate_clipped_count": sum(
                    s["candidate_factor"] < 1 for s in sources.values()
                ),
                "uniform_mean_norm_upper_bound": sum(
                    s["weight"] * s["norm"] for s in sources.values()
                ),
                "candidate_mean_norm_upper_bound": sum(
                    s["weight"] * s["norm"] * s["candidate_factor"]
                    for s in sources.values()
                ),
            }
        hashes = channel.get("proposal_hashes", {})
        equal = (
            hashes["uniform"] == hashes["clipped"]
            if isinstance(hashes.get("uniform"), str)
            and isinstance(hashes.get("clipped"), str)
            else None
        )
        roles[role] = {
            "status": "computed_from_recorded_norms",
            "branches": branches,
            "recorded_clipped_source_branches": sum(
                b["recorded_clipped_count"] for b in branches.values()
            ),
            "candidate_clipped_source_branches": sum(
                b["candidate_clipped_count"] for b in branches.values()
            ),
            "total_source_branches": len(norms) * len(BRANCHES),
            "recorded_uniform_equals_clipped": equal,
        }
    return {
        "schema": "fusion-clipping-inspection-v1",
        "source_id": source["id"],
        "source_sha256": source["sha256"],
        "manifest_identity": value["manifest"].get("identity"),
        "job_id": job_id,
        "round": round_number,
        "threshold_scale": threshold_scale,
        "scope": "norm_only_counterfactual; no tensor reaggregation, rollout or predicted efficacy",
        "roles": roles,
        "limitations": [
            "Individual update directions were not exported. Bounds are not realized aggregate norms.",
            "A change in clipping activation does not demonstrate a benefit for TD3, SAC or cold start.",
            "Candidate thresholds are development hypotheses; a new protocol must freeze them before new target results.",
        ],
    }


def draft_policy_trial(source, change, new_value):
    """Bind one bounded proposed change to a saved manifest, with prerequisites."""
    value = dossier(source)
    if change not in CHANGES:
        raise ValueError("Choose one of the explicitly supported policy trial fields")
    if change == "heating_critic_coordinates":
        if new_value not in ("physical", "normalized"):
            raise ValueError("Heating coordinates must be physical or normalized")
    elif change.endswith("_mix"):
        _number(new_value, "mix", maximum=1)
    elif change.endswith("_lr"):
        _number(new_value, "learning rate", minimum=1e-6, maximum=1e-2)
    else:
        _number(new_value, "clipping fraction", minimum=0.001, maximum=1)
    path, support = CHANGES[change]
    manifest = value["manifest"]
    previous = manifest
    recorded = True
    for key in path:
        if not isinstance(previous, dict) or key not in previous:
            previous, recorded = None, False
            break
        previous = previous[key]
    if recorded and previous == new_value:
        raise ValueError(
            "The proposed value equals the recorded value; choose a real controlled change"
        )
    prerequisites = [
        "Human review and a newly frozen development manifest; do not apply this to the recorded or active run.",
        "Preserve paired seeds, initial states, donor budgets, target shots and evaluation conditions across baseline and candidate.",
        "Retain raw federation outcomes separately from the combined-controller gate fallback.",
        "Evaluate TD3 and SAC together; inspect both beta_N tracking and temperature-profile error.",
        "Do not use confirmation outcomes to choose this candidate.",
    ]
    if support == "requires_runner_change":
        prerequisites.append(
            "The current runner fixes this learner field in learner_configs; expose it, test it and include it in configuration/schema hashes before launch."
        )
    elif support == "programmatic_configuration_only":
        prerequisites.append(
            "StudyConfig accepts this field, but the current CLI does not expose it. Add a tested CLI option or use an explicitly frozen programmatic configuration."
        )
    if not recorded:
        prerequisites.append(
            "The old manifest did not record this field. Establish the source implementation's effective baseline before comparing."
        )
    result = {
        "schema": "fusion-next-policy-trial-v1",
        "status": "proposal_only",
        "source_id": source["id"],
        "source_sha256": source["sha256"],
        "manifest_identity": manifest.get("identity"),
        "change": {
            "field": change,
            "json_pointer": "/" + "/".join(path),
            "before_recorded": recorded,
            "before": previous,
            "candidate": new_value,
        },
        "implementation_status": support,
        "executable": False,
        "applied": False,
        "controls": list(CONTROLS),
        "frozen_fields": {
            name: copy.deepcopy(manifest.get("config", {}).get(name))
            for name in (
                "seeds",
                "rounds",
                "episodes_per_round",
                "donor_pretrain_episodes",
                "facility_factors",
                "gate_factors",
                "report_factors",
                "beta_mae_tolerance",
                "beta_coverage",
                "profile_rmse_tolerance_keV",
            )
        },
        "budget": {
            "recorded_matrix": copy.deepcopy(manifest.get("expected_budget", {})),
            "candidate_budget_status": "not_recomputed",
            "requirement": "Recompute the whole baseline-and-candidate matrix before launch, including donor training, target shots, validation, repair optimizer work and prior calibration.",
        },
        "prerequisites": prerequisites,
        "stop_rule": "Stop on nonfinite learning or invalid simulation; preserve failures. Reject deployment candidates failing the frozen combined-controller gate.",
        "scientific_status": "Unvalidated development hypothesis; no predicted improvement or permission to launch.",
    }
    result["candidate_id"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    return result
